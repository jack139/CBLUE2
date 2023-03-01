import os, re
import time
import tensorflow as tf

from bert4keras.tokenizers import Tokenizer, SpTokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.models import build_transformer_model
from bert4keras.backend import keras, K, batch_gather
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_layer_adaptation
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.layers import Loss

from keras.layers import Lambda, Dense, Input, Permute, Activation
from keras.models import Model

import numpy as np
from tqdm import tqdm
import json
import pickle

train_entity = True

projectName = f'MedDG_生成回复_entity=={train_entity}'
if not train_entity:
    corpus_path = './data/t5_base/corpus.train.tfrecord'
else:
    corpus_path = './data/t5_base/corpus.train.bert_entity.tfrecord'
dev_path = './data/dig_dev_data_with_bert_entity.pk'

with open(dev_path, "rb") as f:
    dev_data = pickle.load(f)


# 此预训练参数涉及隐私数据，还在沟通能否放出
init_checkpoint_path = '../../nlp_model/mt5_base/meddg_pretrain.h5'
pretrain_checkpoint_path = '../../nlp_model/mt5_base/model.ckpt-1000000'
config_path = '../../nlp_model/mt5_base/mt5_base_config.json'
spm_path = '../../nlp_model/mt5_base/sentencepiece_cn.model'
keep_tokens_path = '../../nlp_model/mt5_base/sentencepiece_cn_keep_tokens.json'

# t5 small
#init_checkpoint_path = '../../nlp_model/mt5_small/meddg_pretrain.h5'
#pretrain_checkpoint_path = '../../nlp_model/mt5_small/model.ckpt-1000000'
#config_path = '../../nlp_model/mt5_small/mt5_small_config.json'
#spm_path = '../../nlp_model/mt5_small/sentencepiece_cn.model'
#keep_tokens_path = '../../nlp_model/mt5_small/sentencepiece_cn_keep_tokens.json'


tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
keep_tokens = json.load(open(keep_tokens_path))

# 其他配置
max_in_len = 512
max_out_len = 128

learning_rate = 2e-5
weight_decay_rate = 0.01

batch_size = 8
grad_accum_steps = 1  # 大于1即表明使用梯度累积

num_warmup_steps = 0
num_train_steps = 250000
steps_per_epoch = 10000 # 159653 
epochs = num_train_steps * grad_accum_steps // steps_per_epoch
exclude_from_weight_decay = ['Norm', 'bias']
# tpu_address = 'grpc://xxx.xxx.xxx.xxx:8470'  # 如果用多GPU跑，直接设为None
which_optimizer = 'adam'  # adam 或 lamb，均自带weight decay

init_step = 0

lr_schedule = {
    init_step + num_warmup_steps * grad_accum_steps: 1.0,
    init_step + num_train_steps * grad_accum_steps: 0.0,
}

floatx = K.floatx()

def make_dataset(corpus_path):
    dataset = tf.data.TFRecordDataset([corpus_path])
    dataset = dataset.repeat()

    def parse_function(serialized):

        features = {
            'c_token_ids': tf.io.FixedLenFeature([max_in_len], tf.int64),
            't_token_ids': tf.io.FixedLenFeature([max_out_len], tf.int64),
        }

        features = tf.io.parse_single_example(serialized, features)

        return {
            'Encoder-Input-Token': features['c_token_ids'],
            'Decoder-Input-Token': features['t_token_ids']
        }, {
            'mlm_loss': 0.0,
        }

    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(batch_size * 2000)  # 打乱
    dataset = dataset.batch(batch_size)  # 成批

    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    while True:
        yield K.get_session().run(next_batch)

    #return dataset

# 模型
from bert4keras.models import T5_Decoder as T5_Decoder_Old
from bert4keras.models import T5_Base, T5_Encoder
from bert4keras.layers import *

class T5_Decoder(T5_Decoder_Old):
    
    def apply_final_layers(self, inputs):
        """剩余部分
        """
        c, x = inputs
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            #center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Decoder-Output-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Decoder-Output-Dropout'
        )
        
        scale = np.sqrt(self.hidden_size)
        x = self.apply(
            inputs=x,
            layer=Lambda,
            function=lambda x: x / scale,
            mask=lambda i, m: m,
            name='Decoder-Output-Scale'
        )

        if self.with_lm:
            # 预测token概率部分
            if self.embedding_size != self.hidden_size:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.embedding_size,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-Mapping'
                )
            lm_activation = 'softmax' if self.with_lm is True else self.with_lm
            if self.version == 't5.1.0':
                x = self.apply(
                    inputs=x,
                    layer=Embedding,
                    arguments={'mode': 'dense'},
                    name='Embedding-Token'
                )
                x = self.apply(
                    inputs=x,
                    layer=Activation,
                    activation=lm_activation,
                    name='Dencoder-Output-LM-Activation'
                )
            else:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.vocab_size,
                    activation=lm_activation,
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-LM'
                )

        return x
    
class T5(T5_Base):
    """Google的T5模型（Encoder-Decoder）
    """
    def __init__(self, **kwargs):
        super(T5, self).__init__(**kwargs)
        kwargs['layers'] = self.layers
        e_name, d_name = 'Encoder', 'Decoder'
        if 'name' in kwargs:
            e_name = '%s_%s' % (kwargs['name'], e_name)
            d_name = '%s_%s' % (kwargs['name'], d_name)
            del kwargs['name']  # 防止重复传参
        self._encoder = T5_Encoder(name=e_name, **kwargs)
        self._decoder = T5_Decoder(name=d_name, **kwargs)

    def build(self, **kwargs):
        """同时构建Encoder和Decoder
        """
        self._encoder.build(**kwargs)
        self._decoder.build(**kwargs)
        self.encoder = self._encoder.model
        self.decoder = self._decoder.model
        self.inputs = self.encoder.inputs + self.decoder.inputs[1:]
        self.outputs = self.decoder(
            self.encoder.outputs + self.decoder.inputs[1:]
        )
        self.model = Model(self.inputs, self.outputs)


    @classmethod
    def startswith(cls, inputs):
        return False

class MLMLoss(Layer):

    def call(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = K.cast(mask[1], K.floatx())[:, 1:]  # 解码器自带mask
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask, axis=-1) / (K.sum(y_mask, axis=-1) + K.epsilon())
        
        return loss
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0:1]
    
    def compute_mask(self, inputs, mask):
        return None

def build_transformer_model_with_mlm():
    """带mlm的bert模型
    """
    t5 = build_transformer_model(
        config_path=config_path,
        checkpoint_path=pretrain_checkpoint_path,
        keep_tokens=keep_tokens,
        model=T5,
        version='mt5.1.1',  # for bert4keras > 0.10.8
        return_keras_model=False,
        name='T5',
    )

    encoder = t5.encoder
    decoder = t5.decoder
    model = t5.model

    output = MLMLoss(name='mlm_loss')([model.inputs[1], model.outputs[0]])

    train_model = Model(model.inputs, output)

    #def mlm_loss(y_true, y_pred):
    #  return y_pred

    loss = {
        'mlm_loss': lambda y_true, y_pred: y_pred,
    }

    return encoder, decoder, train_model, loss


def build_transformer_model_for_pretraining():
    """构建训练模型，通用于TPU/GPU
    注意全程要用keras标准的层写法，一些比较灵活的“移花接木”式的
    写法可能会在TPU上训练失败。此外，要注意的是TPU并非支持所有
    tensorflow算子，尤其不支持动态（变长）算子，因此编写相应运算
    时要格外留意。
    """
    encoder, decoder, train_model, loss = build_transformer_model_with_mlm()


    # 优化器
    optimizer = extend_with_weight_decay(Adam)
    # 梯度预归一化
    # optimizer = extend_with_grad_norm(optimizer)
    
    if which_optimizer == 'lamb':
        optimizer = extend_with_layer_adaptation(optimizer)
    optimizer = extend_with_piecewise_linear_lr(optimizer)
    optimizer_params = {
        'learning_rate': learning_rate,
        'lr_schedule': lr_schedule,
        'weight_decay_rate': weight_decay_rate,
        'exclude_from_weight_decay': exclude_from_weight_decay,
        # 'exclude_from_layer_adaptation': exclude_from_weight_decay,
        #'bias_correction': True,
    }
    if grad_accum_steps > 1:
        optimizer = extend_with_gradient_accumulation(optimizer)
        optimizer_params['grad_accum_steps'] = grad_accum_steps

    optimizer = optimizer(**optimizer_params)

    # 模型定型
    train_model.compile(loss=loss, optimizer=optimizer)

    #train_model.load_weights(init_checkpoint_path)

    return train_model, encoder, decoder


#strategy = tf.distribute.MirroredStrategy()  # 建立单机多卡策略
#with strategy.scope():  # 调用该策略
#    train_model, encoder, decoder = build_transformer_model_for_pretraining()
#    train_model.summary()

train_model, encoder, decoder = build_transformer_model_for_pretraining()
train_model.summary()


# 校验
from bert4keras.snippets import AutoRegressiveDecoder

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        c_encoded = inputs[0]
        return self.last_token(decoder).predict([c_encoded, output_ids])

    def generate(self, c_tokens, topk=1):
        c_token_ids = tokenizer.tokens_to_ids(c_tokens)
        c_encoded = encoder.predict(np.array([c_token_ids]))[0]
        output_ids = self.beam_search([c_encoded], topk=topk)  # 基于beam search
        return tokenizer.decode([int(i) for i in output_ids])

autotitle = AutoTitle(start_id=0, end_id=tokenizer._token_end_id, maxlen=max_out_len)


def get_c_s(data_item):
    
    c_tokens = []

    for item in data_item['history'][::-1]:
        item = tokenizer._tokenize(item)

        if len(c_tokens) + len(item) > max_in_len - 1:
            break

        c_tokens = item + c_tokens


    c_tokens = c_tokens + ["</s>"]

#     print(" ".join(c_tokens))
#     print("--------------------------")

    # 回复为当前医生的回答
    t_tokens = tokenizer._tokenize(data_item['response'])

    if len(t_tokens) > max_out_len - 2:
        t_tokens = t_tokens[:max_out_len - 2]

    return c_tokens, t_tokens


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate(data, topk=1):

    smooth = SmoothingFunction().method1

    total = 0
    bleu, bleu_1, bleu_4 = 0, 0, 0
    for data_item in tqdm(data[:2000]): # 只评估2000个，bleu计算耗时
        try:
          c_tokens, t_tokens = get_c_s(data_item)
            
          total += 1
          reply = ' '.join(tokenizer.decode(tokenizer.tokens_to_ids(t_tokens))).lower()
          pred_reply = ' '.join(autotitle.generate(c_tokens, topk)).lower()
          
          if pred_reply.strip() and reply.strip():
#                     scores = self.rouge.get_scores(hyps=pred_reply, refs=reply)
#                     rouge_1 += scores[0]['rouge-1']['f']
#                     rouge_2 += scores[0]['rouge-2']['f']
#                     rouge_l += scores[0]['rouge-l']['f']

              bleu += sentence_bleu(
                  references=[reply.split(' ')],
                  hypothesis=pred_reply.split(' '),
                  smoothing_function=smooth
              )

              bleu_1 += sentence_bleu(
                  references=[reply.split(' ')],
                  hypothesis=pred_reply.split(' '),
                  weights=(1, 0, 0, 0)
              )

              bleu_4 += sentence_bleu(
                  references=[reply.split(' ')],
                  hypothesis=pred_reply.split(' '),
                  weights=(0, 0, 0, 1)
              )

                
                
        except Exception:
            print(data_item)
            print(pred_reply)
            print(reply)


    bleu_1 /= total
    bleu_4 /= total
    bleu /= total
    return {
        'bleu_1': bleu_1,
        'bleu_4': bleu_4,
        'bleu': bleu,
    }


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model_saved_path):
        
        self.best_bleu = 0.
        self.model_saved_path = model_saved_path

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(dev_data)['bleu']

        if metrics > self.best_bleu:
            self.best_bleu = metrics
            train_model.save_weights(os.path.join(self.model_saved_path, 
                'MedDG_gen_base_%smT5_bleu_%.5f.weights'%('entity_' if train_entity else '', metrics)))  # 保存模型

        print("bleu= %.5f, best_bleu= %.5f"%(metrics, self.best_bleu))


if __name__ == '__main__':
    print(projectName + ' Train...')
    resultPath = './data/outputs'
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    evaluator = Evaluator(resultPath)

    dataset = make_dataset(corpus_path)

    train_model.load_weights(os.path.join(resultPath, "MedDG_gen_base_entity_mT5_bleu_0.13120.weights"))

    # 模型训练
    train_model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[evaluator],
    )