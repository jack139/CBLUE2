import json
import datetime,time
import os
import copy
import tensorflow as tf
#os.environ["RECOMPUTE"] = '0'

import pickle
from bert4keras.backend import keras, K, batch_gather
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, SpTokenizer
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.layers import Loss

from keras.layers import Lambda, Dense, Input, Permute, Activation
from keras.models import Model

import numpy as np
#from tqdm.notebook import tqdm
from .i2c import i2c

projectName = 'MedDG_next_entities: 修改输入数据'

# MedBERT
config_path = '../../nlp_model/PCL-MedBERT-wwm/bert_config.json'
checkpoint_path = '../../nlp_model/PCL-MedBERT-wwm/bert_model.ckpt'
dict_path = '../../nlp_model/PCL-MedBERT-wwm/vocab.txt'

# BERT base
#config_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
#checkpoint_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
#dict_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

maxlen = 512
epochs = 100
batch_size = 16
steps_per_epoch = 66258 // batch_size + 1 # 实际训练集数量(最大) 66258

pp = 2
skip_epochs = 50

append_entities_len = 40

learning_rate = 8e-6 # 2e-5 1e-5 8e-6


c2i = { v:idx  for idx, v in enumerate(i2c) }
c2c = { v[0]:v  for v in i2c }


data_path = "./data/"

import pickle

with open(os.path.join(data_path, "train_data.pk"), "rb") as f:
    train_data = pickle.load(f)
    
with open(os.path.join(data_path, "dev_data.pk"), "rb") as f:
    dev_data = pickle.load(f)

print(len(train_data))
print(len(dev_data))

def create_content_label(data_item):
    new_dialog = []
    history = []
    now_topic = []
    his_topic = []
    for sen in data_item:
        aa = sen['Symptom']+sen['Attribute']+sen['Test']+sen['Disease']+sen['Medicine']
        if len(aa) > 0:
            if len(history) > 0 and sen['id'] == 'Doctor':
                new_dialog.append({"history": copy.deepcopy(history), "next_sym": copy.deepcopy(aa), 'now_topic': copy.deepcopy(now_topic)})
            now_topic.extend(aa)
            his_topic.extend(aa)
        history.append(sen['Sentence'])
    for dic in new_dialog:
        future = copy.deepcopy(his_topic[len(dic['now_topic']):])
        dic['future'] = future
        content = ''.join(dic['history'])
        append_entities = set()
        for v in dic['now_topic']:
            append_entities.add(c2c[v])
        label = {
            "Symptom": set(),
            "Medicine": set(),
            "Test": set(),
            "Attribute": set(),
            "Disease": set()
        }
        for v in dic['next_sym']:
            label[c2c[v][1]].add(v)
        yield content, append_entities, label


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        
        batch_token_ids, batch_segment_ids, batch_append_entity_ids = [], [], []
        batch_labels = []
        
        prob = 1.0

        for is_end, data_item in self.sample(random):
            
            for content, append_entities, label in create_content_label(data_item):
                
                #if sum([len(v) for v in label.values()]) == 0:
                #    if np.random.uniform() < prob:
                #        continue
                        
                #print(content)
                #print(label)
                #print(append_entities)
                
                # change prob
                #prob -= (prob / epochs / skip_epochs)

                token_ids, _ = tokenizer.encode(content)
                token_ids = token_ids[1:-1]

                if len(token_ids) > maxlen - 2:
                    token_ids = token_ids[-maxlen+2:]

                token_ids = [tokenizer.token_to_id("[CLS]")] + token_ids + [tokenizer.token_to_id("[SEP]")]
                token_ids += [0] * (maxlen - len(token_ids))

                batch_token_ids.append(token_ids)
                batch_segment_ids.append([0] * maxlen)

                # 增加出现实体
                append_entity_ids = [c2i[item] for item in append_entities if item in c2i]
                append_entity_ids += [0] * (append_entities_len - len(append_entity_ids))

                batch_append_entity_ids.append(append_entity_ids)

                # 生成答案
                label_item = [2 for _ in i2c]

                for k, v in label.items():
                    for v_item in v:
                        label_item[c2i[(v_item, k)]] = 1
                
                batch_labels.append(label_item)
                
                if len(batch_token_ids) == self.batch_size or is_end:

                    yield {
                        'Input-Token': np.array(batch_token_ids),
                        'Input-Segment': np.array(batch_segment_ids),
                        'Input-Append-Entities': np.array(batch_append_entity_ids),
                        'Output-Label-Id': np.array(batch_labels),
                    },{
                        'Circle-Loss': np.zeros((len(batch_token_ids),)),
                    }

                    batch_token_ids, batch_segment_ids, batch_append_entity_ids = [], [], []
                    batch_labels = []

# 模型开始

from bert4keras.layers import Layer, Embedding, Add

class MaskMean(Layer):
    
    def __init__(self, **kwargs):
        super(MaskMean, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(MaskMean, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def compute_mask(self, inputs, mask):
        return None

    def call(self, inputs, mask):

        tokens = inputs
        mask = K.expand_dims(K.cast(mask, dtype="float"), axis=-1)
        
        return K.sum(tokens*mask, axis=1) / K.sum(mask, axis=1)

# 手动定义loss, 以及acc

# y_true 0 mask 1 正例 2 负例
def mult_circle_loss(inputs, mask=None):
    
    y_true, y_pred = inputs
    zeros = K.zeros_like(y_pred[..., :1])
    
    y_true_p = K.cast(K.equal(y_true, 1), K.floatx())
    y_true_n = K.cast(K.equal(y_true, 2), K.floatx())
    
    y_pred_p = -y_pred + (1 - y_true_p) * -1e12
    y_pred_n = y_pred + (1 - y_true_n) * -1e12

    y_pred_p = K.concatenate([y_pred_p, zeros], axis=-1)
    y_pred_n = K.concatenate([y_pred_n, zeros], axis=-1)

    p_loss = tf.reduce_logsumexp(y_pred_p, axis=-1)
    n_loss = tf.reduce_logsumexp(y_pred_n, axis=-1)
    
    return pp * p_loss + n_loss

bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

mean_output = MaskMean(name="Mask-Mean")(bert.model.output)
    
final_output = Dense(
    units=int(mean_output.shape[-1]), 
    kernel_initializer=bert.initializer, 
    activation="tanh",
    name="Label-Tanh"
)(mean_output)

final_output = Dense(
    units=len(i2c), 
    kernel_initializer=bert.initializer, 
    name="Label-Id"
)(final_output)

# 增加转移概率

input_append_entity_ids = Input(shape=(append_entities_len, ), name='Input-Append-Entities')

append_entity_represent = Embedding(
    input_dim=len(i2c),
    output_dim=len(i2c),
    embeddings_initializer=bert.initializer,
    mask_zero=True,
    name='Entities-Trans'
)(input_append_entity_ids)


append_entity_represent = MaskMean(name="Entities-Mean")(append_entity_represent)

# 加和
final_output = Add(name="Final-Add")([final_output, append_entity_represent])


final_input = Input(shape=(len(i2c), ), name='Output-Label-Id')

type_loss = Lambda(mult_circle_loss, name='Circle-Loss')([final_input, final_output])

train_loss = {
    'Circle-Loss': lambda y_true, y_pred: y_pred
}
    

model = Model(bert.model.inputs + [input_append_entity_ids], final_output)
train_model = Model(bert.model.inputs + [input_append_entity_ids, final_input], type_loss)
    
optimizer = Adam(learning_rate=learning_rate)
train_model.compile(loss=train_loss, optimizer=optimizer)

train_model.summary()


def evaluate(data):

    batch_token_ids, batch_segment_ids, batch_append_entity_ids = [], [], []
    
    true_labels = []

    for data_item in data:

        for content, append_entities, label in create_content_label(data_item):

            token_ids, _ = tokenizer.encode(content)
            token_ids = token_ids[1:-1]

            if len(token_ids) > maxlen - 2:
                token_ids = token_ids[-maxlen+2:]

            token_ids = [tokenizer.token_to_id("[CLS]")] + token_ids + [tokenizer.token_to_id("[SEP]")]
            token_ids += [0] * (maxlen - len(token_ids))

            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * maxlen)
            
            true_labels.append(label)

            # 增加出现实体
            append_entity_ids = [c2i[item] for item in append_entities if item in c2i]
            append_entity_ids += [0] * (append_entities_len - len(append_entity_ids))

            batch_append_entity_ids.append(append_entity_ids)


    y = model.predict([batch_token_ids, batch_segment_ids, batch_append_entity_ids], batch_size=batch_size, verbose=1)

    predict_data = []

    for idx in range(len(batch_token_ids)):
        
        predict_data_item = {k:set() for e, k in i2c}
        
        for item_idx, item in enumerate(y[idx]):
            if item > 0:
                e, t = i2c[item_idx]
                predict_data_item[t].add(e)
                
        predict_data.append(predict_data_item)
        
    score = {k:{'p': 0, 's': 0, 'i':0} for e, k in i2c}

    for predict_data_item, true_data_item in zip(predict_data, true_labels):

        for label, pred_entity in  predict_data_item.items():
            if label == 'None': 
                score[label]['p'] += len(pred_entity)
                #score[label]['s'] += len(true_data_item[label])
                score[label]['i'] += len(pred_entity)
            else:
                score[label]['p'] += len(pred_entity)
                score[label]['s'] += len(true_data_item[label])
                score[label]['i'] += len(true_data_item[label] & pred_entity)

    all_p, all_s, all_i = 0, 0, 0

    for k,v in score.items():
        all_p += v['p']
        all_s += v['s']
        all_i += v['i']

        P = v['i'] / v['p'] if v['p'] != 0 else 0
        R = v['i'] / v['s'] if v['s'] != 0 else 0 
        f1 = 2 * P * R / (P + R) if P + R != 0 else 0 

        print(k + " P: %.2f R: %.2f F1: %.2f" % (P*100, R*100, f1*100))

    P = all_i / all_p if all_p != 0 else 0
    R = all_i / all_s if all_s != 0 else 0 
    f1 = 2 * P * R / (P + R) if P + R != 0 else 0 

    print("ALL P: %.2f R: %.2f F1: %.2f" % (P*100, R*100, f1*100))

    return f1

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model_saved_path):
        
        self.best_bleu = 0.
        self.model_saved_path = model_saved_path

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(dev_data)

        if metrics > self.best_bleu:
            self.best_bleu = metrics
            model.save_weights(os.path.join(self.model_saved_path, 'MedDG_next2_MedBert_f1_%.5f.weights'%metrics))

if __name__ == '__main__':
    print(projectName + ' Train...')
    resultPath = './data/outputs'
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    train_generator = data_generator(train_data, batch_size)

    model.load_weights(os.path.join(resultPath, "MedDG_next2_MedBert_f1_0.21818.weights"))

    train_model.fit(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[Evaluator(resultPath)]
    )