#! -*- coding:utf-8 -*-
# 通过梯度惩罚增强模型的泛化性能
# 蕴含打分模型

import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dense, Lambda
from sklearn.metrics import f1_score
from tqdm import tqdm
import json

maxlen = 128
batch_size = 8
config_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    for l in json.load(open(filename)):
        D.append((l['text1'], l['text2'], int(l['label'])))
    return D

# 加载数据集
train_data = load_data('data/train_score_samples2.json')
valid_data = load_data('data/dev_score_samples2.json')


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=2,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
    return loss + 0.5 * epsilon * gp

model.compile(
    loss=loss_with_gradient_penalty,
    optimizer=Adam(2e-6),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)



def evaluate(data):
    f1_true = np.array([], 'int32')
    f1_pred = np.array([], 'int32')
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        f1_true = np.append(f1_true, y_true)
        f1_pred = np.append(f1_pred, y_pred)
    return f1_score(f1_true, f1_pred, average='macro')

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = evaluate(valid_generator)
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            model.save_weights('score3_gp_best_model_f1_%.5f.weights'%val_f1)
        print(
            u'val_f1: %.5f, best_val_f1: %.5f\n' %
            (val_f1, self.best_val_f1)
        )


def predict_to_file(in_file, out_file):
    """预测到文件
    """
    f2 = open(out_file, 'w', encoding='utf-8')
    with open(in_file, encoding='utf-8') as f:
        for l in tqdm(f, ncols=100):
            d = json.loads(l)

            text1 = d['text1']
            text2 = d['text2']
            token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            d['label'] = str(label)

            s = json.dumps(d, ensure_ascii=False)
            f2.write(s + '\n')
    f2.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    #model.load_weights('score3_gp_best_model_f1_0.92038.weights')
    
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=30,
        callbacks=[evaluator]
    )

else:
    model.load_weights('best_model.weights')
    predict_to_file('data/score_test.json', 'score_test.json')
