#! -*- coding:utf-8 -*-
# 句子对分类任务

import numpy as np
from bert4keras.backend import keras, K, search_layer
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dense, Lambda
from tqdm import tqdm
import json

import jieba
jieba.initialize()

num_classes = 4
maxlen = 128
batch_size = 16

# base
#config_path = '../../nlp_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
#checkpoint_path = '../../nlp_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
#dict_path = '../../nlp_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

# large
config_path = '../../nlp_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    for l in json.load(open(filename)):
        D.append((l['query'], l['title'], int(l['label'])))
    return D


# 加载数据集
train_data = load_data('../dataset/KUAKE-QTR/KUAKE-QTR_train.json')
valid_data = load_data('../dataset/KUAKE-QTR/KUAKE-QTR_dev.json')


# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1 + ' ' + text2, maxlen=maxlen
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
    units=num_classes,
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
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)



def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('qtr_gp_best_roberta_acc_%.5f.weights'%val_acc)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


def predict_to_file(in_file, out_file):
    """预测到文件
    """
    data = json.load(open(in_file))
    for l in tqdm(data, ncols=100):
        text1 = l['query']
        text2 = l['title']
        token_ids, segment_ids = tokenizer.encode(text1 + ' ' + text2, maxlen=maxlen)
        label = model.predict([[token_ids], [segment_ids]])[0].argmax()
        l['label'] = str(label)

    json.dump(
        data,
        open(out_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=30,
        callbacks=[evaluator]
    )

else:
    model.load_weights('../ckpt/qtr_gp_best_model_acc_0.69104.weights')
    predict_to_file('../dataset/KUAKE-QTR/KUAKE-QTR_test.json', 'KUAKE-QTR_test.json')
