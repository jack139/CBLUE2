#! -*- coding: utf-8 -*-
# SimCSE 中文测试

import sys
import json
from tqdm import tqdm
import numpy as np
import scipy.stats
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open
from bert4keras.snippets import sequence_padding
from keras.models import Model


# 基本参数
#model_type, pooling, task_name, dropout_rate = sys.argv[1:]
pooling = 'cls'
assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']
dropout_rate = 0.3

maxlen = 256
batch_size = 16
epochs = 1


# 加载数据集
data_path = 'data'

def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    for l in json.load(open(filename)):
        D.append((l['text1'], l['text2'], int(l['label'])))
    return D


# 模型路径
config_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# 建立模型
def get_encoder(
    config_path,
    checkpoint_path,
    model='bert',
    pooling='first-last-avg',
    dropout_rate=0.1
):
    """建立编码器
    """
    assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']

    if pooling == 'pooler':
        bert = build_transformer_model(
            config_path,
            checkpoint_path,
            model=model,
            with_pool='linear',
            dropout_rate=dropout_rate
        )
    else:
        bert = build_transformer_model(
            config_path,
            checkpoint_path,
            model=model,
            dropout_rate=dropout_rate
        )

    outputs, count = [], 0
    while True:
        try:
            output = bert.get_layer(
                'Transformer-%d-FeedForward-Norm' % count
            ).output
            outputs.append(output)
            count += 1
        except:
            break

    if pooling == 'first-last-avg':
        outputs = [
            keras.layers.GlobalAveragePooling1D()(outputs[0]),
            keras.layers.GlobalAveragePooling1D()(outputs[-1])
        ]
        output = keras.layers.Average()(outputs)
    elif pooling == 'last-avg':
        output = keras.layers.GlobalAveragePooling1D()(outputs[-1])
    elif pooling == 'cls':
        output = keras.layers.Lambda(lambda x: x[:, 0])(outputs[-1])
    elif pooling == 'pooler':
        output = bert.output

    # 最后的编码器
    encoder = Model(bert.inputs, output)
    return encoder


encoder = get_encoder(
    config_path,
    checkpoint_path,
    pooling=pooling,
    dropout_rate=dropout_rate
)


# 语料id化
def convert_to_ids(data, tokenizer, maxlen=64):
    """转换文本数据为id形式
    """
    a_token_ids, b_token_ids, labels = [], [], []
    for d in tqdm(data):
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    return a_token_ids, b_token_ids, labels



class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids = []


def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)

class Evaluator(keras.callbacks.Callback):
    """保存验证集分数最好的模型
    """
    def __init__(self):
        self.best_val_score = 0.

    def on_epoch_end(self, epoch, logs=None):
        encoder.save_weights('IR_simcse_e%03d.weights' % epoch)


def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def text2vec(text):
    token_ids = tokenizer.encode(text, maxlen=maxlen)[0]
    x_vecs = encoder.predict([[token_ids], np.zeros_like([token_ids])])
    x_vecs = l2_normalize(x_vecs)
    return x_vecs[0]


if __name__ == '__main__':

    # 准备数据
    datasets = {
        f : load_data('%s/%s.json' % (data_path, f)) for f in ['train', 'dev']
    }

    all_names, all_weights, all_token_ids, all_labels = [], [], [], []
    train_token_ids = []
    for name, data in datasets.items():
        a_token_ids, b_token_ids, labels = convert_to_ids(data, tokenizer, maxlen)
        all_names.append(name)
        all_weights.append(len(data))
        all_token_ids.append((a_token_ids, b_token_ids))
        all_labels.append(labels)
        train_token_ids.extend(a_token_ids)
        train_token_ids.extend(b_token_ids)

    # SimCSE训练
    encoder.summary()
    encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
    train_generator = data_generator(train_token_ids, batch_size)

    evaluator = Evaluator()

    encoder.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )


    # 语料向量化
    all_vecs = []
    for a_token_ids, b_token_ids in all_token_ids:
        a_vecs = encoder.predict([a_token_ids,
                                  np.zeros_like(a_token_ids)],
                                 verbose=True)
        b_vecs = encoder.predict([b_token_ids,
                                  np.zeros_like(b_token_ids)],
                                 verbose=True)
        all_vecs.append((a_vecs, b_vecs))

    # 标准化，相似度，相关系数
    all_corrcoefs = []
    for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
        a_vecs = l2_normalize(a_vecs)
        b_vecs = l2_normalize(b_vecs)
        sims = (a_vecs * b_vecs).sum(axis=1)
        corrcoef = compute_corrcoef(labels, sims)
        all_corrcoefs.append(corrcoef)

    all_corrcoefs.extend([
        np.average(all_corrcoefs),
        np.average(all_corrcoefs, weights=all_weights)
    ])

    for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
        print('%s: %s' % (name, corrcoef))

    '''
    train: 0.7091344058221786
    dev: 0.7462282029574684
    avg: 0.7276813043898235
    w-avg: 0.7095016747766761
    '''
else:

    encoder.load_weights('IR_simcse_e000.weights')
