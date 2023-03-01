#! -*- coding: utf-8 -*-
# 简单的线性变换（白化）操作

import os
import json, pickle
import numpy as np
import pandas as pd
import scipy.stats
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding#, open 
from keras.models import Model
from tqdm import tqdm

maxlen = 128

def load_data(filename):
    """加载数据 csv
    单条格式：text1,text2,label
    """
    outputs = {'text1': [], 'text2': [], 'label': []}
    train_cache_df = pd.read_csv(filename)
    outputs['text1'] = train_cache_df['text1'].values.tolist()
    outputs['text2'] = train_cache_df['text2'].values.tolist()
    outputs['label'] = train_cache_df['label'].values.tolist()

    D = []
    for text1, text2, label in zip(outputs['text1'], outputs['text2'], outputs['label']):
        D.append((text1, text2, float(label)))

    return D


# bert配置
config_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立模型
bert = build_transformer_model(config_path, checkpoint_path)

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

# first-last-avg
outputs = [
    keras.layers.GlobalAveragePooling1D()(outputs[0]),
    keras.layers.GlobalAveragePooling1D()(outputs[-1])
]
output = keras.layers.Average()(outputs)

# 最后的编码器
encoder = Model(bert.inputs, output)

# 变换矩阵和偏置项
with open("data/kernel_bias.pk", "rb") as f:
    kernel, bias = pickle.load(f)

def convert_to_vecs(data, maxlen=64):
    """转换文本数据为id形式
    """
    a_token_ids, b_token_ids, labels = [], [], []
    for d in data:
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    a_vecs = encoder.predict([a_token_ids,
                              np.zeros_like(a_token_ids)],
                             verbose=False)
    b_vecs = encoder.predict([b_token_ids,
                              np.zeros_like(b_token_ids)],
                             verbose=False)
    return a_vecs, b_vecs, np.array(labels)


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def get_sim(text1, text2):
    # 语料向量化 - 计算
    a_vecs, b_vecs, labels = convert_to_vecs([[text1, text2, 0]], maxlen=maxlen)

    # 变换，标准化，相似度
    a_vecs = transform_and_normalize(a_vecs, kernel, bias)
    b_vecs = transform_and_normalize(b_vecs, kernel, bias)
    sims = (a_vecs * b_vecs).sum(axis=1)

    return sims


if __name__ == '__main__':
    print(get_sim('糖尿病反复低血糖', '糖尿病性低血糖症'))
    print(get_sim('高血压冠心病不稳定心绞痛', '臀部良性肿瘤'))
    print(get_sim('1型糖尿病性植物神经病变', '1型糖尿病性自主神经病'))
    print(get_sim('双侧额叶脑出血术后', '额叶交界性肿瘤'))

    '''
    test_D = load_data('data/CHIP-CDN/train_ntn.csv')

    p = n = 0
    for text1, text2, label in tqdm(test_D):
        if text2=='[BLANK]':
            continue
        s = get_sim(text1, text2)
        label_ = 0. if s<0.59 else 1.
        if label_==label:
            p += 1
        n += 1

    print('acc= %.5f'%(p/n))
    '''