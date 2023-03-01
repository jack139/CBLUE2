#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别
# 数据集 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open, to_array
from keras.models import Model
import tensorflow as tf

# 建立默认session
graph = tf.Graph()  # 解决多线程不同模型时，keras或tensorflow冲突的问题
session = tf.Session(graph=graph)

maxlen = 256
categories = set(['bod', 'dep', 'dis', 'dru', 'equ', 'ite', 'mic', 'pro', 'sym'])
categories = list(sorted(categories))

# bert配置
config_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

with graph.as_default():
    with session.as_default():

        model = build_transformer_model(config_path, checkpoint_path)
        output = GlobalPointer(len(categories), 64)(model.output)

        model = Model(model.input, output)
        #model.summary()

        ckpt_weights = '../ckpt/cmeee_plus_best_globalpointer_f1_0.66070.weights'
        model.load_weights(ckpt_weights)
        print('Load weights: ', ckpt_weights)

        # https://stackoverflow.com/questions/40850089/is-keras-thread-safe
        model._make_predict_function() # have to initialize before threading

class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=maxlen)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        with graph.as_default(): # 解决多线程不同模型时，keras或tensorflow冲突的问题
            with session.as_default():
                scores = model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[l])
            )
        return entities

NER = NamedEntityRecognizer()


if __name__ == '__main__':
    print(NER.recognize("左膝退变伴游离体"))
