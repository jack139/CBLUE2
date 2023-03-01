#! -*- coding:utf-8 -*-
# 通过梯度惩罚增强模型的泛化性能
# 蕴含打分模型

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from keras.layers import Dense, Lambda
from tqdm import tqdm
import json
import tensorflow as tf

# 建立默认session
graph = tf.Graph()  # 解决多线程不同模型时，keras或tensorflow冲突的问题
session = tf.Session(graph=graph)


maxlen = 128
config_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

with graph.as_default():
    with session.as_default():

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
        #model.summary()

        ckpt_weights = './score2_gp_best_model_f1_0.92936.weights'
        model.load_weights(ckpt_weights)
        print('Load weights: ', ckpt_weights)

        # https://stackoverflow.com/questions/40850089/is-keras-thread-safe
        model._make_predict_function() # have to initialize before threading

def predict(text1, text2):
    """预测
    """
    token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
    with graph.as_default(): # 解决多线程不同模型时，keras或tensorflow冲突的问题
        with session.as_default():
            label = model.predict([[token_ids], [segment_ids]]) #[0].argmax()
    return label[:, 1]

if __name__ == '__main__':
    print(
        predict("1型糖尿病性植物神经病变", "1型糖尿病性自主神经病"),
        predict("1型糖尿病性植物神经病变", "1型糖尿病性前期肾病")
    )