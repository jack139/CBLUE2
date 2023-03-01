#!/usr/bin/python

import pickle
import math
import numpy as np
from sklearn.datasets import load_digits

from keras import backend as K
import tensorflow as tf

from keras.optimizers import SGD, Adam
from keras.layers import Dense

from keras.layers import Input, Lambda
from keras.models import Model

from neural_tensor_layer import NeuralTensorLayer

maxlen = 128
input_dim = 768

def get_data():
    with open("data/train_ntn_embedding.pk", "rb") as f:
        train_data = pickle.load(f)

    X_e1, X_e2, X_e3 = [], [], []

    
    for i in train_data:
        e1, e2 = np.array(i[0]), np.array(i[1])

        e3 = e1.copy()
        np.random.shuffle(e3) # 随机顺序当做负例

        # 按列求和
        e1 = np.sum(e1, axis=0)
        e2 = np.sum(e2, axis=0)
        e3 = np.sum(e3, axis=0)

        # 转为一维向量
        '''
        x, y = e1.shape
        e1 = np.concatenate((e1, np.zeros([maxlen-x, y])),axis=0)
        e1 = e1.reshape(maxlen*y)

        x, y = e2.shape
        e2 = np.concatenate((e2, np.zeros([maxlen-x, y])),axis=0)
        e2 = e2.reshape(maxlen*y)

        x, y = e3.shape
        e3 = np.concatenate((e3, np.zeros([maxlen-x, y])),axis=0)
        e3 = e3.reshape(maxlen*y)
        '''

        X_e1.append(e1)
        X_e2.append(e2)
        X_e3.append(e3)

    Y = [0]*len(train_data)
    Y = np.array(Y)

    # 拆分训练集
    L = int(math.floor(len(X_e1) * 0.1))
    X_train_e1 = X_e1[:L]
    X_train_e2 = X_e2[:L]
    X_train_e3 = X_e3[:L]
    y_train = Y[:L]
    X_test_e1 = X_e1[L + 1:]
    X_test_e2 = X_e2[L + 1:]
    X_test_e3 = X_e3[L + 1:]
    y_test = Y[L + 1:]

    return np.array([X_train_e1, X_train_e2, X_train_e3]), y_train, \
        np.array([X_test_e1, X_test_e2, X_test_e3]), y_test


'''
# loss from tf implementation
def loss(predictions, regularization = 0.0001):
    print("Beginning building loss")
    temp1 = tf.maximum(tf.sub(predictions[1, :], predictions[0, :]) + 1, 0)
    temp1 = tf.reduce_sum(temp1)
    temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))
    temp = temp1 + (regularization * temp2)
    return temp
'''

def max_margin_loss(inputs, regularization = 0.0001):
    #print("Beginning building loss")
    t, tc = inputs
    temp1 = tf.maximum(tf.subtract(tc, t) + 1, 0)
    temp1 = tf.reduce_sum(temp1)
    temp2 = tf.sqrt(sum([tf.cast(tf.reduce_sum(tf.square(var)), tf.float32) for var in tf.trainable_variables()]))
    temp = temp1 + (regularization * temp2)
    return temp


input1 = Input(shape=(input_dim,), dtype='float32') # 正例 e1
input2 = Input(shape=(input_dim,), dtype='float32') # 正例 e2
input3 = Input(shape=(input_dim,), dtype='float32') # 负例 e3
btp_pos = NeuralTensorLayer(output_dim=32, input_dim=input_dim)([input1, input2])
btp_neg = NeuralTensorLayer(output_dim=32, input_dim=input_dim)([input1, input3])
#output = Dense(output_dim=1)(btp_pos)

margin_loss = Lambda(max_margin_loss, name='Margin-Loss')([btp_pos, btp_neg])

train_loss = {
    'Margin-Loss': lambda y_true, y_pred: y_pred
}

model = Model([input1, input2], btp_pos)
train_model = Model(input=[input1, input2, input3], output=margin_loss)

train_model.compile(
    loss=train_loss,
    #optimizer=Adam(2e-3),
    optimizer=SGD(lr=2e-3, decay=1e-6, momentum=0.9, nesterov=True),
    #metrics=['acc'],
)
train_model.summary()

# 训练集
X_train, Y_train, X_test, Y_test = get_data()
X_train = X_train.astype(np.float32)
Y_train = Y_train.astype(np.float32)
# 测试集
X_test = X_test.astype(np.float32)
Y_test = Y_test.astype(np.float32)


if __name__ == '__main__':

    # 训练
    train_model.fit(
        [X_train[0], X_train[1], X_train[2]],
        Y_train, 
        epochs=50, 
        batch_size=5
    )

    # 测试
    score = train_model.evaluate(
        [X_test[0], X_test[1], X_test[2]], 
        Y_test, 
        batch_size=5
    )

    print(score)

    model.save_weights('data/ntn_embeddings.weights')

    #print(K.get_value(model.layers[2].W))

    y = model.predict([X_test[0][:5], X_test[1][:5]], batch_size=1, verbose=1)
    print(y)
    print([np.average(i) for i in y])

    y = model.predict([X_test[0][:5], X_test[2][:5]], batch_size=1, verbose=1)
    print(y)
    print([np.average(i) for i in y])

else:
    pass
    #model.load_weights('ntn_embeddings.weights')
