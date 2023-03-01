#!/usr/bin/python

import math
import numpy as np
from sklearn.datasets import load_digits

from keras import backend as K
import  tensorflow as tf

from keras.optimizers import SGD, Adam
from keras.layers import Dense

from keras.layers import Input, Lambda
from keras.models import Model

from neural_tensor_layer import NeuralTensorLayer


def get_data():
    digits = load_digits()
    L = int(math.floor(digits.data.shape[0] * 0.15))
    X_train = digits.data[:L]
    y_train = digits.target[:L]
    X_test = digits.data[L + 1:]
    y_test = digits.target[L + 1:]
    return X_train, y_train, X_test, y_test

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
    temp1 = K.max(tc - t + 1, 0)
    temp1 = K.sum(temp1)
    temp2 = K.sqrt(sum([K.sum(K.square(var)) for var in tf.trainable_variables()]))
    temp = temp1 + (regularization * temp2)
    return temp

input1 = Input(shape=(64,), dtype='float32') # 正例 e1
input2 = Input(shape=(64,), dtype='float32') # 正例 e2
input3 = Input(shape=(64,), dtype='float32') # 负例 e3
btp_pos = NeuralTensorLayer(output_dim=32, input_dim=64)([input1, input2])
btp_neg = NeuralTensorLayer(output_dim=32, input_dim=64)([input1, input3])
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
    optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
    #metrics=['acc'],
)
train_model.summary()

# 训练集
X_train, Y_train, X_test, Y_test = get_data()
X_train = X_train.astype(np.float32)
X_train2 = X_train.copy()
np.random.shuffle(X_train2) # 随机顺序当做负例
Y_train = Y_train.astype(np.float32)

# 测试集
X_test = X_test.astype(np.float32)
X_test2 = X_test.copy()
np.random.shuffle(X_test2) # 随机顺序当做负例
Y_test = Y_test.astype(np.float32)

def train():
    # 训练
    train_model.fit(
        [X_train, X_train, X_train2],
        Y_train, 
        epochs=40, 
        batch_size=5
    )

    # 测试
    score = train_model.evaluate(
        [X_test, X_test, X_test2], 
        Y_test, 
        batch_size=5
    )

    print(score)

    model.save_weights('ntn_example.weights')


if __name__ == '__main__':

    if 1:
        train()
    else:
        model.load_weights('ntn_example.weights')

    #print(K.get_value(model.layers[2].W))

    y = model.predict([X_test[:10], X_test[:10]], batch_size=1, verbose=1)
    #print(y)
    #print(np.sum(y, axis=1))
    a1 = np.average(y, axis=1)

    y = model.predict([X_test[:10], X_test[20:30]], batch_size=1, verbose=1)
    #print(y)
    #print(np.sum(y, axis=1))
    a2 = np.average(y, axis=1)

    print(Y_test[:10])
    print(Y_test[20:30])

    print(a1)
    print(a2)
    print(a1>a2)


    
