#! -*- coding: utf-8 -*-
# 简单的线性变换（白化）操作, 分析测试 

import os
import json, pickle
import numpy as np
import scipy.stats
#from bert4keras.backend import keras, K
#from bert4keras.tokenizers import Tokenizer
#from bert4keras.models import build_transformer_model
#from bert4keras.snippets import sequence_padding#, open 
#from keras.models import Model

'''
def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    for l in json.load(open(filename)):
        D.append((l['text1'], l['text2'], float(l['label'])))
    return D

# 加载数据集
datasets = {
    'sts-b-train': load_data('../dataset/CHIP-STS/CHIP-STS_train.json'),
    'sts-b-test': load_data('../dataset/CHIP-STS/CHIP-STS_dev.json')
}

# bert配置
config_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化
    """
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)


# 建立模型
bert = build_transformer_model(config_path, checkpoint_path)

encoder_layers, count = [], 0
while True:
    try:
        output = bert.get_layer(
            'Transformer-%d-FeedForward-Norm' % count
        ).output
        encoder_layers.append(output)
        count += 1
    except:
        break

n_last, outputs = 2, []
for i in range(n_last):
    outputs.append(GlobalAveragePooling1D()(encoder_layers[-i]))

output = keras.layers.Average()(outputs)

# 最后的编码器
encoder = Model(bert.inputs, output)


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
                             verbose=True)
    b_vecs = encoder.predict([b_token_ids,
                              np.zeros_like(b_token_ids)],
                             verbose=True)
    return a_vecs, b_vecs, np.array(labels)
'''

def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    # return None, None
    # return W, -mu
    return W[:, :256], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


# 语料向量化
#cache_file = "data/whitening_cache.pk"
cache_file = "data/whitening_cache_first_last_avg.pk"
if os.path.exists(cache_file): # 从文件装入
    with open(cache_file, "rb") as f:
        all_names, all_weights, all_vecs, all_labels = pickle.load(f)
else:
    '''
    # 语料向量化 - 计算
    all_names, all_weights, all_vecs, all_labels = [], [], [], []
    for name, data in datasets.items():
        a_vecs, b_vecs, labels = convert_to_vecs(data)
        all_names.append(name)
        all_weights.append(len(data))
        all_vecs.append((a_vecs, b_vecs))
        all_labels.append(labels)

    # 保存文件： 测试用
    with open(cache_file, "wb") as f:
        pickle.dump([all_names, all_weights, all_vecs, all_labels], f)
    '''
    raise ValueError('No cache file found!')

# 计算变换矩阵和偏置项
kernel, bias = compute_kernel_bias([v for vecs in all_vecs for v in vecs])

# 变换，标准化，相似度，相关系数
all_corrcoefs = []
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    a_vecs = transform_and_normalize(a_vecs, kernel, bias)
    b_vecs = transform_and_normalize(b_vecs, kernel, bias)
    sims = (a_vecs * b_vecs).sum(axis=1)

    # 相关性
    corrcoef = compute_corrcoef(labels, sims)
    all_corrcoefs.append(corrcoef)

    # 取值范围
    max_min = {0 : [10., -10.], 1 : [10., -10.] }
    for a, b, label, sim in zip(a_vecs, b_vecs, labels, sims):
        label_ = int(label)
        if sim < max_min[label_][0]:
            max_min[label_][0] = sim
        if sim > max_min[label_][1]:
            max_min[label_][1] = sim

    for x in max_min.keys():
        print("%d: %.5f ~ %.5f"%(x, max_min[x][0], max_min[x][1]))

    # 计算阈值（对两个标签的数据适用，多标签数据无效）
    from sklearn.metrics import f1_score, accuracy_score

    distances = sims
    identical = labels

    thresholds = np.arange(-1.0, 1.0, 0.01)

    f1_scores = [f1_score(identical==0., distances < t) for t in thresholds]
    acc_scores = [accuracy_score(identical==0., distances < t) for t in thresholds]

    opt_idx = np.argmax(f1_scores)
    # Threshold at maximal F1 score
    opt_tau = thresholds[opt_idx]
    # Accuracy at maximal F1 score
    opt_acc = accuracy_score(identical==0., distances < opt_tau)

    print("maximal F1 threshold=%.2f\tacc=%.5f"%(opt_tau, opt_acc))

    opt_idx2 = np.argmax(acc_scores)
    # Threshold at maximal ACC score
    opt_tau2 = thresholds[opt_idx2]
    # Accuracy at maximal ACC score
    opt_acc2 = accuracy_score(identical==0., distances < opt_tau2)

    print("maximal ACC threshold=%.2f\tacc=%.5f"%(opt_tau2, opt_acc2))



all_corrcoefs.extend([
    np.average(all_corrcoefs),
    np.average(all_corrcoefs, weights=all_weights)
])

for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    print('%s: %s' % (name, corrcoef))
