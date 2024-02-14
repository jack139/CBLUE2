#! -*- coding:utf-8 -*-
# 三元组抽取任务，基于“半指针-半标注”结构
# 数据集：CBLUE 2 IMCS-SR
# 说明：由于使用了EMA，需要跑足够多的步数(5000步以上）才生效，如果
#      你的数据总量比较少，那么请务必跑足够多的epoch数，或者去掉EMA。

import sys
import json
import numpy as np
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import Loss
from bert4keras.layers import LayerNormalization
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from tqdm import tqdm
from norms import sym2id, id2sym

maxlen = 256
batch_size = 32
learning_rate = 1e-4 # 1-e4, 3e-5, 2e-5

config_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object'])
                             for spo in l['spo_list']]
            })
    return D


# 加载数据集
train_data = load_data('./data/train_no_blank.jsonl')
valid_data = load_data('./data/dev.jsonl')

#predicate2id, id2predicate = {}, {}
#for l in norms:
#    if l not in predicate2id:
#        id2predicate[len(predicate2id)] = l
#        predicate2id[l] = len(predicate2id)

predicate2id = sym2id # id2sym 的 key 是字符串，不是整数， 要与之前兼容
id2predicate = {}
for k in predicate2id.keys():
    id2predicate[predicate2id[k]] = k

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回所有下标；否则返回 []。
    """
    n = len(pattern)
    pos = []
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            pos.append(i)
    return pos


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=maxlen)
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o, in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != [] and o_idx != []:
                    # 处理 多个 subject 情况，只保留，离 value最近的
                    minp = abs(s_idx[0]-o_idx[0])
                    minx = s_idx[0]
                    for s_pos in s_idx:
                        if abs(s_pos-o_idx[0])<minp:
                            minp = abs(s_pos-o_idx[0])
                            minx = s_pos
                    s_idx = minx

                    o_idx = o_idx[0]  # o_idx 只对比第1个， ？？？可能会误判

                    # 截取
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject（这里没有实现错误！这就是想要的效果！！）
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(
                        batch_subject_labels
                    )
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_object_labels)
                    yield [
                        batch_token_ids, batch_segment_ids,
                        batch_subject_labels, batch_subject_ids,
                        batch_object_labels
                    ], None
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


def extract_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]


# 补充输入
subject_labels = Input(shape=(None, 2), name='Subject-Labels')
subject_ids = Input(shape=(2,), name='Subject-Ids')
object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels')

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

# 预测subject
output = Dense(
    units=2, activation='sigmoid', kernel_initializer=bert.initializer
)(bert.model.output)
subject_preds = Lambda(lambda x: x**2)(output)

subject_model = Model(bert.model.inputs, subject_preds)

# 传入subject，预测object
# 通过Conditional Layer Normalization将subject融入到object的预测中
output = bert.model.layers[-2].get_output_at(-1)  # 自己想为什么是-2而不是-1
subject = Lambda(extract_subject)([output, subject_ids])
output = LayerNormalization(conditional=True)([output, subject])
output = Dense(
    units=len(predicate2id) * 2,
    activation='sigmoid',
    kernel_initializer=bert.initializer
)(output)
output = Lambda(lambda x: x**4)(output)
object_preds = Reshape((-1, len(predicate2id), 2))(output)

object_model = Model(bert.model.inputs + [subject_ids], object_preds)


class TotalLoss(Loss):
    """subject_loss与object_loss之和，都是二分类交叉熵
    """
    def compute_loss(self, inputs, mask=None):
        subject_labels, object_labels = inputs[:2]
        subject_preds, object_preds, _ = inputs[2:]
        if mask[4] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[4], K.floatx())
        # sujuect部分loss
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        # object部分loss
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        # 总的loss
        return subject_loss + object_loss


subject_preds, object_preds = TotalLoss([2, 3])([
    subject_labels, object_labels, subject_preds, object_preds,
    bert.model.output
])

# 训练模型
train_model = Model(
    bert.model.inputs + [subject_labels, subject_ids, object_labels],
    [subject_preds, object_preds]
)

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learning_rate=learning_rate)
train_model.compile(optimizer=optimizer)


def extract_spoes(text, return_s_idx=False):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    # 抽取subject
    subject_preds = subject_model.predict([token_ids, segment_ids])
    subject_preds[:, [0, -1]] *= 0
    start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        object_preds[:, [0, -1]] *= 0
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.6)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1,
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        if return_s_idx: # 返回 subject 的位置，用于最后还原结果
            return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1], s[0])
                    for s, p, o, in spoes]
        else:
            return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                    for s, p, o, in spoes]            
    else:
        return []


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(f'f1: {f1:.5f}, precision: {precision:.5f}, recall: {recall:.5f}')
        s = json.dumps({
                'text': d['text'],
                'spo_list': list(T),
                'spo_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            },
           ensure_ascii=False,
           indent=4
        )
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        optimizer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            train_model.save_weights(f'sr-v2_best_f1_{f1:.5f}.weights')
        optimizer.reset_old_weights()
        print(f'f1: {f1:.5f}, precision: {precision:.5f}, recall: {recall:.5f}, best f1: {self.best_val_f1:.5f}')


def predict_to_file(in_file, out_file):
    """预测到文件
    """
    D = {}
    data = json.load(open(in_file))

    for d in tqdm(data.keys(), ncols=100):
        D[d] = {}
        D[d]['global'] = {}

        for l in data[d]['dialogue']:
            D[d][l['sentence_id']] = {}
            text = '[unused1][unused2][unused3]'+l['sentence']

            spo_list = {}
            R = set([SPO(spo) for spo in extract_spoes(text, return_s_idx=True)])
            for e in R:
                core_name = e[0]+'@'+str(e[3])
                if core_name not in spo_list.keys():
                    spo_list[core_name] = [{
                        'predicate': e[1],
                        'object': e[2],
                    }]
                else:
                    spo_list[core_name].append({
                        'predicate': e[1],
                        'object': e[2],
                    })

            # spo_list里应该有多条，需要合并
            for core_name in spo_list.keys():

                _core_name = core_name.split('@')[0]

                event = {
                    "entity": _core_name,
                    "symptom_norm": "",
                    "symptom_type": "",
                }

                for e in spo_list[core_name]:
                    if e['object'][:9]=='[unused1]':
                        symptom_type = "0"
                    elif e['object'][:9]=='[unused2]':
                        symptom_type = "1"
                    elif e['object'][:9]=='[unused3]':
                        symptom_type = "2"
                    else:
                        symptom_type = "UNKNOWN: "+e['object']

                    ## 测试用
                    #D[d][l['sentence_id']] = { e['predicate'] : (_core_name, symptom_type) }

                    ## 生成提交结果
                    D[d][l['sentence_id']][e['predicate']] = symptom_type
                    D[d]['global'][e['predicate']] = symptom_type

    json.dump(
        D,
        open(out_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


if __name__ == '__main__':

    #train_model.load_weights('sr_best_f1_0.59188.weights')

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    train_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=50,
        callbacks=[evaluator]
    )

else:
    train_model.load_weights('sr-v2_best_f1_0.63732.weights')
    predict_to_file('../dataset/3.0/IMCS-V2/IMCS-V2_test.json', 'IMCS-V2-SR_test.json')
