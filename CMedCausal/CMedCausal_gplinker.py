#! -*- coding:utf-8 -*-
# 三元组抽取任务，基于GlobalPointer的仿TPLinker设计
# 文章介绍：https://kexue.fm/archives/8888
# 数据集：http://ai.baidu.com/broad/download?dataset=sked
# 最优f1=0.827
# 说明：由于使用了EMA，需要跑足够多的步数(5000步以上）才生效，如果
#      你的数据总量比较少，那么请务必跑足够多的epoch数，或者去掉EMA。

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import sparse_multilabel_categorical_crossentropy
from bert4keras.tokenizers import Tokenizer
from bert4keras.layers import EfficientGlobalPointer as GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from tqdm import tqdm

learning_rate=2e-5
maxlen = 512
batch_size = 12
config_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o, o2, s_idx, o_idx, o2_idx)]}
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object'], spo['object2'], 
                              spo['s_idx'], spo['o_idx'], spo['o2_idx'])
                             for spo in l['spo_list']]
            })
    return D


# 加载数据集
train_data = load_data('data/train.jsonl')
valid_data = load_data('data/dev.jsonl')
predicate2id = {
    '1' : 0,
    '2' : 1,
    '3' : 2,
}
id2predicate = {
    0 : '1',
    1 : '2',
    2 : '3',
}


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回所有下标；否则返回[]。
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
        batch_entity_labels, batch_head_labels, batch_tail_labels, batch_head_labels2, batch_tail_labels2 = [], [], [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=maxlen)
            # 整理三元组 {(s, o, p)}
            spoes = set()
            for s, p, o, o2, s_n, o_n, o2_n in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                o2 = tokenizer.encode(o2)[0][1:-1]
                sh = search(s, token_ids)
                oh = search(o, token_ids)
                o2h = search(o2, token_ids)
                if sh != [] and oh != [] and o2h != []:
                    sh = sh[s_n]
                    st = sh + len(s) - 1
                    oh = oh[o_n]
                    ot = oh + len(o) - 1
                    o2h = o2h[o2_n]
                    o2t = o2h + len(o2) - 1
                    spoes.add((sh, st, p, oh, ot, o2h, o2t))

            # 构建标签
            entity_labels = [set() for _ in range(3)]
            head_labels = [set() for _ in range(len(predicate2id))]
            tail_labels = [set() for _ in range(len(predicate2id))]
            head_labels2 = [set() for _ in range(len(predicate2id))]
            tail_labels2 = [set() for _ in range(len(predicate2id))]
            for sh, st, p, oh, ot, o2h, o2t in spoes:
                entity_labels[0].add((sh, st))
                entity_labels[1].add((oh, ot))
                entity_labels[2].add((o2h, o2t))
                head_labels[p].add((sh, oh))
                tail_labels[p].add((st, ot))
                head_labels2[p].add((sh, o2h))
                tail_labels2[p].add((st, o2t))
            for label in entity_labels + head_labels + tail_labels + head_labels2 + tail_labels2:
                if not label:  # 至少要有一个标签
                    label.add((0, 0))  # 如果没有则用0填充
            entity_labels = sequence_padding([list(l) for l in entity_labels])
            head_labels = sequence_padding([list(l) for l in head_labels])
            tail_labels = sequence_padding([list(l) for l in tail_labels])
            head_labels2 = sequence_padding([list(l) for l in head_labels2])
            tail_labels2 = sequence_padding([list(l) for l in tail_labels2])
            # 构建batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_head_labels2.append(head_labels2)
            batch_tail_labels2.append(tail_labels2)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_entity_labels = sequence_padding(
                    batch_entity_labels, seq_dims=2
                )
                batch_head_labels = sequence_padding(
                    batch_head_labels, seq_dims=2
                )
                batch_tail_labels = sequence_padding(
                    batch_tail_labels, seq_dims=2
                )
                batch_head_labels2 = sequence_padding(
                    batch_head_labels2, seq_dims=2
                )
                batch_tail_labels2 = sequence_padding(
                    batch_tail_labels2, seq_dims=2
                )
                yield [batch_token_ids, batch_segment_ids], [
                    batch_entity_labels, batch_head_labels, batch_tail_labels, batch_head_labels2, batch_tail_labels2
                ]
                batch_token_ids, batch_segment_ids = [], []
                batch_entity_labels, batch_head_labels, batch_tail_labels, batch_head_labels2, batch_tail_labels2 = [], [], [], [], []


def globalpointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    shape = K.shape(y_pred)
    y_true = y_true[..., 0] * K.cast(shape[2], K.floatx()) + y_true[..., 1]
    y_pred = K.reshape(y_pred, (shape[0], -1, K.prod(shape[2:])))
    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, True)
    return K.mean(K.sum(loss, axis=1))


# 加载预训练模型
base = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False
)

# 预测结果
entity_output = GlobalPointer(heads=3, head_size=64)(base.model.output)
head_output = GlobalPointer(
    heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
)(base.model.output)
tail_output = GlobalPointer(
    heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
)(base.model.output)
head_output2 = GlobalPointer(
    heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
)(base.model.output)
tail_output2 = GlobalPointer(
    heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
)(base.model.output)
outputs = [entity_output, head_output, tail_output, head_output2, tail_output2]

# 构建模型
AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learning_rate=learning_rate)
model = keras.models.Model(base.model.inputs, outputs)
model.compile(loss=globalpointer_crossentropy, optimizer=optimizer)
model.summary()


def extract_spoes(text, threshold=0):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    outputs = model.predict([token_ids, segment_ids])
    outputs = [o[0] for o in outputs]
    # 抽取subject和object
    subjects, objects, objects2 = set(), set(), set()
    outputs[0][:, [0, -1]] -= np.inf
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        if l == 0:
            subjects.add((h, t))
        elif l == 1:
            objects.add((h, t))
        else:
            objects2.add((h, t))
    # 识别对应的predicate
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            # 先收集 obejct
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            '''
            ps = set(p1s) & set(p2s)
            for p in ps:
                if p!=1: # 1 对应 '2', 有 object2
                    spoes.add((
                        text[mapping[sh][0]:mapping[st][-1] + 1], id2predicate[p],
                        text[mapping[oh][0]:mapping[ot][-1] + 1],
                        text[mapping[oh][0]:mapping[ot][-1] + 1],
                        (mapping[sh][0], mapping[st][-1] + 1),
                        (mapping[oh][0], mapping[ot][-1] + 1),
                        (mapping[oh][0], mapping[ot][-1] + 1)
                    ))
            '''
            # 再收集 object2
            for o2h, o2t in objects2:
                p3s = np.where(outputs[3][:, sh, oh] > threshold)[0]
                p4s = np.where(outputs[4][:, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s) & set(p3s) & set(p4s)
                for p in ps:
                    if p==1: # 1 对应 '2', 有 object2
                        spoes.add((
                            text[mapping[sh][0]:mapping[st][-1] + 1], id2predicate[p],
                            text[mapping[oh][0]:mapping[ot][-1] + 1],
                            text[mapping[o2h][0]:mapping[o2t][-1] + 1],
                            (mapping[sh][0], mapping[st][-1] + 1),
                            (mapping[oh][0], mapping[ot][-1] + 1),
                            (mapping[o2h][0], mapping[o2t][-1] + 1)
                        ))
                    else:
                        spoes.add((
                            text[mapping[sh][0]:mapping[st][-1] + 1], id2predicate[p],
                            text[mapping[oh][0]:mapping[ot][-1] + 1],
                            text[mapping[oh][0]:mapping[ot][-1] + 1],
                            (mapping[sh][0], mapping[st][-1] + 1),
                            (mapping[oh][0], mapping[ot][-1] + 1),
                            (mapping[oh][0], mapping[ot][-1] + 1)
                        ))

    return list(spoes)


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
            tuple(tokenizer.tokenize(spo[3])),
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
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
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
            model.save_weights('causal-v2_gp_model_f1_%.5f.weights'%f1)
        optimizer.reset_old_weights()
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


def predict_to_file(in_file, out_file):
    """预测到文件
    """
    data = json.load(open(in_file))
    for d in tqdm(data, ncols=100):
        d['relation_of_mention'] = []
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        for e in R:
            if e[1]!='2':
                d['relation_of_mention'].append({
                    "head" : {
                        "mention" : e[0],
                        "start_idx" : e[4][0],
                        "end_idx" : e[4][1]
                    },
                    'relation': int(e[1]),
                    'tail' : {
                        "type": "mention",
                        "mention" : e[2],
                        "start_idx" : e[5][0],
                        "end_idx" : e[5][1]
                    }
                })
            else:
                d['relation_of_mention'].append({
                    "head": {
                        "mention": e[0],
                        "start_idx": e[4][0],
                        "end_idx": e[4][1]
                    },
                    "relation": 2,
                    "tail": {
                        "type": "relation",
                        "head": {
                            "mention": e[2],
                            "start_idx": e[5][0],
                            "end_idx": e[5][1]
                        },
                        "relation": 1,
                        "tail": {
                            "mention": e[3],
                            "start_idx": e[6][0],
                            "end_idx": e[6][1]
                        }
                    }
                })

        #break

    json.dump(
        data,
        open(out_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    model.load_weights('causal-v2_gp_model_f1_0.52648.weights')

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=50,
        callbacks=[evaluator]
    )

else:

    model.load_weights('causal-v2_gp_model_f1_0.52648.weights')
    predict_to_file('../dataset/3.0/CMedCausal/CMedCausal_test.json', 'CMedCausal_test.json')
