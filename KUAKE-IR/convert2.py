import os
import random
import json

# 使用 cosent 向量结果生成 负例

'''
[
  {
    "text1": "糖尿病能吃减肥药吗？能治愈吗？",
    "text2": "糖尿病为什么不能吃减肥药",
    "label": "1",
  },
]

'''

dataset_path = '../dataset/3.0/KUAKE-IR'
corpus_file = 'corpus.tsv'


def load_corpus(filename):
    print(f"load {filename} ...")

    max1 = 0
    corpus = {}
    with open(os.path.join(dataset_path, filename), encoding='utf-8') as f:
        for l in f:
            k, v = l.strip().split('\t')
            corpus[k] = v

            if len(v)>256:
                max1 += 1

    print(len(corpus), "loaded.", f"\tmax={max1}")
    return corpus


CORPUS = load_corpus(corpus_file)
CORPUS_key = list(CORPUS.keys())


def convert(infile_query, infile_idx, consent_recall_file, outfile):

    print(f"{infile_idx} --> {outfile}")

    # cosent 召回的数据
    cosent_R = {}
    with open(consent_recall_file, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            cosent_R[l["text"]] = (l["recall"], l["score"])

    # 文本数据
    Q = load_corpus(infile_query)

    D = []
    max2 = 0

    with open(os.path.join(dataset_path, infile_idx), encoding='utf-8') as f:
        for l in f:
            k1, k2 = l.strip().split('\t')

            D.append({
                'text1' : CORPUS[k2],
                'text2' : Q[k1],
                'label' : "1"
            })

            ## 负例
            if cosent_R[k1][0][0]!=k2:
                k3 = cosent_R[k1][0][0]
            else:
                k3 = cosent_R[k1][0][1]
            D.append({
                'text1' : CORPUS[k3],
                'text2' : Q[k1],
                'label' : "0"
            })

            if len(CORPUS[k2]) > 256:
                max2 += 1

            if len(CORPUS[k3]) > 256:
                max2 += 1

    json.dump(
        D,
        open(outfile, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    print(f"total={len(D)}\tmax2={max2}")


if __name__ == '__main__':
    convert('KUAKE-IR_dev_query.txt', 'KUAKE-IR_dev.tsv', '../KUAKE-IR_vec/data/cosent_dev_recall.jsonl', 'data/dev.json')
    convert('KUAKE-IR_train_query.txt', 'KUAKE-IR_train.tsv', '../KUAKE-IR_vec/data/cosent_train_recall.jsonl', 'data/train.json')
