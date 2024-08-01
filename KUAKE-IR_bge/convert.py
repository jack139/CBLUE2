import os
import random
import json

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


max_len = 512


def load_corpus(filename):
    print(f"load {filename} ...")

    max1 = 0
    corpus = {}
    with open(os.path.join(dataset_path, filename), encoding='utf-8') as f:
        for l in f:
            k, v = l.strip().split('\t')
            corpus[k] = v

            if len(v)>max_len:
                max1 += 1

    print(len(corpus), "loaded.", f"\tmax={max1}")
    return corpus


CORPUS = load_corpus(corpus_file)
CORPUS_key = list(CORPUS.keys())



# 生成 微调 reranker 数据
def convert(infile_query, infile_idx, outfile):

    # 从 qdrant 生成负例
    from filter_bge import embedding_model, vector_storage

    print(f"{infile_idx} --> {outfile}, corpus len: {len(CORPUS_key)}")

    Q = load_corpus(infile_query)

    D = []
    max2 = 0

    with open(os.path.join(dataset_path, infile_idx), encoding='utf-8') as f:
        for l in f:
            k1, k2 = l.strip().split('\t')

            d = {
                "query": Q[k1],
                "pos": [CORPUS[k2]],
                "neg": []
            }

            if len(CORPUS[k2]) > max_len:
                max2 += 1

            ## 负例

            # query 向量
            query_embedding = embedding_model.embed(Q[k1]) # 使用多轮问题检索
            # 从 qdrant 生成负例
            results = vector_storage.get_payloads_by_vector(query_embedding, 10) #

            for x in results:
                if k2==x['text_id']:
                    #print(f"crash k2! {k1}, {k2} {x['text_id']}")
                    continue

                d["neg"].append(x['text'])

                if len(x['text']) > max_len:
                    max2 += 1

                if len(d["neg"])==8: # 控制负例个数
                    break

            assert len(d["neg"])==8, "neg list not enough!"

            D.append(d)


    with open(outfile, 'w') as output_data:
        for d in D:
            output_data.write(json.dumps(d, ensure_ascii=False) + '\n')


    print(f"total={len(D)}\tmax2={max2}")


if __name__ == '__main__':
    #convert('KUAKE-IR_dev_query.txt', 'KUAKE-IR_dev.tsv', 'data/dev.jsonl')
    convert('KUAKE-IR_train_query.txt', 'KUAKE-IR_train.tsv', 'data/train.jsonl')
