import os
import pickle
import numpy as np
from tqdm import tqdm


import IR_cosent  # 使用 CoSent
corpus_vec_pkl = "data/corpus_vec2_cosent.pkl"
#test_vec_pkl = "data/test_vec2_cosent.pkl"

'''
import IR_simcse as IR_cosent # 使用 SimCSE
corpus_vec_pkl = "data/corpus_vec2_simcse.pkl"
test_vec_pkl = "data/test_vec2_simcse.pkl"
'''


dataset_path = '../dataset/3.0/KUAKE-IR'
corpus_file = 'corpus.tsv'
#test_query = 'KUAKE-IR_test.query.txt'


# 文档资料转向量
def corpus2vecs():
    if not os.path.exists(corpus_vec_pkl):
        corpus_idx = []
        corpus_vec = []
        with open(os.path.join(dataset_path, corpus_file), encoding='utf-8') as f:
            for l in tqdm(f):
                idx, doc = l.strip().split('\t')

                corpus_idx.append(idx)
                corpus_vec.append(IR_cosent.text2vec(doc))

        corpus_np_vec = np.array(corpus_vec, dtype=np.float16)

        with open(corpus_vec_pkl, "wb") as f:
            pickle.dump([corpus_idx, corpus_np_vec], f)
    else:
        with open(corpus_vec_pkl, "rb") as f:
            corpus_idx, corpus_np_vec = pickle.load(f)

    return corpus_idx, corpus_np_vec


def test2vecs(test_query, test_vec_pkl):
    if not os.path.exists(test_vec_pkl):
        test_idx = []
        test_vec = []
        with open(os.path.join(dataset_path, test_query), encoding='utf-8') as f:
            for l in tqdm(f):
                k, query = l.strip().split('\t')

                vec = IR_cosent.text2vec(query)

                test_idx.append(k)
                test_vec.append(vec)

        test_np_vec = np.array(test_vec, dtype=np.float16)

        with open(test_vec_pkl, "wb") as f:
            pickle.dump([test_idx, test_np_vec], f)

    else:
        with open(test_vec_pkl, "rb") as f:
            test_idx, test_np_vec = pickle.load(f)

    return test_idx, test_np_vec


print("corpus2vecs ...")
corpus_idx, corpus_np_vec = corpus2vecs()
print(len(corpus_idx), corpus_np_vec.shape)

print("test2vecs ...")
test_idx, test_np_vec = test2vecs('KUAKE-IR_test.query.txt', "data/test_vec2_cosent.pkl")
print(len(test_idx), test_np_vec.shape)

print("dev2vecs ...")
test_idx, test_np_vec = test2vecs('KUAKE-IR_dev_query.txt', "data/dev_vec2_cosent.pkl")
print(len(test_idx), test_np_vec.shape)

print("train2vecs ...")
test_idx, test_np_vec = test2vecs('KUAKE-IR_train_query.txt', "data/train_vec2_cosent.pkl")
print(len(test_idx), test_np_vec.shape)
