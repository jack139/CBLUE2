import os
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm


# 使用 CoSent
corpus_vec_pkl = "data/corpus_vec2_cosent.pkl"

'''
# 使用 SimCSE
corpus_vec_pkl = "data/corpus_vec2_simcse.pkl"
'''

# 文档的向量
with open(corpus_vec_pkl, "rb") as f:
    corpus_idx, corpus_np_vec = pickle.load(f)

corpus_tensor = torch.tensor(corpus_np_vec).cuda()
cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6) # 相似度


def search(test_vec_pkl, test_file, search_size=500):
    # 测试文本的向量
    with open(test_vec_pkl, "rb") as f:
        test_idx, test_np_vec = pickle.load(f)


    f2 = open(test_file, "w") # 结果文件

    for k, vec in tqdm(zip(test_idx, test_np_vec)):

        vec_tensor = torch.tensor(vec).cuda()
        output = cosine(corpus_tensor, vec_tensor)

        dists = output.cpu().numpy()
        ind = np.argpartition(dists, -search_size)[-search_size:]
        #print(ind)

        top_idx = np.array(corpus_idx)[ind]
        top_dist = dists[ind]
        top_n = zip(top_idx, top_dist)
        top_n = sorted(top_n, key=lambda x: x[1], reverse=True) # 按距离降序，是相似度，不是距离

        #rank_list = ','.join([i[0] for i in top_n])
        #f2.write(f"{k}\t{rank_list}\n")
        f2.write(json.dumps({"text":k, "recall":[i[0] for i in top_n], "score":[float(i[1]) for i in top_n]}, ensure_ascii=False) + "\n")

        #print(k, "-->", top_n)

        #break

    f2.close()


print("start ...")

search("data/test_vec2_cosent.pkl", 'data/cosent_test_recall.jsonl')
search("data/dev_vec2_cosent.pkl", 'data/cosent_dev_recall.jsonl', 10)
search("data/train_vec2_cosent.pkl", 'data/cosent_train_recall.jsonl', 10)
