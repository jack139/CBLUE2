import os
import json
import pickle
from tqdm import tqdm
from convert import CORPUS
from IR_gp import tokenizer, model, maxlen # 微调的模型

dataset_path = '../dataset/3.0/KUAKE-IR'
test_query = 'KUAKE-IR_test.query.txt'
test_file = './KUAKE-IR_test.tsv'

consent_recall_file = '../KUAKE-IR_vec/data/cosent_test_recall.jsonl'
#consent_recall_file = '../KUAKE-IR_vec/data/cosent_test_recall_simcse.jsonl'

score_pkl_file = 'data/score_all.pkl'

# cosent 召回的数据
cosent_R = {}
with open(consent_recall_file, encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        cosent_R[l["text"]] = (l["recall"], l["score"])

print("start ...")


if not os.path.exists(score_pkl_file):
    score_all = []
    with open(os.path.join(dataset_path, test_query), encoding='utf-8') as f:
        for l in tqdm(f):
            k, query = l.strip().split('\t')
            #print(k)

            score = []
            for n, idx in enumerate(cosent_R[k][0][:200]):
                text1 = CORPUS[idx]
                text2 = query
                token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
                prob = model.predict([[token_ids], [segment_ids]])[0] # [p1, p2]
                score.append([idx, prob[1]])

            score_all.append((k, score))

            #break

    with open(score_pkl_file, "wb") as f:
        pickle.dump(score_all, f)

else:
    with open(score_pkl_file, "rb") as f:
        score_all = pickle.load(f)


# 生成结果
with open(test_file, "w") as f:

    for k, score in tqdm(score_all):
        # 处理 score ： 正例 标签的概率 + cosent 计算的相似度
        for n in range(len(score)): 
            #score[n][1] = score[n][1] + cosent_R[k][1][n] / 2 # 16.2347
            #score[n][1] = score[n][1] / 2  + cosent_R[k][1][n] # 14.8262
            score[n][1] = score[n][1] # 不处理  21.5994

        # 按 打分排序
        score = sorted(score, key=lambda x: x[1], reverse=True)

        # 写入文件
        rank_list = ','.join([str(i[0]) for i in score[:10]])
        f.write(f"{k}\t{rank_list}\n")
