
#  统计数据相关性

import os
from tqdm import tqdm

dataset_path = '../dataset/3.0/KUAKE-IR'
train_query = 'KUAKE-IR_train.tsv'
dev_query = 'KUAKE-IR_dev.tsv'

D = {}

def load_d(filename, tag):
    with open(os.path.join(dataset_path, filename), encoding='utf-8') as f:
        for l in f:
            query, doc = l.strip().split('\t')
            if doc in D.keys():
                D[doc].append(f"{tag}_{query}")
            else:
                D[doc] = [f"{tag}_{query}"]


load_d(dev_query, "dev")
print(len(D))
load_d(train_query, "train")
print(len(D))

t1 = t2 = 0
for k in D.keys():
    t1 += 1
    if len(D[k])>1:
        print(k, D[k])
        t2 += 1
    #print(k, D[k])

print(f"total={t1}, no_single={t2}")

# total=99872, no_single=687
