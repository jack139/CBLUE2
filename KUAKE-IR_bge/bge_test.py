import os
from filter_bge import query_and_rerank
import reranker

dataset_path = '../dataset/3.0/KUAKE-IR'
test_query = 'KUAKE-IR_test.query.txt'
test_file = './KUAKE-IR_test.tsv'

top = 50

reranker.init_model(device=0)

print("start ...")

f2 = open(test_file, "w")

with open(os.path.join(dataset_path, test_query), encoding='utf-8') as f:
    for l in f:
        k, query = l.strip().split('\t')
        print(k)

        rank = query_and_rerank(reranker, query, top)
        rank_list = ','.join([i[1] for i in rank[:10]])
        f2.write(f"{k}\t{rank_list}\n")

        #print(k, query, "-->", rank[0][0], corpus_id[rank[0][0]], corpus[rank[0][0]])

f2.close()
