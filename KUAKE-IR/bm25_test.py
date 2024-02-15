import os
from filter_corpus import corpus, corpus_rank, corpus_id

dataset_path = '../dataset/3.0/KUAKE-IR'
test_query = 'KUAKE-IR_test.query.txt'
test_file = './KUAKE-IR_test.tsv'

print("start ...")

f2 = open(test_file, "w")

with open(os.path.join(dataset_path, test_query), encoding='utf-8') as f:
    for l in f:
        k, query = l.strip().split('\t')
        print(k)

        rank = corpus_rank.get_document(query)
        rank_list = ','.join([str(corpus_id[i[0]]) for i in rank[:10]])
        f2.write(f"{k}\t{rank_list}\n")

        #print(k, query, "-->", rank[0][0], corpus_id[rank[0][0]], corpus[rank[0][0]])

        #break

f2.close()
