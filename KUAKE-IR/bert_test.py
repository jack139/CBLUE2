import os
from filter_corpus import corpus, corpus_rank, corpus_id
from IR_gp import tokenizer, model, maxlen # 微调的模型

dataset_path = '../dataset/3.0/KUAKE-IR'
test_query = 'KUAKE-IR_test.query.txt'
test_file = './KUAKE-IR_test.tsv'

print("start ...")

f2 = open(test_file, "w")

with open(os.path.join(dataset_path, test_query), encoding='utf-8') as f:
    for l in f:
        k, query = l.strip().split('\t')
        print(k)

        # bm25召回相似文档
        rank = corpus_rank.get_document(query)

        score = []
        for i in rank[:500]:
            idx = i[0]
            text1 = corpus[idx]
            text2 = query
            token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
            #label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            prob = model.predict([[token_ids], [segment_ids]])[0] # [p1, p2]
            #print(prob)
            score.append((corpus_id[idx], prob[1])) # 取 正例 标签的概率

        # 按 概率排序
        score = sorted(score, key=lambda x: x[1], reverse=True)

        # 写如文件
        rank_list = ','.join([str(i[0]) for i in score[:10]])
        f2.write(f"{k}\t{rank_list}\n")

        #break

f2.close()
