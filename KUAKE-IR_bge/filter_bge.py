import os
import numpy as np
from tqdm import tqdm

from camel.embeddings import BGEM3Encoder
from camel.storages import QdrantStorage, VectorRecord
from camel.types import VectorDistance

# model path
llm_path = "/LLMs"
embeddings_path = f"{llm_path}/lm_model/bge-m3"
#embeddings_path = "ckpt/embed/checkpoint-4000" # finetune test

# qdrant settings
vec_db_host = "127.0.0.1"
vec_collection_name = "kuake_ir"

rerank_treshold = 0.0


embedding_model = BGEM3Encoder(
    model_name=embeddings_path,
    device=0
)

vector_storage = QdrantStorage(
    embedding_model.get_output_dim(),
    url_and_api_key=(vec_db_host, None),
    collection_name=vec_collection_name,
    distance=VectorDistance.COSINE,
)


'''
1. 文档库 生成 bge_m3 向量, 存入 qdrant
2. 查询 生成 bge_m3 向量, 检索 qdrant
3. bge_reanker 对检索结果重排
'''


# 文档库 生成 bge_m3 向量, 存入 qdrant
def gen_embed_and_save_qdrant(total_process, process_num):

    import convert

    print("Before: ", vector_storage._get_collection_info(vec_collection_name))

    if process_num==0:
        vector_storage.clear() # 清除 collection 所有数据
        print("Cleaning ...")

    # 分割 process
    total_keys = sorted(convert.CORPUS_key)
    total = len(total_keys)
    page_size = total // total_process + 1
    start_idx = page_size*process_num
    end_idx = page_size*(process_num+1)

    print(f"total= {total}, total_process= {total_process}, process_num= {process_num}")
    print(f"from: {start_idx} to: {end_idx}")

    for k in tqdm(total_keys[start_idx:end_idx]):

        text_id = k
        text = convert.CORPUS[k]

        # 生成向量
        embedding = embedding_model.embed(text)

        #print(embedding)

        records = []
        records.append(
            VectorRecord(
                #id=c2['uuid'],
                vector=embedding, 
                payload={"text": text, "text_id": text_id}
            )
        )
        vector_storage.add(records)

    print("After: ", vector_storage._get_collection_info(vec_collection_name))


# 检索
def query_and_rerank(reranker, query_text, top=100):

    # query 向量
    query_embedding = embedding_model.embed(query_text) # 使用多轮问题检索
    results = vector_storage.get_payloads_by_vector(query_embedding, top) # 召回 top 数量，再rerank

    # 重排序
    content = [result["text"] for result in results]
    rerank_score = np.array(reranker.rerank_list(query_text, content, normalize=True))
    content_rerank = rerank_score.argsort() # content 重新排序的 index
    content_rerank = content_rerank[::-1]
    # 大于阈值的都进入上下文
    content2 = [(results[c]["text"], results[c]["text_id"], rerank_score[c]) for c in content_rerank \
                            if rerank_score[c]>rerank_treshold]

    return content2


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--total-process", type=int, default=1)
    parser.add_argument("-p", "--process-number", type=int, default=0)
    args = parser.parse_args()

    # 准备 向量库
    gen_embed_and_save_qdrant(args.total_process, args.process_number)
