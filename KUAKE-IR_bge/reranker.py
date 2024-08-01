from FlagEmbedding import FlagReranker

llm_path = "/LLMs"
rerank_path = f"{llm_path}/lm_model/bge-reranker-v2-m3"
#rerank_path = "ckpt/reranker/checkpoint-2000" # finetune test

reranker = None

def init_model(device=0):
    if device==-1:
        device = "cpu"
        use_fp16 = False
    else:
        device = f'cuda:{device}'
        use_fp16 = True

    global reranker

    print(f"Load rerank model {rerank_path} in device {device} ... ")

    # Setting use_fp16 to True speeds up computation with a slight performance degradation
    reranker = FlagReranker(rerank_path, use_fp16=use_fp16, device=device) 

    # warm up
    score = reranker.compute_score([['query', 'passage']])


def rerank_list(query, text_list, normalize=False):
    params = []
    for c in text_list:
        params.append([query, c])
    return reranker.compute_score(params, normalize=normalize)
