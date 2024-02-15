import os
import pickle
from utils.bm25 import CORPUS_RANK
from tqdm import tqdm

rank_pkl_file = "data/corpus_ranks2.pkl"
if not os.path.exists(rank_pkl_file):

    import convert

    corpus = []
    corpus_id = []
    for k in tqdm(convert.CORPUS.keys()):
        corpus_id.append(k)
        corpus.append(convert.CORPUS[k])

    corpus_rank = CORPUS_RANK(corpus)

    with open(rank_pkl_file, "wb") as f:
        pickle.dump([ corpus, corpus_rank, corpus_id ], f)

else:
    with open(rank_pkl_file, "rb") as f:
        corpus, corpus_rank, corpus_id = pickle.load(f)
