# 准备score模型训练数据： 增加NER信息 

import pickle, json
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(filename):
    """加载数据 csv
    单条格式：text1,text2,label
    """
    outputs = {'text1': [], 'text2': [], 'label': []}
    train_cache_df = pd.read_csv(filename)
    outputs['text1'] = train_cache_df['text1'].values.tolist()
    outputs['text2'] = train_cache_df['text2'].values.tolist()
    outputs['label'] = train_cache_df['label'].values.tolist()
    return outputs


def get_NER(s):
    from ner.CMeEE_gp import NER
    return [ (s[c[0]:c[1]+1], c[2]) for c in NER.recognize(s) ]

# 增加NER信息
def transfer_data(infile, outfile):
    train_data = load_data(infile)

    D = []
    max_len = 0

    for text1, text2, label in tqdm(zip(train_data['text1'], train_data['text2'], train_data['label'])):
        # 编码
        text1ner = get_NER(text1) + [(text1, 'text')]
        text2ner = get_NER(text2) + [(text2, 'text')]
        t = {
            'text1' : '#'.join([ c[0]+'#'+c[1] for c in text1ner]),
            'text2' : '#'.join([ c[0]+'#'+c[1] for c in text2ner]),
            'label' : int(label)
        }
        D.append(t)
        if len(t['text1'])+len(t['text2'])>max_len:
            max_len = len(t['text1'])+len(t['text2'])

    json.dump(
        D,
        open(outfile, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    print('maxlen= ', max_len)

    #with open("train_ntn_embedding.pk", "wb") as f:
    #    pickle.dump(D, f)


# 文本做前处理，只对text1
def transfer_data2(infile, outfile):
    from baseline.text_process import clean, extend_x

    train_data = load_data(infile)

    D = []
    max_len = 0

    for text1, text2, label in tqdm(zip(train_data['text1'], train_data['text2'], train_data['label'])):
        t = {
            'text1' : clean(text1), # 前处理 text1
            'text2' : text2,
            'label' : int(label)
        }
        D.append(t)
        if len(t['text1'])+len(t['text2'])>max_len:
            max_len = len(t['text1'])+len(t['text2'])

    json.dump(
        D,
        open(outfile, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    print('maxlen= ', max_len)



if __name__ == '__main__':
    #transfer_data("data/CHIP-CDN/train_samples.csv", "data/train_score_samples.json")
    #transfer_data("data/CHIP-CDN/eval_samples_000.csv", "data/dev_score_samples.json")

    transfer_data2("data/CHIP-CDN/train_samples.csv", "data/train_score_samples2.json")
    transfer_data2("data/CHIP-CDN/eval_samples_000.csv", "data/dev_score_samples2.json")
