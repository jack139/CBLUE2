# 准备训练数据： 文本转换为句子向量
import pickle
import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
import pandas as pd
from tqdm import tqdm

maxlen = 128

config_path = '../../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

text = '语言模型语言模型'

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

train_data = load_data("../data/CHIP-CDN/train_ntn.csv")

D = []
for text1, text2, label in tqdm(zip(train_data['text1'], train_data['text2'], train_data['label'])):
    # 编码
    token_ids, segment_ids = tokenizer.encode(text1, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    e1 = model.predict([token_ids, segment_ids])

    token_ids, segment_ids = tokenizer.encode(text2, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    e2 = model.predict([token_ids, segment_ids])

    D.append([e1[0].tolist(), e2[0].tolist()])


#json.dump(
#    D,
#    open("train_ntn_embedding.json", 'w', encoding='utf-8'),
#    indent=4,
#    ensure_ascii=False
#)

with open("train_ntn_embedding.pk", "wb") as f:
    pickle.dump(D, f)

# 转为一维向量
'''
x, y = embeddings.shape

new_embeddings = np.concatenate((embeddings, np.zeros([maxlen-x, y])),axis=0)
print(new_embeddings)
print(new_embeddings.shape)

new_embeddings = new_embeddings.reshape(maxlen*y)
print(new_embeddings)
print(new_embeddings.shape)
'''
