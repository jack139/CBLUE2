import json
import datetime,time
import os
import  re
import shutil
import tensorflow as tf
import pickle
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
# from bert4keras.snippets import open
from bert4keras.snippets import ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

maxlen = 256
epochs = 10
batch_size = 16
bert_layers = 12
learning_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率


# bert配置
config_path = '../../nlp_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射
labels = ['Symptom', 'Medicine', 'Test','Attribute','Disease']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1

data_path = "./data/"

"""
后面的代码使用的是bert类型的模型，如果你用的是albert，那么前几行请改为：
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='albert',
)
output_layer = 'Transformer-FeedForward-Norm'
output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
"""

model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output
output = Dense(num_labels)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)

model.load_weights('./data/outputs/MedDG_ner_Bert_f1_0.97428.weights')


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

with open(os.path.join(data_path, "test_add_info.pk"), "rb") as f:
    test_data = pickle.load(f)

test_crf_res = []

#sentence的数据处理
for each_data in tqdm(test_data):
    each_res = []
    for item in each_data:
        item_res = []
        item_res.append(item['Sentence'])
        item_ner = NER.recognize(item['Sentence'])
#         print('原文',item['Sentence'],'标签',item_ner)
        item_res.append(item_ner)
        each_res.append(item_res)
    test_crf_res.append(each_res)


def get_youyin(sentence):
    aa = "(吃早饭|吃饭规律|三餐规律|饮食规律|作息规律|辣|油炸|熬夜|豆类|油腻|生冷|煎炸|浓茶|喝酒|抽烟|吃的多|暴饮暴食|不容易消化的食物|情绪稳定|精神紧张|夜宵).*?(吗|啊？|呢？|么|嘛？)"
    bb = "吃得过多过饱|(有没有吃|最近吃|喜欢吃|经常进食|经常吃).*?(辣|油炸|油腻|生冷|煎炸|豆类|不容易消化的食物|夜宵)"
    cc = "(工作压力|精神压力|压力).*?(大不大|大吗)|(心情|情绪|精神).*(怎么样|怎样|如何)|(活动量|运动|锻炼).*?(大|多|少|多不多|咋样|怎么样|怎样).*?(吗|呢)"
    if re.search(aa,sentence) or re.search(bb,sentence) or re.search(cc,sentence):
        return True
    return False

def get_location(sentence):
    cc = '哪个部位|哪个位置|哪里痛|什么部位|什么位置|哪个部位|哪个位置|哪一块|那个部位痛|肚脐眼以上|描述一下位置|具体部位|具体位置'
    if re.search(cc,sentence) is not None:
        return True
    return False

def get_xingzhi(sentence):
    cc = '是哪种疼|怎么样的疼|绞痛|钝痛|隐痛|胀痛|隐隐作痛|疼痛.*性质|(性质|什么样).*(的|得)(痛|疼)'
    if re.search(cc,sentence) is not None:
        return True
    return False

def get_fan(sentence):
    cc = '(饭.*?前|吃东西前|餐前).*(疼|痛|不舒服|不适)|(饭.*?后|吃东西后|餐后).*(疼|痛|不舒服|不适)|(早上|早晨|夜里|半夜|晚饭).*(疼|痛|不舒服|不适)'
    if re.search(cc,sentence) is not None:
        aa = re.search(cc,sentence).span()
        if aa[1] - aa[0] <20:
            return True
    return False

def get_tong_pinglv(sentence):
    cc = '持续的疼|疼一会儿会自行缓解|持续的，还是阵发|症状减轻了没|(疼|痛).*轻|现在没有症状了吗|现在还有症状吗|(一阵一阵|一直|持续).*(疼|痛)|一阵阵.*(痛|疼)|阵发性|持续性'
    if re.search(cc,sentence) is not None:
        aa = re.search(cc,sentence).span()
        return True
    return False

def get_tong(sentence):
    if get_tong_pinglv(sentence) or get_fan(sentence) or get_xingzhi(sentence):
        return True
    return False


def get_other_sym(sentence):
    cc = '(还有什么|还有啥|有没有其|都有啥|都有什么|还有别的|有其他|有没有什么|还有其他).*(症状|不舒服)|别的不舒服|有其他不舒服|主要是什么症状|主要.*症状|哪些不适症状|哪些.*症状|出现了什么症状'
    if re.search(cc,sentence) is not None:
        aa = re.search(cc,sentence).span()
        return True
    return False

def get_time(sentence):
    aa = "(情况|症状|痛|发病|病|感觉|疼|这样|不舒服|大约).*?(多久|多长时间|几周了？|几天了？)"
    bb = "(，|。|、|？)(多长时间了|多久了|有多久了|有多长时间了)|^(多久了|多长时间了|有多久了|有多长时间了|几天了|几周了)"
    cc = "有多长时间|有多久"
#     match_need = "多久了？|多长时间了？|几天了？|几周了？|多久了|多长时间了|几天了|几周了"
    if re.search(aa,sentence) is not None or re.search(bb,sentence) is not None:
        return True
    return False


test_data_add_entities = []

for test_data_item, test_entities_item in zip(test_data, test_crf_res):
    
    test_data_add_entities_item = []
    
    for test_data_item_s, test_entities_item_s in zip(test_data_item, test_entities_item):
        
        test_data_item_s =  {k: set(v) if type(v) == list else v for k,v in test_data_item_s.items()}
        
        s, entities = test_entities_item_s
        
        for entity_text, entity_type in entities:
            
            test_data_item_s[entity_type].add(entity_text)
        
        # 增加属性
        
        if get_location(s):
            test_data_item_s['Attribute'].add('位置')
        if get_youyin(s):
            test_data_item_s['Attribute'].add('诱因')
        if get_tong(s):
            test_data_item_s['Attribute'].add('性质')
        if get_time(s):
            test_data_item_s['Attribute'].add('时长')
            
        test_data_item_s =  {k: list(v) if type(v) == set else v for k,v in test_data_item_s.items()}
        
        test_data_add_entities_item.append(test_data_item_s)
    
    test_data_add_entities.append(test_data_add_entities_item)
            
with open(os.path.join(data_path, "test_add_info_entities.pk"), "wb") as f:
    pickle.dump(test_data_add_entities, f)