import json
import datetime, time
import os
import copy
import tensorflow as tf
#os.environ["RECOMPUTE"] = '0'

import pickle
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, SpTokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator

from keras.layers import Lambda, Dense
from keras.models import Model

import numpy as np
#from tqdm.notebook import tqdm
from .i2c import i2c

projectName = 'MedDG_next_entities: 简单多分类'

# MedBERT
config_path = '../../nlp_model/PCL-MedBERT-wwm/bert_config.json'
checkpoint_path = '../../nlp_model/PCL-MedBERT-wwm/bert_model.ckpt'
dict_path = '../../nlp_model/PCL-MedBERT-wwm/vocab.txt'

# BERT base
#config_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
#checkpoint_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
#dict_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

maxlen = 512
epochs = 100
batch_size = 16
steps_per_epoch = 66258 // batch_size + 1  # 实际训练集数量(最大) 66258

learning_rate = 4e-5 # 4e-5 2e-5


c2i = { v:idx  for idx, v in enumerate(i2c) }
c2ii = { v[0]:idx  for idx, v in enumerate(i2c) }
c2c = { v[0]:v  for v in i2c }


data_path = "./data/"

with open(os.path.join(data_path, "train_data.pk"), "rb") as f:
    train_data = pickle.load(f)
    
with open(os.path.join(data_path, "dev_data.pk"), "rb") as f:
    dev_data = pickle.load(f)

print(len(train_data))
print(len(dev_data))

def create_content_label(data_item):
    new_dialog = []
    history = []
    now_topic = []
    his_topic = []
    for sen in data_item:
        aa = sen['Symptom']+sen['Attribute']+sen['Test']+sen['Disease']+sen['Medicine']
        if len(aa) > 0:
            if len(history) > 0 and sen['id'] == 'Doctor':
                new_dialog.append({"history": copy.deepcopy(history), "next_sym": copy.deepcopy(aa), 'now_topic': copy.deepcopy(now_topic)})
            now_topic.extend(aa)
            his_topic.extend(aa)
        history.append(sen['Sentence'])
    for dic in new_dialog:
        future = copy.deepcopy(his_topic[len(dic['now_topic']):])
        dic['future'] = future
        content = ''.join(dic['history'])
        append_entities = '[unused1]'.join(dic['now_topic'])
        yield content, append_entities, dic['next_sym']


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        
        batch_token_ids, batch_segment_ids, batch_append_entity_ids = [], [], []
        batch_labels = []
        
        prob = 1.0

        for is_end, data_item in self.sample(random):
            
            for content, append_entities, label in create_content_label(data_item):
                token_ids, segment_ids = tokenizer.encode(append_entities, content, maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                # 生成答案
                label_item = [0 for _ in i2c]
                for v in label:
                    label_item[c2ii[v]] = 1
                batch_labels.append(label_item)

                #print(content)
                #print(append_entities)
                #print(label)
                #print(label_item)

                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    #batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], np.array(batch_labels)
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []

# 模型开始
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=len(i2c), 
    #activation="tanh", #'softmax',
    kernel_initializer=bert.initializer,
    name="Label-Id"
)(output)

train_model = keras.models.Model(bert.model.input, output)
train_model.summary()

    
optimizer = Adam(learning_rate=learning_rate)
train_model.compile(loss=multilabel_categorical_crossentropy, optimizer=optimizer)

train_model.summary()


# 评价
def evaluate(data):

    batch_token_ids, batch_segment_ids = [], []
    
    true_labels = []

    for data_item in data:

        for content, append_entities, label in create_content_label(data_item):

            token_ids, segment_ids = tokenizer.encode(append_entities, content, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            _label = {
                "Symptom": set(),
                "Medicine": set(),
                "Test": set(),
                "Attribute": set(),
                "Disease": set()
            }
            for v in label:
                _label[c2c[v][1]].add(v)
            true_labels.append(_label)

    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)

    y = train_model.predict([batch_token_ids, batch_segment_ids], batch_size=batch_size, verbose=1)

    predict_data = []

    for idx in range(len(batch_token_ids)):
        
        predict_data_item = {k:set() for e, k in i2c}
        
        for item_idx, item in enumerate(y[idx]):
            if item > 0:
                e, t = i2c[item_idx]
                predict_data_item[t].add(e)
                
        predict_data.append(predict_data_item)
        
    score = {k:{'p': 0, 's': 0, 'i':0} for e, k in i2c}

    for predict_data_item, true_data_item in zip(predict_data, true_labels):

        for label, pred_entity in  predict_data_item.items():
            if label == 'None': 
                score[label]['p'] += len(pred_entity)
                #score[label]['s'] += len(true_data_item[label])
                score[label]['i'] += len(pred_entity)
            else:
                score[label]['p'] += len(pred_entity)
                score[label]['s'] += len(true_data_item[label])
                score[label]['i'] += len(true_data_item[label] & pred_entity)

    all_p, all_s, all_i = 0, 0, 0

    for k,v in score.items():
        all_p += v['p']
        all_s += v['s']
        all_i += v['i']

        P = v['i'] / v['p'] if v['p'] != 0 else 0
        R = v['i'] / v['s'] if v['s'] != 0 else 0 
        f1 = 2 * P * R / (P + R) if P + R != 0 else 0 

        print(k + " P: %.2f R: %.2f F1: %.2f" % (P*100, R*100, f1*100))

    P = all_i / all_p if all_p != 0 else 0
    R = all_i / all_s if all_s != 0 else 0 
    f1 = 2 * P * R / (P + R) if P + R != 0 else 0 

    print("ALL P: %.2f R: %.2f F1: %.2f" % (P*100, R*100, f1*100))

    return f1

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model_saved_path):
        
        self.best_bleu = 0.
        self.model_saved_path = model_saved_path

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(dev_data)

        if metrics > self.best_bleu:
            self.best_bleu = metrics
            train_model.save_weights(os.path.join(self.model_saved_path, 'MedDG_next3_MedBert_f1_%.5f.weights'%metrics))

if __name__ == '__main__':
    print(projectName + ' Train...')
    resultPath = './data/outputs'
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    train_generator = data_generator(train_data, batch_size)

    train_model.load_weights(os.path.join(resultPath, "MedDG_next3_MedBert_f1_0.11358.weights"))

    train_model.fit(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[Evaluator(resultPath)]
    )