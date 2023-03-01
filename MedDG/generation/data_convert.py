import pickle
import random
import shutil
import copy
from tqdm import tqdm
import os
import tensorflow as tf
from tqdm import tqdm
import json
from bert4keras.snippets import parallel_apply
from bert4keras.tokenizers import Tokenizer, SpTokenizer

spm_path = '../../nlp_model/mt5_base/sentencepiece_cn.model'
keep_tokens_path = '../../nlp_model/mt5_base/sentencepiece_cn_keep_tokens.json'

tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
keep_tokens = json.load(open(keep_tokens_path))


with open('./data/160_last_topic2num.pk', 'rb') as f:
    topic2num = pickle.load(f)

import re

def add_split_token(sentence):
    
    if sentence is not None and len(sentence) > 0 and re.search(u'[\u4e00-\u9fa5]+', sentence[-1]):
        sentence = sentence + "。"
    
    return sentence

def get_e(sen):
    
    return (
        sorted(set(sen['Symptom'])) 
        + sorted(set(sen['Attribute'])) 
        + sorted(set(sen['Test'])) 
        + sorted(set(sen['Disease'])) 
        + sorted(set(sen['Medicine']))
    )


with open('./data/train_data_with_bert_entites.pk','rb') as f:
    train_dataset = pickle.load(f)

with open('./data/dev_data_with_bert_entites.pk','rb') as f:
    dev_dataset = pickle.load(f)

def create_data(dataset, use_bert_entities=True):
    dig_train_data = []

    for dialog in tqdm(dataset):
        new_dialog = []
        history = []

        for sen in dialog:

            sen['Sentence'] = add_split_token(sen['Sentence'])

            if len(history) > 0 and sen['id'] == 'Doctor':
                
                if use_bert_entities:
                    ee = sen['bert_word']
                else:
                    ee = get_e(sen)
                
                new_dialog.append({
                    "history": copy.deepcopy(history),
                    "bert_word": copy.deepcopy(ee),
                    'response': sen['Sentence'],
                })


            history.append(("病人：" if sen['id'] == "Patients" else "医生：") + sen['Sentence'])


        for dic in new_dialog:
            dic['history'][-1] = dic['history'][-1] + "<" + ','.join(dic['bert_word']) + ">"
#             del dic['bert_word']
            dig_train_data.append(dic)
    
    return dig_train_data

dig_train_data = create_data(train_dataset, use_bert_entities=False)
dig_train_data_with_bert_entity = create_data(train_dataset, use_bert_entities=True)
dig_dev_data = create_data(dev_dataset, use_bert_entities=False)
dig_dev_data_with_bert_entity = create_data(dev_dataset, use_bert_entities=True)


with open('./data/dig_train_data.pk', 'wb') as f:
    pickle.dump(dig_train_data, f)

with open('./data/dig_train_data_with_bert_entity.pk', 'wb') as f:
    pickle.dump(dig_train_data_with_bert_entity, f)

with open('./data/dig_dev_data.pk', 'wb') as f:
    pickle.dump(dig_dev_data, f)

with open('./data/dig_dev_data_with_bert_entity.pk', 'wb') as f:
    pickle.dump(dig_dev_data_with_bert_entity, f)


print(len(dig_train_data))
print(len(dig_train_data_with_bert_entity))
print(len(dig_dev_data))
print(len(dig_dev_data_with_bert_entity))


# 构造训练与开发集的tfrecord


max_in_len = 512
max_out_len = 128


def create_int_feature(x):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=x))


def deal_dig(data_digs):
    
    for data_item in data_digs:

        c_tokens = []
        
        for item in data_item['history'][::-1]:
            item = tokenizer._tokenize(item)
            
            if len(c_tokens) + len(item) > max_in_len - 1:
                break

            c_tokens = item + c_tokens

        
        t_tokens = tokenizer._tokenize(data_item['response'])
        
        if len(t_tokens) > max_out_len - 2:
            t_tokens = t_tokens[:max_out_len - 2]

        c_tokens = c_tokens + ["</s>"]
        t_tokens = ["<pad>"] + t_tokens + ["</s>"]

#         print(" ".join(c_tokens))
#         print(" ".join(t_tokens))
#         print("--------------------------")

        c_token_ids = tokenizer.tokens_to_ids(c_tokens)
        t_token_ids = tokenizer.tokens_to_ids(t_tokens)

        c_token_ids += [0] * (max_in_len - len(c_token_ids))
        t_token_ids += [0] * (max_out_len - len(t_token_ids))


        tf_features = tf.train.Features(feature={
            'c_token_ids': create_int_feature(c_token_ids),
            't_token_ids': create_int_feature(t_token_ids)
        })
        tf_example = tf.train.Example(features=tf_features)
        serialized_instance = tf_example.SerializeToString()

        yield serialized_instance



def write_to_tf_record(writer_path, data):
    count = 0

    writer = tf.io.TFRecordWriter(writer_path)

    for serialized_instance in deal_dig(tqdm(data)):
        writer.write(serialized_instance)
        count += 1

    writer.close()
    return count


writer_path_base = "./data/t5_base/"
if os.path.exists(writer_path_base):
    shutil.rmtree(writer_path_base)

os.makedirs(writer_path_base)

print(write_to_tf_record(writer_path_base+"corpus.train.tfrecord", dig_train_data))
print(write_to_tf_record(writer_path_base+"corpus.train.bert_entity.tfrecord", dig_train_data_with_bert_entity))
print(write_to_tf_record(writer_path_base+"corpus.dev.tfrecord", dig_dev_data))
print(write_to_tf_record(writer_path_base+"corpus.dev.bert_entity.tfrecord", dig_dev_data_with_bert_entity))

import numpy as np

def random_replace(token):
    token = token.copy()
    
    # 替换概率 0.5
    if np.random.random() > 0.5:
        
        token = [t if np.random.random() > 0.10 else np.random.choice(token) for t in token]
        
    return token

def deal_dig_add_difficult(data_digs):
    
    for data_item in data_digs:

        c_tokens = []
        
        for item in data_item['history'][::-1]:
            item = tokenizer._tokenize(item)
            
            if len(c_tokens) + len(item) > max_in_len - 1:
                break

            c_tokens = item + c_tokens

        
        t_tokens = tokenizer._tokenize(data_item['response'])
        
        if len(t_tokens) > max_out_len - 2:
            t_tokens = t_tokens[:max_out_len - 2]
            
        
        re_t_tokens = random_replace(t_tokens)
        

        c_tokens = c_tokens + ["</s>"]
        t_tokens = ["<pad>"] + t_tokens + ["</s>"]
        re_t_tokens = ["<pad>"] + re_t_tokens + ["</s>"]

#         print(" ".join(c_tokens))
#         print(" ".join(t_tokens))
#         print("--------------------------")

        c_token_ids = tokenizer.tokens_to_ids(c_tokens)
        t_token_ids = tokenizer.tokens_to_ids(t_tokens)
        re_t_token_ids = tokenizer.tokens_to_ids(re_t_tokens)

        c_token_ids += [0] * (max_in_len - len(c_token_ids))
        t_token_ids += [0] * (max_out_len - len(t_token_ids))
        re_t_token_ids += [0] * (max_out_len - len(re_t_token_ids))


        tf_features = tf.train.Features(feature={
            'c_token_ids': create_int_feature(c_token_ids),
            't_token_ids': create_int_feature(t_token_ids),
            're_t_token_ids': create_int_feature(re_t_token_ids)
        })
        tf_example = tf.train.Example(features=tf_features)
        serialized_instance = tf_example.SerializeToString()

        yield serialized_instance


def write_to_tf_record_difficult(writer_path, data):
    count = 0

    writer = tf.io.TFRecordWriter(writer_path)

    for _ in range(5):
        for serialized_instance in deal_dig_add_difficult(tqdm(data)):
            writer.write(serialized_instance)
            count += 1

    writer.close()
    return count


writer_path_base = "./data/t5_difficult/"
if os.path.exists(writer_path_base):
    shutil.rmtree(writer_path_base)

os.makedirs(writer_path_base)

print(write_to_tf_record_difficult(writer_path_base+"corpus.train.tfrecord", dig_train_data))
print(write_to_tf_record_difficult(writer_path_base+"corpus.train.bert_entity.tfrecord", dig_train_data_with_bert_entity))
print(write_to_tf_record_difficult(writer_path_base+"corpus.dev.tfrecord", dig_dev_data))
print(write_to_tf_record_difficult(writer_path_base+"corpus.dev.bert_entity.tfrecord", dig_dev_data_with_bert_entity))