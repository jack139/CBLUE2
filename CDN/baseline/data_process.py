#! -*- coding:utf-8 -*-

import os
import json
import pickle
import pandas as pd
import jieba
import tqdm
import numpy as np
from gensim import corpora, models, similarities
from . import text_process

RECALL_K_ADD = 2000  # 测试集至少需要 1918

def load_json(input_file):
    with open(input_file, 'r') as f:
        samples = json.load(f)
    return samples

def str_q2b(text):
    ustring = text
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


class CDNDataProcessor(object):
    def __init__(self, root, recall_k=200, negative_sample=3, NER=None, data_dir='CHIP-CDN',
        train_json='CHIP-CDN_train.json', dev_json='CHIP-CDN_dev.json', test_json='CHIP-CDN_test.json'):
        self.task_data_dir = os.path.join(root, data_dir)
        self.train_path = os.path.join(self.task_data_dir, train_json)
        self.dev_path = os.path.join(self.task_data_dir, dev_json)
        self.test_path = os.path.join(self.task_data_dir, test_json)

        self.ner_train_set_path = os.path.join(self.task_data_dir, 'ner_train_set.json')
        self.ner_icd_set_path = os.path.join(self.task_data_dir, 'ner_icd_set.json')

        self.label_path = os.path.join(self.task_data_dir, '国际疾病分类 ICD-10北京临床版v601.xlsx')
        self.label2id, self.id2label = self._get_labels()

        self.recall_k = recall_k
        self.recall_k_total = self.recall_k + RECALL_K_ADD
        self.negative_sample = negative_sample

        self.dictionary = None
        self.index = None
        self.tfidf = None
        self.dictionary, self.index, self.tfidf = self._init_label_embedding()

        self.num_labels_cls = 2
        self.num_labels_num = 3

        self.recall = None

        self.NER = NER
        self.NER_ICD, self.NER_TRAIN = self._load_ner_set()

    def get_train_sample(self, dtype='cls', do_augment=1):
        """
        do_augment: data augment
        """
        samples = self._pre_process(self.train_path, is_predict=False)
        if dtype == 'cls':
            outputs, recall_orig_samples, recall_orig_samples_scores = self._get_cls_samples(orig_samples=samples, mode='train', do_augment=do_augment)
            return outputs, recall_orig_samples, recall_orig_samples_scores
        else:
            outputs = self._get_num_samples(orig_sample=samples, is_predict=False)
        return outputs

    def get_dev_sample(self, dtype='cls', do_augment=1):
        samples = self._pre_process(self.dev_path, is_predict=False)
        if dtype == 'cls':
            outputs, recall_orig_samples, recall_orig_samples_scores = self._get_cls_samples(orig_samples=samples, mode='eval', do_augment=do_augment)
            return outputs, recall_orig_samples, recall_orig_samples_scores
        else:
            outputs = self._get_num_samples(orig_sample=samples, is_predict=False)
        return outputs

    def get_val_sample(self, dtype='cls', do_augment=1):
        samples = self._pre_process(self.dev_path, is_predict=False)
        if dtype == 'cls':
            outputs, recall_orig_samples, recall_orig_samples_scores = self._get_cls_samples(orig_samples=samples, mode='val', do_augment=do_augment)
            return outputs, recall_orig_samples, recall_orig_samples_scores
        else:
            outputs = self._get_num_samples(orig_sample=samples, is_predict=False)
        return outputs

    def get_test_sample(self, dtype='cls'):
        samples = self._pre_process(self.test_path, is_predict=True)
        if dtype == 'cls':
            outputs, recall_orig_samples, recall_orig_samples_scores = self._get_cls_samples(orig_samples=samples, mode='test')
            return outputs, recall_orig_samples, recall_orig_samples_scores
        else:
            outputs = self._get_num_samples(orig_sample=samples, is_predict=True)
        return outputs

    def get_test_orig_text(self):
        samples = load_json(self.test_path)
        texts = [sample['text'] for sample in samples]
        return texts

    def get_val_orig_text(self):
        samples = load_json(self.dev_path)
        texts = [sample['text'] for sample in samples]
        return texts

    def _pre_process(self, path, is_predict=False):
        samples = load_json(path)
        outputs = {'text': [], 'label': []}

        for sample in samples:
            text = text_process.clean(sample['text']) # 文本做前处理
            text = self._process_single_sentence(text, mode="text")
            if is_predict:
                outputs['text'].append(text)
            else:
                label = self._process_single_sentence(sample['normalized_result'], mode="label")
                outputs['label'].append([label_ for label_ in label.split("##")])
                outputs['text'].append(text)
        return outputs

    def _save_cache(self, outputs, recall_orig_samples, mode='train'):
        cache_df = pd.DataFrame(outputs)
        cache_df.to_csv(os.path.join(self.task_data_dir, f'{mode}_samples.csv'), index=False)
        recall_orig_cache_df = pd.DataFrame(recall_orig_samples)
        recall_orig_cache_df['label'] = recall_orig_cache_df.label.apply(lambda x: " ".join([str(i) for i in x]))
        recall_orig_cache_df['recall_label'] = recall_orig_cache_df.recall_label.apply(
            lambda x: " ".join([str(i) for i in x]))
        recall_orig_cache_df.to_csv(os.path.join(self.task_data_dir, f'{mode}_recall_orig_samples.csv'),
                                    index=False)

    def _load_cache(self, mode='train'):
        outputs = {'text1': [], 'text2': [], 'label': []}
        recall_orig_samples = {'text': [], 'label': [], 'recall_label': []}

        train_cache_df = pd.read_csv(os.path.join(self.task_data_dir, f'{mode}_samples.csv'))
        outputs['text1'] = train_cache_df['text1'].values.tolist()
        outputs['text2'] = train_cache_df['text2'].values.tolist()
        outputs['label'] = train_cache_df['label'].values.tolist()

        train_recall_orig_cache_df = pd.read_csv(os.path.join(self.task_data_dir, f'{mode}_recall_orig_samples.csv'))
        recall_orig_samples['text'] = train_recall_orig_cache_df['text'].values.tolist()
        recall_orig_samples['label'] = train_recall_orig_cache_df['label'].values.tolist()
        recall_orig_samples['recall_label'] = train_recall_orig_cache_df['recall_label'].values.tolist()
        recall_orig_samples['label'] = [[int(label) for label in str(recall_orig_samples['label'][i]).split()] for i in
                                        range(len(recall_orig_samples['label']))]
        recall_orig_samples['recall_label'] = [[int(label) for label in str(recall_orig_samples['recall_label'][i]).split()] for i in
                                               range(len(recall_orig_samples['recall_label']))]
        recall_samples_scores = np.load(os.path.join(self.task_data_dir, f'{mode}_recall_score.npy'))

        return outputs, recall_orig_samples, recall_samples_scores

    def _get_cls_samples(self, orig_samples, mode='train', do_augment=1):
        if os.path.exists(os.path.join(self.task_data_dir, f'{mode}_samples.csv')) and \
                os.path.exists(os.path.join(self.task_data_dir, f'{mode}_recall_score.npy')) and \
                os.path.exists(os.path.join(self.task_data_dir, f'{mode}_recall_orig_samples.csv')):
            outputs, recall_orig_samples, recall_samples_scores = self._load_cache(mode=mode)
            return outputs, recall_orig_samples, recall_samples_scores

        outputs = {'text1': [], 'text2': [], 'label': []}

        texts = orig_samples['text']
        recall_samples_idx, recall_samples_scores = self._recall(texts)
        np.save(os.path.join(self.task_data_dir, f'{mode}_recall_score.npy'), recall_samples_scores)
        recall_orig_samples = {'text': [], 'label': [], 'recall_label': []}

        if mode == 'train':
            labels = orig_samples['label']
            for i in range(do_augment):
                for text, label in zip(texts, labels):
                    for label_ in label:
                        outputs['text1'].append(text)
                        outputs['text2'].append(label_)
                        outputs['label'].append(1)

            for text, orig_label, recall_label in zip(texts, labels, recall_samples_idx):
                orig_label_ids = [self.label2id[label] for label in orig_label]
                cnt_label = 0

                recall_orig_samples['text'].append(text)
                recall_orig_samples['label'].append(orig_label_ids)
                recall_orig_samples['recall_label'].append(recall_label)

                cur_idx = 0
                for label_ in recall_label:
                    if cnt_label >= self.negative_sample:
                        break
                    if label_==-1:
                        continue
                    if label_ not in orig_label_ids:
                        outputs['text1'].append(text)
                        outputs['text2'].append(self.id2label[label_])
                        outputs['label'].append(0)
                        orig_label_ids.append(label_)
                        cnt_label += 1
                    cur_idx += 1
                cnt_label = 0
                recall_label = np.random.permutation(recall_label[cur_idx:])
                for label_ in recall_label:
                    if cnt_label >= self.negative_sample:
                        break
                    if label_==-1:
                        continue
                    if label_ not in orig_label_ids:
                        outputs['text1'].append(text)
                        outputs['text2'].append(self.id2label[label_])
                        outputs['label'].append(0)
                        orig_label_ids.append(label_)
                        cnt_label += 1

            self._save_cache(outputs, recall_orig_samples, mode='train')

        elif mode == 'eval':
            labels = orig_samples['label']

            for i in range(do_augment):
                for text, label in zip(texts, labels):
                    for label_ in label:
                        outputs['text1'].append(text)
                        outputs['text2'].append(label_)
                        outputs['label'].append(1)

            cnt_not_recall = 0
            cnt_all_labels = 0
            max_ner_count = 0

            for text, orig_label, recall_label in zip(texts, labels, recall_samples_idx):
                orig_label_ids = [self.label2id[label] for label in orig_label]
                recall_orig_samples['text'].append(text)
                recall_orig_samples['recall_label'].append(recall_label)
                recall_orig_samples['label'].append(orig_label_ids)

                # # 检查 recall 集合里是否包含了 原始label
                # recall_ner_idx = self._recall_by_NER(text)
                # recall_label_with_ner = list(set(recall_label.tolist() + recall_ner_idx))
                # #print(len(recall_label), len(recall_ner_idx), len(recall_label_with_ner))
                # max_ner_count = max(len(recall_label_with_ner), max_ner_count)
                # for lb in orig_label_ids:
                #     if lb not in recall_label_with_ner:
                #         print(text, '--', self.id2label[lb])
                #         cnt_not_recall += 1
                # cnt_all_labels += len(orig_label_ids)
                # ---------------------

                cnt_label = 0
                cur_idx = 0
                for label_ in recall_label:
                    if cnt_label >= self.negative_sample:
                        break
                    if label_==-1:
                        continue
                    if label_ not in orig_label_ids:
                        outputs['text1'].append(text)
                        outputs['text2'].append(self.id2label[label_])
                        outputs['label'].append(0)
                        orig_label_ids.append(label_)
                        cnt_label += 1
                    cur_idx += 1

                cnt_label = 0
                recall_label = np.random.permutation(recall_label[cur_idx:])
                for label_ in recall_label:
                    if cnt_label >= self.negative_sample:
                        break
                    if label_==-1:
                        continue
                    if label_ not in orig_label_ids:
                        outputs['text1'].append(text)
                        outputs['text2'].append(self.id2label[label_])
                        outputs['label'].append(0)
                        orig_label_ids.append(label_)
                        cnt_label += 1

            self._save_cache(outputs, recall_orig_samples, mode='eval')

            #print("not_recall= %d\tall_labels= %d\tNR= %.4f"%(cnt_not_recall,cnt_all_labels,cnt_not_recall * 0.1 / cnt_all_labels))
            #print(max_ner_count)

        elif mode == 'val': # 配合 val_cdn.py
            labels = orig_samples['label']

            for text, orig_label, recall_label in zip(texts, labels, recall_samples_idx):
                orig_label_ids = [self.label2id[label] for label in orig_label]
                recall_orig_samples['text'].append(text)
                recall_orig_samples['recall_label'].append(recall_label)
                recall_orig_samples['label'].append(orig_label_ids) # 正确的label

                for label_ in recall_label:
                    outputs['text1'].append(text)
                    if label_==-1:
                        outputs['text2'].append('[BLANK]')
                    else:
                        outputs['text2'].append(self.id2label[label_])
                    outputs['label'].append(0)
            self._save_cache(outputs, recall_orig_samples, mode='val')

        else:
            max_ner_count = 0
            for text, recall_label in zip(texts, recall_samples_idx):

                recall_orig_samples['text'].append(text)
                recall_orig_samples['recall_label'].append(recall_label)
                recall_orig_samples['label'].append([0])

                # 检查 recall 集合里是否包含了 原始label
                #recall_ner_idx = self._recall_by_NER(text)
                #recall_label_with_ner = list(set(recall_label.tolist() + recall_ner_idx))
                #max_ner_count = max(len(recall_ner_idx), max_ner_count)
                # ---------------------

                for label_ in recall_label:
                    outputs['text1'].append(text)
                    if label_==-1:
                        outputs['text2'].append('[BLANK]')
                    else:
                        outputs['text2'].append(self.id2label[label_])
                    outputs['label'].append(0)
            self._save_cache(outputs, recall_orig_samples, mode='test')

            #print("max_ner_count= ", max_ner_count)

        return outputs, recall_orig_samples, recall_samples_scores

    def _get_num_samples(self, orig_sample, is_predict=False):
        outputs = {'text1': [], 'text2': [], 'label': []}

        if not is_predict:
            texts = orig_sample['text']
            labels = orig_sample['label']

            for text, label in zip(texts, labels):
                outputs['text1'].append(text)
                num_labels = len(label)
                if num_labels > 2:
                    num_labels = 3
                outputs['label'].append(num_labels-1)
        else:
            outputs['text1'] = orig_sample['text']

        return outputs

    def _init_label_embedding(self):
        all_label_list = []
        for idx in range(len(self.label2id.keys())):
            all_label_list.append(list(jieba.cut(self.id2label[idx])))

        dictionary = corpora.Dictionary(all_label_list)  # 词典
        corpus = [dictionary.doc2bow(doc) for doc in all_label_list]  # 语料库
        tfidf = models.TfidfModel(corpus)  # 建立模型
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))

        return dictionary, index, tfidf


    def _recall_by_NER(self, s):
        D = set([])
        ner_text = [ s[c[0]:c[1]+1] for c in self.NER.recognize(s) ]
        for i in ner_text:
            if i not in self.NER_TRAIN.keys():
                #print('1--> %s not found!!!'%i)
                continue

            for j in self.NER_TRAIN[i]:
                if j not in self.NER_ICD.keys():
                    #print('2--> %s not found!!!'%j)
                    continue

                for x in self.NER_ICD[j]:
                    D.add(x)

        return list(D) # 返回的是label id 列表

    def _score_test(self, text):
        x = text

        x_split = list(jieba.cut(x))
        x_vec = self.dictionary.doc2bow(x_split)
        x_sim = self.index[self.tfidf[x_vec]]  # 相似度分数 (1, labels)

        x_dices = np.zeros(len(self.label2id.keys()))
        x_set = set(x)

        for j, y in enumerate(self.label2id.keys()):
            y_set = set(y)
            x_dices[j] = len(x_set & y_set) / min(len(x_set), len(y_set))

        x_scores = x_sim + x_dices

        return x_scores

    def _recall(self, texts):
        recall_scores_idx = np.full((len(texts), self.recall_k_total), -1, dtype=np.int) # 空位的是 -1
        recall_scores = np.zeros((len(texts), self.recall_k_total))
        for i, x in tqdm.tqdm(enumerate(texts), total=len(texts)):
            x_split = list(jieba.cut(x))
            x_vec = self.dictionary.doc2bow(x_split)
            x_sim = self.index[self.tfidf[x_vec]]  # 相似度分数 (1, labels)

            x_dices = np.zeros(len(self.label2id.keys()))
            x_set = set(x)

            for j, y in enumerate(self.label2id.keys()):
                y_set = set(y)
                x_dices[j] = len(x_set & y_set) / min(len(x_set), len(y_set))

            x_scores = x_sim + x_dices
            x_scores_idx = np.argsort(x_scores)[:len(x_scores) - self.recall_k - 1:-1]  # 由大到小排序,取前K个
            x_scores = np.sort(x_scores)[:len(x_scores) - self.recall_k - 1:-1]
            recall_scores[i,:self.recall_k] += x_scores
            recall_scores_idx[i,:self.recall_k] = x_scores_idx

            # 使用 NER recall labels
            recall_ner_idx0 = self._recall_by_NER(x)
            recall_ner_idx = []
            for xx in recall_ner_idx0:
                if xx in x_scores_idx: # 去除重复的
                    continue
                recall_ner_idx.append(xx)
            recall_scores_idx[i,self.recall_k:self.recall_k+len(recall_ner_idx)] = np.array(recall_ner_idx)
            # 对 NER 的label进行 score 初始化，3种：max, average, zero
            #recall_scores[i,self.recall_k:self.recall_k+len(recall_ner_idx)] += np.full(len(recall_ner_idx), x_scores.max())
            #recall_scores[i,self.recall_k:self.recall_k+len(recall_ner_idx)] += np.full(len(recall_ner_idx), np.average(x_scores))
            recall_scores[i,self.recall_k:self.recall_k+len(recall_ner_idx)] += np.zeros(len(recall_ner_idx))

        return recall_scores_idx, recall_scores

    def _get_labels(self):
        if os.path.exists(os.path.join(self.task_data_dir, 'labels.pkl')):
            with open(os.path.join(self.task_data_dir, 'labels.pkl'), "rb") as f:
                label2id, id2label = pickle.load(f)
            print("Load labels.pkl ...")
            return label2id, id2label

        df = pd.read_excel(self.label_path, header=None, engine='openpyxl')
        normalized_word = df[1].unique().tolist()
        label2id = {word: idx for idx, word in enumerate(normalized_word)}
        id2label = {idx: word for idx, word in enumerate(normalized_word)}

        num_label = len(label2id.keys())
        samples = self._pre_process(self.train_path)
        for labels in samples['label']:
            for label in labels:
                if label not in label2id:
                    label2id[label] = num_label
                    id2label[num_label] = label
                    num_label += 1

        samples = self._pre_process(self.dev_path)
        for labels in samples['label']:
            for label in labels:
                if label not in label2id:
                    label2id[label] = num_label
                    id2label[num_label] = label
                    num_label += 1

        with open(os.path.join(self.task_data_dir, 'labels.pkl'), "wb") as f:
            pickle.dump([label2id, id2label], f)
            print("Saving labels.pkl ...")

        return label2id, id2label

    def _process_single_sentence(self, sentence, mode='text'):
        sentence = str_q2b(sentence)
        sentence = sentence.strip('"')
        if mode == "text":
            sentence = sentence.replace("\\", ";")
            sentence = sentence.replace(",", ";")
            sentence = sentence.replace("、", ";")
            sentence = sentence.replace("?", ";")
            sentence = sentence.replace(":", ";")
            sentence = sentence.replace(".", ";")
            sentence = sentence.replace("/", ";")
            sentence = sentence.replace("~", "-")
            sentence = sentence.replace(";;", ";")
        return sentence


    # 准备数据 - 相关标准词
    # 1. 对训练集中原始词和标准词做NER，抽取 bdy dis sym 做索引，原始词出现的上述类，对应标准词中的上述3类内容
    # 2. 对ICD-10数据做 NER，每条的 bdy dis sym，使用上述3类做索引，方便后续查询
    def _NER_train_set(self):
        
        D = {}
        '''
            D = {
                '左膝' : ['膝骨关节', '膝关节'],
            }
        '''
        data = self._pre_process(self.train_path) # 训练集
        for text, label in tqdm.tqdm(zip(data['text'], data['label'])):
            text_cats = [ text[c[0]:c[1]+1] for c in self.NER.recognize(text) ]
            norm_cats = []
            for i in label:
                norm_cats.extend([ i[c[0]:c[1]+1] for c in self.NER.recognize(i) ])

            for i in text_cats:
                if i in D.keys():
                    D[i] = list(set(D[i]+norm_cats))
                else:
                    D[i] = list(set(norm_cats))

        data = self._pre_process(self.dev_path) # 训练集
        for text, label in tqdm.tqdm(zip(data['text'], data['label'])):
            text_cats = [ text[c[0]:c[1]+1] for c in self.NER.recognize(text) ]
            norm_cats = []
            for i in label:
                norm_cats.extend([ i[c[0]:c[1]+1] for c in self.NER.recognize(i) ])

            for i in text_cats:
                if i in D.keys():
                    D[i] = list(set(D[i]+norm_cats))
                else:
                    D[i] = list(set(norm_cats))

        json.dump(
            D,
            open(self.ner_train_set_path, 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )


    def _NER_ICD_set(self):

        D = {}
        '''
            D = {
                '左膝' : [0, 14],
            }
        '''
        # self.label2id, self.id2label
        for label in tqdm.tqdm(self.label2id.keys()):
            text_cats = [ label[c[0]:c[1]+1] for c in self.NER.recognize(label) ]

            for i in text_cats:
                if i in D.keys():
                    D[i].append(self.label2id[label])
                else:
                    D[i] = [self.label2id[label]]
                D[i] = list(set(D[i]))

        json.dump(
            D,
            open(self.ner_icd_set_path, 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )

    def _load_ner_set(self):
        if not os.path.exists(self.ner_train_set_path):
            self._NER_train_set()

        if not os.path.exists(self.ner_icd_set_path):
            self._NER_ICD_set()

        return json.load(open(self.ner_icd_set_path)), json.load(open(self.ner_train_set_path))
