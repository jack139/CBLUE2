#! -*- coding:utf-8 -*-

import os
import json, pickle
import numpy as np
from model.score_model import predict as predict_score
from model.num_model import predict as predict_num
from tqdm import tqdm
from baseline.data_process import CDNDataProcessor, RECALL_K_ADD
from baseline import text_process
from whitening_helper import get_sim

RECALL_K=400
NUM_NEGATIVE_SAMPLES=5
DO_AUGMENT=6

RECALL_K_TOTAL = RECALL_K + RECALL_K_ADD

data_processor = CDNDataProcessor(root='./data', 
    recall_k=RECALL_K, 
    negative_sample=NUM_NEGATIVE_SAMPLES,
    #test_json='test.json'
)


def cdn_commit_prediction(text, preds, num_preds, recall_labels, recall_scores, output_dir, id2label):
    text1 = text

    pred_result = []
    active_indices = (preds >= 0.4)
    for text, active_indice, pred, num, recall_label, recall_score in \
        tqdm(zip(text1, active_indices, preds, num_preds, recall_labels, recall_scores), total=len(text1)):

        tmp_dict = {'text': text, 'normalized_result': []}

        final_pred = pred[active_indice]
        recall_score = recall_score[active_indice]
        recall_label = recall_label[active_indice]

        # ner的recall标签，填score
        x_scores = data_processor._score_test(text)
        for i in range(len(recall_label)):
            if recall_score[i] == 0:
                recall_score[i] = x_scores[recall_label[i]] + 0.2

        # 计算 whitening 分数
        #sim_score = recall_score.copy()
        #for i in range(len(recall_label)):
        #    sim_score[i] = get_sim(text, id2label[recall_label[i]])

        if len(final_pred):
            #final_score = recall_score / 6 + sim_score / 15 + final_pred
            final_score = recall_score / 4 + final_pred
            final_score = np.argsort(final_score)[::-1]
            recall_label = recall_label[final_score]

            num = num + 1
            ji, ban, dou = text.count("```及"), \
                           text.count("```伴"), \
                           text.count(";")+text.count("；")+text.count("，")+ \
                           text.count(",")+text.count("、") #+text.count("/")
            if (ji + ban + dou + 1) > num:
                num = ji + ban + dou + 1
            if num == 1:
                tmp_dict['normalized_result'].append(recall_label[0])
            elif num == 2:
                tmp_dict['normalized_result'].extend(recall_label[:2].tolist())
            else:
                sum_ = max((ji + ban + dou + 1), num, 3)
                tmp_dict['normalized_result'].extend(recall_label[:sum_].tolist())
            tmp_dict['normalized_result'] = [id2label[idx] for idx in tmp_dict['normalized_result']]

        if len(tmp_dict['normalized_result']) == 0:
            tmp_dict['normalized_result'] = [text]
        tmp_dict['normalized_result'] = "##".join(tmp_dict['normalized_result'])
        pred_result.append(tmp_dict)

    with open(output_dir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))


def predict_to_file(out_file):
    pickle_path = './data/CHIP-CDN'

    # 蕴含模型预测
    test_samples, recall_orig_test_samples, recall_orig_test_samples_scores = data_processor.get_test_sample(dtype='cls')

    if os.path.exists(os.path.join(pickle_path, 'cls_preds.pkl')):
        # 从cache装入
        with open(os.path.join(pickle_path, 'cls_preds.pkl'), "rb") as f:
            cls_preds = pickle.load(f)
        print("Load cls_preds.pkl ...")
    else:
        # 重新计算
        cls_preds = np.zeros(len(test_samples['text1']), dtype=np.float32)
        for n, l in tqdm(enumerate(zip(test_samples['text1'], test_samples['text2'])), total=len(test_samples['text1'])):
            if l[1]=='[BLANK]':
                continue
            pred = predict_score(l[0], l[1])
            cls_preds[n] = pred[0]

        cls_preds = cls_preds.reshape(len(cls_preds) // RECALL_K_TOTAL, RECALL_K_TOTAL)

        # 保存 cache
        with open(os.path.join(pickle_path, 'cls_preds.pkl'), "wb") as f:
            pickle.dump(cls_preds, f)
            print("Saving cls_preds.pkl ...")


    # 数量模型预测
    test_samples = data_processor.get_test_sample(dtype='num')
    orig_texts = data_processor.get_test_orig_text()

    if os.path.exists(os.path.join(pickle_path, 'num_preds.pkl')):
        # 从cache装入
        with open(os.path.join(pickle_path, 'num_preds.pkl'), "rb") as f:
            num_preds = pickle.load(f)
        print("Load num_preds.pkl ...")
    else:
        # 重新计算
        num_preds = None
        for l in tqdm(test_samples['text1']):
            pred = predict_num(l)[0].argmax()
            if num_preds is None:
                num_preds = np.array([pred])
            else:
                num_preds = np.append(num_preds, [pred], axis=0)

        # 保存 cache
        with open(os.path.join(pickle_path, 'num_preds.pkl'), "wb") as f:
            pickle.dump(num_preds, f)
            print("Saving num_preds.pkl ...")

    recall_labels = np.array(recall_orig_test_samples['recall_label'])
    cdn_commit_prediction(orig_texts, cls_preds, num_preds, recall_labels, 
        recall_orig_test_samples_scores, out_file, data_processor.id2label)


if __name__ == '__main__':
    # 先run prepare_data2.py 生成 test数据
    predict_to_file('CHIP-CDN_test.json')
