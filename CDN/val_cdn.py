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
    #dev_json='test_dev.json'
)


def cdn_commit_prediction(text, preds, num_preds, recall_labels, recall_scores, output_dir, id2label, val_labels):
    text1 = text

    test_num = 5

    n = [0] * test_num
    m = [0] * test_num
    k = [0] * test_num


    pred_result = []
    active_indices = (preds >= 0.5)
    for text, active_indice, pred, num, recall_label, recall_score, val_label in \
        tqdm(zip(text1, active_indices, preds, num_preds, recall_labels, recall_scores, val_labels), total=len(text1)):

        tmp_dict = {'text': text, 'normalized_result': [], 'top_10': []}
        top_10, top_10_score = [], []

        final_pred = pred[active_indice]
        recall_score = recall_score[active_indice]
        recall_label = recall_label[active_indice]
        
        # ner的recall标签，填score
        #text = text_process.clean(text) # 文本做前处理
        #text = data_processor._process_single_sentence(text, mode="text")
        x_scores = data_processor._score_test(text)
        for i in range(len(recall_label)):
            if recall_score[i] == 0:
                recall_score[i] = x_scores[recall_label[i]] + 0.2

        # 计算 whitening 分数
        sim_score = recall_score.copy()
        #for i in range(len(recall_label)):
        #    sim_score[i] = get_sim(text, id2label[recall_label[i]])

        for xxx in range(test_num):
            tmp_dict['normalized_result'] = []

            if len(final_pred):
                if xxx==0:
                    final_score2 = recall_score / 3 + final_pred
                elif xxx==1:
                    final_score2 = recall_score / 4 + final_pred
                elif xxx==2:
                    final_score2 = recall_score / 5 + final_pred
                elif xxx==3:
                    final_score2 = recall_score / 6 + final_pred
                elif xxx==4:
                    final_score2 = recall_score / 2 + final_pred
                elif xxx==5:
                    final_score2 = recall_score / 4 + sim_score / 6 + final_pred
                elif xxx==6:
                    final_score2 = recall_score / 4 + sim_score / 6 + final_pred
                elif xxx==7:
                    final_score2 = recall_score / 4 + sim_score / 6 + final_pred
                elif xxx==8:
                    final_score2 = recall_score / 4 + sim_score / 6 + final_pred
                else: # xxx==9
                    final_score2 = recall_score / 4 + sim_score / 6 + final_pred

                final_score = np.argsort(final_score2)[::-1]
                #sim_max_idx = sim_score.argmax() # sim_score 最大的
                #final_score = np.delete(final_score, np.where(final_score==sim_max_idx)) # 删除sim 最大的
                #final_score = np.insert(final_score, 0, sim_max_idx) # 添加到首个
                recall_label2 = recall_label[final_score]

                num2 = num + 1
                ji, ban, dou = text.count("```及"), \
                               text.count("```伴"), \
                               text.count(";")+text.count("；")+text.count("，")+ \
                               text.count(",")+text.count("、")#+text.count("/")
                if (ji + ban + dou + 1) > num2:
                    num2 = ji + ban + dou + 1
                if num2 == 1:
                    tmp_dict['normalized_result'].append(recall_label2[0])
                elif num2 == 2:
                    tmp_dict['normalized_result'].extend(recall_label2[:2].tolist())
                else:
                    sum_ = max((ji + ban + dou + 1), num2, 3)
                    tmp_dict['normalized_result'].extend(recall_label2[:sum_].tolist())
                tmp_dict['normalized_result'] = [id2label[idx] for idx in tmp_dict['normalized_result']]

                n[xxx] += 1 # 预测的数量

                top_10 = [str((id2label[idx], idx)) for idx in recall_label2[:10].tolist()]
                top_10_score = ['%.4f, %.4f, %.4f, %.4f'%(final_score2[idx], recall_score[idx], final_pred[idx], sim_score[idx]) \
                    for idx in final_score[:10].tolist()]

            if len(tmp_dict['normalized_result']) == 0:
                tmp_dict['normalized_result'] = [tmp_dict['text']]
            
            tmp_dict['origin_label'] = [id2label[idx] for idx in val_label]  # 正确答案 

            # 统计
            m[xxx] += 1 # 总数量
            if set(tmp_dict['normalized_result'])==set(tmp_dict['origin_label']):
                k[xxx] += 1 # 正确的数量

            tmp_dict['normalized_result'] = "##".join(tmp_dict['normalized_result'])
            tmp_dict['origin_label'] = "##".join(tmp_dict['origin_label'])
            tmp_dict['top_10'] = [ v+" "+k for k,v in zip(top_10, top_10_score)]

        # 只会保存最后一次的
        pred_result.append(tmp_dict)


    with open(output_dir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))


    for xxx in range(test_num):
        P = k[xxx] / n[xxx]
        R = k[xxx] / m[xxx]
        F1 = 2 * P * R / (P+R)
        print("%d:\tP= %.4f\tR= %.4f\tf1 = %.4f"%(xxx, P, R, F1))


def predict_to_file(out_file):
    pickle_path = './data/CHIP-CDN'

    # 蕴含模型预测
    eval_samples, recall_orig_eval_samples, recall_orig_eval_samples_scores = data_processor.get_val_sample(dtype='cls', do_augment=DO_AUGMENT)

    if os.path.exists(os.path.join(pickle_path, 'val_cls_preds.pkl')):
        # 从cache装入
        with open(os.path.join(pickle_path, 'val_cls_preds.pkl'), "rb") as f:
            cls_preds = pickle.load(f)
        print("Load val_cls_preds.pkl ...")
    else:
        # 重新计算
        cls_preds = np.zeros(len(eval_samples['text1']), dtype=np.float32)
        for n, l in tqdm(enumerate(zip(eval_samples['text1'], eval_samples['text2'])), total=len(eval_samples['text1'])):
            if l[1]=='[BLANK]':
                continue
            pred = predict_score(l[0], l[1])
            cls_preds[n] = pred[0]

        cls_preds = cls_preds.reshape(len(cls_preds) // RECALL_K_TOTAL, RECALL_K_TOTAL)

        # 保存 cache
        with open(os.path.join(pickle_path, 'val_cls_preds.pkl'), "wb") as f:
            pickle.dump(cls_preds, f)
            print("Saving val_cls_preds.pkl ...")


    # 数量模型预测
    eval_samples = data_processor.get_val_sample(dtype='num')
    orig_texts = data_processor.get_val_orig_text()

    if os.path.exists(os.path.join(pickle_path, 'val_num_preds.pkl')):
        # 从cache装入
        with open(os.path.join(pickle_path, 'val_num_preds.pkl'), "rb") as f:
            num_preds = pickle.load(f)
        print("Load val_num_preds.pkl ...")
    else:
        # 重新计算
        num_preds = None
        for l in tqdm(eval_samples['text1']):
            pred = predict_num(l)[0].argmax()
            if num_preds is None:
                num_preds = np.array([pred])
            else:
                num_preds = np.append(num_preds, [pred], axis=0)

        # 保存 cache
        with open(os.path.join(pickle_path, 'val_num_preds.pkl'), "wb") as f:
            pickle.dump(num_preds, f)
            print("Saving val_num_preds.pkl ...")

    recall_labels = np.array(recall_orig_eval_samples['recall_label'])
    orig_labels = np.array(recall_orig_eval_samples['label'])
    cdn_commit_prediction(orig_texts, cls_preds, num_preds, recall_labels, 
        recall_orig_eval_samples_scores, out_file, data_processor.id2label, orig_labels)


if __name__ == '__main__':
    # 先run prepare_data2.py 生成 test数据
    predict_to_file('data/CHIP-CDN_val.json')
