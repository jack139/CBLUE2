#! -*- coding:utf-8 -*-

from baseline.data_process import CDNDataProcessor
from ner.CMeEE_gp import NER

RECALL_K=400
NUM_NEGATIVE_SAMPLES=5
DO_AUGMENT=6

def prepare_data(data_dir):
    data_processor = CDNDataProcessor(root=data_dir, recall_k=RECALL_K, negative_sample=NUM_NEGATIVE_SAMPLES, NER=NER,
        #train_json='example_gold.json',
        #dev_json='test_dev.json'
        #test_json='test.json'
    )

    # 蕴含模型
    train_samples, recall_orig_train_samples, recall_orig_train_samples_scores = data_processor.get_train_sample(dtype='cls', do_augment=DO_AUGMENT)
    eval_samples, recall_orig_eval_samples, recall_orig_eval_samples_scores = data_processor.get_dev_sample(dtype='cls', do_augment=DO_AUGMENT)
    test_samples, recall_orig_test_samples, recall_orig_test_samples_scores = data_processor.get_test_sample(dtype='cls')
    eval_samples, recall_orig_eval_samples, recall_orig_eval_samples_scores = data_processor.get_val_sample(dtype='cls', do_augment=DO_AUGMENT)

    # 数量模型
    #train_samples = data_processor.get_train_sample(dtype='num', do_augment=1)
    #eval_samples = data_processor.get_dev_sample(dtype='num')
    #test_samples = data_processor.get_test_sample(dtype='num')

    # 初始化 NER 数据, CDNDataProcessor初始化时会自动执行，不需要显示执行
    #data_processor._NER_train_set()
    #data_processor._NER_ICD_set()

if __name__ == '__main__':
    prepare_data('./data')

