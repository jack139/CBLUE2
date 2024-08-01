## CBLUE 2/3 任务实现

[CBLUE 中文生物医学语言理解评测](https://tianchi.aliyun.com/cblue)

[CBLUE 1 Baseline](https://github.com/CBLUEbenchmark/CBLUE)



### CBLUE 2 内容

- [x] 中文医学命名实体识别（CMeEE）
- [x] 中文医学文本实体关系抽取（CMeIE）
- [x] 临床术语标准化任务（CHIP-CDN）
- [x] 临床试验筛选标准短文本分类（CHIP-CTC）
- [x] 平安医疗科技疾病问答迁移学习（CHIP-STS）
- [x] 医疗搜索检索词意图分类（KUAKE-QIC）
- [x] 医疗搜索查询词-页面标题相关性（KUAKE-QTR）
- [x] 医疗搜索查询词-查询词相关性（KUAKE-QQR）
- [x] 蕴含实体的中文医疗对话生成（MedDG）
- [x] 医疗对话临床发现阴阳性判别（CHIP-MDCFNPC）
- [x] 临床发现事件抽取（CHIP-CDEE）
- [x] 命名实体识别（IMCS-NER）
- [x] 对话意图识别（IMCS-IR）
- [x] 症状的归一化标签和类别标签预测（IMCS-SR）
- [x] 自动生成诊疗报告（IMCS-MRG）



### CBLUE 3 新增

- [x] 中文医学命名实体识别V2（CMeEE-V2）
- [x] 中文医学文本实体关系抽取V2（CMeIE-V2）
- [x] 医学因果关系抽取（CMedCausal）
- [x] 医学段落检索（KUAKE-IR）
- [x] 命名实体识别V2（IMCS-V2-NER）
- [x] 对话意图识别V2（IMCS-V2-DAC）
- [x] 症状的归一化标签和类别标签预测V2（IMCS-V2-SR）
- [x] 自动生成诊疗报告V2（IMCS-V2-MRG）
- [x] 医疗文本诊疗决策树抽取（Text2DT）




| 数据集/语言模型 | BERT-base | RoBERTa-large | Other models |
| --------------- | :-------: | :-------: | :-------: |
| CMeEE-F1        |  67.840   |           |           |
| CMeIE-F1        |  56.569   |  59.340   |           |
| CHIP-CDN-F1     |  60.809   |           |           |
| CHIP-CTC-F1     |  70.433   |           |           |
| CHIP-STS-F1     |  85.310   |  84.914   |           |
| KUAKE-QIC-Acc   |  85.908   |           |           |
| KUAKE-QTR-Acc   |  61.154   |           |           |
| KUAKE-QQR-Acc   |  85.026   |           |           |
| MedDG-Ave       |           |           |  15.827   |
| CHIP-MDCFNPC-F1 |  77.908   |           |           |
| CHIP-CDEE-F1    |  41.198   |  50.2175  |           |
| IMCS-NER-F1     |  92.028   |           |           |
| IMCS-SR-F1      |  64.665   |  65.662   |           |
| IMCS-MRG-Ave    |           |           |  59.640   |
| IMCS-IR-F1      |  77.887   |           |           |
| CMeEE-V2-F1     |  73.2150  |  74.0448  |           |
| CMeIE-V2-F1     |  52.1963  |  54.8591  |           |
| CMedCausal-F1   |  32.6060  |  36.2170  |           |
| KUAKE-IR-MRR@10 |  21.5994  |           |           |
| IMCS-V2-NER-F1  |  88.2173  |  88.0096  |           |
| IMCS-V2-DAC-Acc |  82.6112  |           |           |
| IMCS-V2-SR-Utterance-Level-F1 |  64.8370  |  68.2290  |         |
| IMCS-V2-SR-Dialog-Level-F1    |  62.3330  |  65.1595  |         |
| IMCS-V2-MRG-Ave |           |           |  51.1026  |
| Text2DT_Tree_Level_Score      |  47.9338  |           |         |
