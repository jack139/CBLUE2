## BM25 方法

```bash
python3 bm25_test.py
```



## Bert 方法

BM25 召回， Bert 筛选

```bash
# 生成训练数据
python3 convert.py

# 判别模型训练 (keras)
python3 IR_gp.py

# 生成结果
python3 bert_test.py
```
