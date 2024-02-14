## 向量检索方法

CoSent/SimCSE 召回， Bert 筛选

```bash
# 生成训练数据
python3 convert.py

# 训练 CoSent/SimCSE模型 (keras)
python3 IR_cosent.py

# 生成向量 (keras)
python3 cosent_vecs.py

# 检索, 生成找回数据 (pytorch 2.0)
python3.9 cosent_search.py

# 判别模型目录
cd ../KUAKE-IR

# 生成判别模型训练数据
python3 convert.py

# 训练判别模型 IR_gp
python3 IR_gp.py

# 生成结果 （需要 IR_gp 模型）
python3 bert_cosent_test.py
```



## 结果比较

- CoSent 21.5994
- SimCSE 11.8766
