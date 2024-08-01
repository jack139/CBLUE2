## bge-m3 + bge-reranker 模型

```bash
# 生成向量库
python3.9 filter_bge.py -t 4 -p 0
python3.9 filter_bge.py -t 4 -p 1
python3.9 filter_bge.py -t 4 -p 2
python3.9 filter_bge.py -t 4 -p 3

# 生成结果
python3.9 bge_test.py
```



## 结果比较

- Top=50 30.4581
- Top=100 30.5164
