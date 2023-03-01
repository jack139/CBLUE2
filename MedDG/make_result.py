import os
import pickle, json


test_path = "../dataset/MedDG/MedDG_test.json"
result_path = "./data/ans_max_confidence.pk"

test_data = json.load(open(test_path))

with open(result_path, "rb") as f:
    result_data = pickle.load(f)

final_result = []

for item, output in zip(test_data, result_data):
    item['output'] = output
    final_result.append(item)

json.dump(
    final_result,
    open('MedDG_test.json', 'w', encoding='utf-8'),
    indent=4,
    ensure_ascii=False
)

print("测试数据预处理完成", len(test_data))
