import os
import pickle, json
from sklearn.model_selection import train_test_split


all_data_path = "../dataset/MedDG/MedDG_train.json"
all_test_path = "../dataset/MedDG/MedDG_test.json"

train_data_path = "./data/train_data.pk"
dev_data_path = "./data/dev_data.pk"
test_data_path = "./data/test_add_info.pk"


# 训练数据
all_data = json.load(open(all_data_path))

train_data, dev_data = train_test_split(all_data, test_size=0.1, random_state=123456)

with open(train_data_path, "wb") as f:
    pickle.dump(train_data, f)

with open(dev_data_path, "wb") as f:
    pickle.dump(dev_data, f)

print("训练数据预处理完成", len(all_data))


# 测试数据
test_data = json.load(open(all_test_path))

test_add_info = []

for data_item in test_data:
    
    test_add_info_item = []
    
    for s in data_item['history']:
        s_label = "Patient" if s[:3]=="患者：" else "Doctor"

        test_add_info_item.append(
            {
                'id': s_label,
                'Sentence': s[3:],
                'Symptom': [],
                'Medicine': [],
                'Test': [],
                'Attribute': [],
                'Disease': []
            }
        )
    
    test_add_info.append(test_add_info_item)

with open(test_data_path, "wb") as f:
    pickle.dump(test_add_info, f)

#json.dump(
#    test_add_info,
#    open(test_data_path+'.json', 'w', encoding='utf-8'),
#    indent=4,
#    ensure_ascii=False
#)

print("测试数据预处理完成", len(test_data))
