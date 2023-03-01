import os
import pickle
import json


data_path = "data"

for file in os.listdir(data_path):
    filename, ext = os.path.splitext(file)
    if ext!='.pk':
        continue

    with open(os.path.join(data_path, file), "rb") as f:
        data = pickle.load(f)

    json.dump(
        data,
        open(os.path.join(data_path, filename+'.json'), 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    print(file, '... ok')