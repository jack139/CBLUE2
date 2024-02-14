import os
import json

data1 = json.load(open("Text2DT_test_result.json"))
data2 = json.load(open("data/Text2DT/Text2DT_test.json"))

for n, i in enumerate(data2):
	data1[n]['text'] = i['text']

json.dump(
    data1,
    open("Text2DT_test.json", 'w', encoding='utf-8'),
    indent=4,
    ensure_ascii=False
)
