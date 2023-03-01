import sys
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Decode")
parser.add_argument("-p", dest="result_path", required=True, type=str, help="摘要文件的路径") 
args = parser.parse_args()

result_path = args.result_path

D = {}

files = os.listdir(result_path)

for f in files:
    fn = os.path.join(result_path, f)
    sentence_id = f.split('_')[1]
    text = open(fn).read().strip().replace(' ', '')
    D[sentence_id] = { 'report' : text }

json.dump(
    D,
    open('IMCS-MRG_test.json', 'w', encoding='utf-8'),
    indent=4,
    ensure_ascii=False
)
