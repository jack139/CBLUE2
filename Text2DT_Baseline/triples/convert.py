import json

''' 目标格式
{
    "text": "急性胰腺炎@有研究显示，进行早期 ERCP （24 小时内）可以降低梗阻性胆总管结石患者的并发症发生率和死亡率； "
        "但是，对于无胆总管梗阻的胆汁性急性胰腺炎患者，不需要进行早期 ERCP。", 
    "spo_list": [
        {
            "predicate": "1", 
            "subject": "急性胰腺炎", 
            "object": "ERCP", 
        }
    ]
}
'''


def check_data(filename):
    max_n = total = 0
    D = []
    predicate = set()
    for l in json.load(open(filename)):
        spo_list = []

        if len(l['text'])>128:
            max_n += 1

        for node in l['tree']:
            for event in node['triples']:
                predicate.add(event[1])

                spo_list.append({
                    'predicate'    : event[1],
                    'subject'      : event[0],
                    'object'       : event[2],
                })

                total += 1

        D.append({
            'text' : l['text'],
            'spo_list' : spo_list,
        })

        #break

    print(list(predicate))
    print(max_n, total)

    return D


def check_file(infile, outfile, write_file=False):
    data = check_data(infile)

    if write_file:
        with open(outfile, 'w') as output_data:
            for json_content in data:
                output_data.write(json.dumps(json_content, ensure_ascii=False) + '\n')


if __name__ == '__main__':

    infile = '../data/Text2DT/Text2DT_train.json'
    outfile = 'data/train.jsonl'
    check_file(infile, outfile, True)

    infile = '../data/Text2DT/Text2DT_dev.json'
    outfile = './data/dev.jsonl'
    check_file(infile, outfile, True)
