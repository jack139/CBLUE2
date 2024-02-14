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
            "object2": "ERCP", 
            "s_idx": 1,
            "o_idx": 1,
            "o2_idx": 1,
        }
    ]
}
'''


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    pos = []
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            pos.append(i)
    return pos


def check_data(filename):
    max_n = nn = total = 0
    D = []
    for l in json.load(open(filename)):
        spo_list = []

        if len(l['text'])>512:
            max_n += 1

        for event in l['relation_of_mention']:
            if event['relation']!=2:

                # 检查位置
                s_idx = search(event['head']['mention'], l['text'])
                if event['head']['start_idx'] not in s_idx:
                    print('head', s_idx, event['head']['mention'], l)
                    nn += 1
                else:
                    s_idx = s_idx.index(event['head']['start_idx'])

                o_idx = search(event['tail']['mention'], l['text'])
                if event['tail']['start_idx'] not in o_idx:
                    print('tail', s_idx, event['tail']['mention'], l)
                    nn += 1
                else:
                    o_idx = o_idx.index(event['tail']['start_idx'])

                spo_list.append({
                    'predicate'    : str(event['relation']),
                    'subject'      : event['head']['mention'],
                    'object'       : event['tail']['mention'],
                    'object2'      : event['tail']['mention'], # 使用 相同的 object
                    's_idx'        : s_idx,
                    'o_idx'        : o_idx,
                    'o2_idx'       : o_idx,
                })

                total += 1

            else:

                # 检查位置
                s_idx = search(event['head']['mention'], l['text'])
                if event['head']['start_idx'] not in s_idx:
                    print('head', s_idx, event['head']['mention'], l)
                    nn += 1
                else:
                    s_idx = s_idx.index(event['head']['start_idx'])

                o_idx = search(event['tail']['head']['mention'], l['text'])
                if event['tail']['head']['start_idx'] not in o_idx:
                    print('tail1', s_idx, event['tail']['head']['mention'], l)
                    nn += 1
                else:
                    o_idx = o_idx.index(event['tail']['head']['start_idx'])

                o2_idx = search(event['tail']['tail']['mention'], l['text'])
                if event['tail']['tail']['start_idx'] not in o2_idx:
                    print('tail2', s_idx, event['tail']['tail']['mention'], l)
                    nn += 1
                else:
                    o2_idx = o2_idx.index(event['tail']['tail']['start_idx'])

                spo_list.append({
                    'predicate'    : str(event['relation']),
                    'subject'      : event['head']['mention'],
                    'object'       : event['tail']['head']['mention'],
                    'object2'      : event['tail']['tail']['mention'],
                    's_idx'        : s_idx,
                    'o_idx'        : o_idx,
                    'o2_idx'       : o2_idx,
                })

                total += 1


        D.append({
            'text' : l['text'],
            'spo_list' : spo_list,
        })

        #break

    print(max_n, nn, total)

    return D


def check_file(infile, outfile, write_file=False):
    data = check_data(infile)

    if write_file:
        with open(outfile, 'w') as output_data:
            for json_content in data:
                output_data.write(json.dumps(json_content, ensure_ascii=False) + '\n')


if __name__ == '__main__':

    infile = '../dataset/3.0/CMedCausal/CMedCausal_train.json'
    outfile = './data/train.jsonl'
    check_file(infile, outfile, True)

    infile = '../dataset/3.0/CMedCausal/CMedCausal_dev.json'
    outfile = './data/dev.jsonl'
    check_file(infile, outfile, True)
