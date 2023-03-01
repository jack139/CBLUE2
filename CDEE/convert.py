import json

from dicts2 import text2new, value2new

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
    nn = 0
    max_n = 0
    D = []
    for l in json.load(open(filename)):
        l['text'] = '[unused1]'+l['text'].upper().strip() # 对 value为空的数据，指向这里 ‘[unused1]’,, 字母大写
        if len(l['text'])>512:
            max_n += 1
        spo_list = []
        for event in l['event']:
            if event["core_name"] is None:
                event["core_name"] = "[unused1]"  # 空的 core name 也使用 [unused1]
            else:
                event["core_name"] = event["core_name"].upper().strip() # 字母大写

            all_none = True
            for predicate in ['tendency', 'character', 'anatomy_list']:
                if event[predicate]:
                    all_none = False
                    if predicate=='tendency':
                        spo_list.append({
                            'predicate'    : predicate,
                            'subject'      : event['core_name'],
                            'object'       : event[predicate].upper().strip()
                        })
                    else:
                        for v in event[predicate]:
                            spo_list.append({
                                'predicate'    : predicate,
                                'subject'      : event['core_name'],
                                'object'       : v.upper().strip()
                            })
            if all_none:
                spo_list.append({
                    'predicate'    : "character",
                    'subject'      : event['core_name'],
                    'object'       : "[unused1]"
                })

        for x in spo_list:
            s_idx = search(x['subject'], l['text'])
            if s_idx==[]: # 进行替换处理
                if x['subject'] in text2new.keys():
                    for xx in text2new[x['subject']]: # 尝试多个值
                        s_idx = search(xx, l['text'])
                        if len(s_idx)>0:
                            x['subject'] = xx # 修改 subject
                            break

            o_idx = search(x['object'], l['text'])
            if o_idx==[]: # 进行替换处理
                if x['object'] in value2new.keys():
                    for xx in value2new[x['object']]: # 尝试多个值
                        o_idx = search(xx, l['text'])
                        if len(o_idx)>0:
                            x['object'] = xx # 修改 value
                            break

            # 检查缺失
            if s_idx==[] or o_idx==[]:
                if x['predicate']=='tendency':
                    print(s_idx, o_idx, l['text'], x)
                    nn += 1
                if x['predicate']!='tendency' and s_idx==[]:
                    print(s_idx, o_idx, l['text'], x)
                    nn += 1
                if x['predicate']!='tendency' and o_idx==[]:
                    print(s_idx, o_idx, l['text'], x)
                    nn += 1

            #elif len(o_idx)>1 and len(s_idx)>1:
            #    print(s_idx, o_idx, l['text'], x)

            elif len(s_idx)>1: # 处理 多个 subject 情况，只保留，离 value最近的
                               # o_idx 只对比第1个， ？？？可能会误判
                minp = abs(s_idx[0]-o_idx[0])
                minx = s_idx[0]
                for p in s_idx:
                    if abs(p-o_idx[0])<minp:
                        minp = abs(p-o_idx[0])
                        minx = p
                #print(s_idx)
                s_idx = [minx]
                #print(s_idx, l['text'], x)

            # 位置写入到 json 中
            # 不写入，因为 bert 编码后长度会改变，例如 "2017" 会变成 长度为 1
            #x['s_idx'] = s_idx[0]
            #x['o_idx'] = o_idx[0]


        D.append({
            'id' : l['id'],
            'text' : l['text'],
            'spo_list' : spo_list,
        })

    print(max_n, nn)

    return D


def check_file(infile, outfile, write_file=False):
    data = check_data(infile)

    if write_file:
        with open(outfile, 'w') as output_data:
            for json_content in data:
                output_data.write(json.dumps(json_content, ensure_ascii=False) + '\n')

if __name__ == '__main__':

    # 训练数据使用 dicts
    # 验证数据使用 dicts2 -- 与之前保持一致，方便比较

    #infile = '../dataset/CHIP-CDEE/CHIP-CDEE_train.json'
    #outfile = './data/CHIP-CDEE_train.jsonl'
    #check_file(infile, outfile, True)

    infile = '../dataset/CHIP-CDEE/CHIP-CDEE_dev.json'
    outfile = './data/CHIP-CDEE_dev.jsonl'
    check_file(infile, outfile, True)
