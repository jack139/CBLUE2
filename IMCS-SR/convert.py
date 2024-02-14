import json
from norms import sym2id

'''
[unused1]   "0" 代表确定病人没有患有该症状，
[unused2]   "1" 代表确定病人患有该症状，
[unused3]   "2" 代表无法根据上下文确定病人是否患有该症状。
'''

norms = list(sym2id.keys())


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


def check_data(filename, include_blank):
    nn = 0

    D = []
    text = ''
    entities = []
    all_idx = 0
    start_idx = 0
    etype = ''


    data = json.load(open(filename))

    for k in data.keys():
        for l in data[k]['dialogue']:
            text = l['sentence']

            # 找到实体
            for label in l['BIO_label'].split():
                if label[0]=='O':
                    if etype!='' and etype=='Symptom':
                        entities.append(text[start_idx:all_idx])
                    start_idx = 0
                    etype = ''
                elif label[0]=='B':
                    if etype!='' and etype=='Symptom':
                        entities.append(text[start_idx:all_idx])
                    start_idx = all_idx
                    etype = label.split('-')[1]
                elif label[0]=='I':
                    pass
                else:
                    print('unknown label: ', label)

                all_idx += 1


            # 一行text结束
            if etype!='' and etype=='Symptom':
                entities.append(text[start_idx:all_idx])

            assert len(entities)==len(l['symptom_norm'])

            text = '[unused1][unused2][unused3]'+text
            spo_list = []
            for entity, sym_norm, sym_type in zip(entities, l['symptom_norm'], l['symptom_type']):
                if sym_norm not in norms:
                    print(sym_norm, "not in norms")
                    continue
                #assert sym_norm in norms
                spo_list.append({
                    'predicate'    : sym_norm,
                    'subject'      : entity,
                    'object'       : "[unused%d]"%(int(sym_type)+1)
                })
                nn += 1

            for x in spo_list:
                s_idx = search(x['subject'], text)
                assert s_idx!=[]

                o_idx = search(x['object'], text)
                assert o_idx!=[]

                if len(s_idx)>1: # 处理 多个 subject 情况，只保留，离 value最近的
                                   # o_idx 只对比第1个， ？？？可能会误判
                    minp = abs(s_idx[0]-o_idx[0])
                    minx = s_idx[0]
                    for p in s_idx:
                        if abs(p-o_idx[0])<minp:
                            minp = abs(p-o_idx[0])
                            minx = p
                    #print(s_idx)
                    s_idx = [minx]
                    #print(s_idx, text, x)

                # 位置写入到 json 中
                # 不写入，因为 bert 编码后长度会改变，例如 "2017" 会变成 长度为 1
                #x['s_idx'] = s_idx[0]
                #x['o_idx'] = o_idx[0]


            if include_blank or len(spo_list)>0:
                D.append({
                    'text' : text,
                    'spo_list' : spo_list,
                })

            text = ''
            entities = []
            all_idx = 0
            start_idx = 0
            etype = ''


    print(len(D), nn)

    return D


def check_file(infile, outfile, write_file=False, include_blank=True):
    print(f"{infile} --> {outfile}")

    data = check_data(infile, include_blank)

    if write_file:
        with open(outfile, 'w') as output_data:
            for json_content in data:
                output_data.write(json.dumps(json_content, ensure_ascii=False) + '\n')


if __name__ == '__main__':

    infile = '../dataset/IMCS-IR/new_split/data/IMCS-V2_train.json'
    #infile = '../dataset/3.0/IMCS-V2/IMCS-V2_train.json'
    outdir = './data/train.jsonl'
    check_file(infile, outdir, True)

    outdir = './data/train_no_blank.jsonl'
    check_file(infile, outdir, True, False)

    infile = '../dataset/IMCS-IR/new_split/data/IMCS-V2_dev.json'
    #infile = '../dataset/3.0/IMCS-V2/IMCS-V2_dev.json'
    outdir = './data/dev.jsonl'
    check_file(infile, outdir, True)
