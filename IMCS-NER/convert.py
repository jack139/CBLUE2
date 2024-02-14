import os
import json

'''
  {
    "text": "对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。",
    "entities": [
      {
        "start_idx": 3,
        "end_idx": 9,
        "type": "bod",
        "entity": "SARST细胞"
      },
      {
        "start_idx": 19,
        "end_idx": 24,
        "type": "dis",
        "entity": "成人SARS"
      }
    ]
  },

'''

def convert(infile, outfile, include_blank=True):

    print(f"{infile} --> {outfile}")

    D = []
    text = ''
    entities = []
    all_idx = 0
    start_idx = 0
    etype = ''

    data = json.load(open(infile))

    for k in data.keys():
        for diag in data[k]['dialogue']:
            text = diag['sentence']

            for label in diag['BIO_label'].split():
                if label[0]=='O':
                    if etype!='':
                        entities.append({
                            "start_idx": start_idx,
                            "end_idx": all_idx - 1,
                            "type": etype,
                            "entity": text[start_idx:all_idx],
                        })
                    start_idx = 0
                    etype = ''
                elif label[0]=='B':
                    if etype!='':
                        entities.append({
                            "start_idx": start_idx,
                            "end_idx": all_idx - 1,
                            "type": etype,
                            "entity": text[start_idx:all_idx],
                        })                
                    start_idx = all_idx
                    etype = label.split('-')[1]
                elif label[0]=='I':
                    pass
                else:
                    print('unknown label: ', label)

                all_idx += 1


            # 一行text结束
            if etype!='':
                entities.append({
                    "start_idx": start_idx,
                    "end_idx": all_idx - 1,
                    "type": etype,
                    "entity": text[start_idx:all_idx],
                })

            # 加入数据集
            if include_blank or len(entities)>0:
                D.append({
                    'text' : text,
                    'entities' : entities,
                })

            text = ''
            entities = []
            all_idx = 0
            start_idx = 0
            etype = ''

    json.dump(
        D,
        open(os.path.join(outfile), 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    print(len(D))


if __name__ == '__main__':
    convert('../dataset/IMCS-IR/new_split/data/IMCS-V2_train.json', './data/train.json', True)
    convert('../dataset/IMCS-IR/new_split/data/IMCS-V2_dev.json', './data/dev.json')
    #convert('../dataset/3.0/IMCS-V2/IMCS-V2_train.json', './data/train.json', True)
    #convert('../dataset/3.0/IMCS-V2/IMCS-V2_dev.json', './data/dev.json')
