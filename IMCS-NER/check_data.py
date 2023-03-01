import json

def load_data(filename, text_name='sentence'):
    """加载数据
    单条格式：(文本, 标签id)
    """
    max_len = 0    
    max_cnt = 0

    data = json.load(open(filename))

    for k in data.keys():
        for l in data[k]['dialogue']:
            if len(l[text_name])>256:
                print(len(l[text_name]), l[text_name][:10])
                max_cnt += 1
            if len(l[text_name])>max_len:
                max_len = len(l[text_name])
    return max_len, max_cnt

if __name__ == '__main__':
    print('train:', load_data('../dataset/IMCS-NER/IMCS_train.json'))
    print('dev:', load_data('../dataset/IMCS-NER/IMCS_train.json'))
