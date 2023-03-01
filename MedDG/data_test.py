import json
import pickle
import copy
import numpy as np
from tqdm import tqdm


i2c = [
    ('None', 'None'),
    ('胃痛', 'Symptom'),
    ('肌肉酸痛', 'Symptom'),
    ('咽部痛', 'Symptom'),
    ('胃肠不适', 'Symptom'),
    ('咽部灼烧感', 'Symptom'),
    ('腹胀', 'Symptom'),
    ('稀便', 'Symptom'),
    ('肠梗阻', 'Symptom'),
    ('胸痛', 'Symptom'),
    ('饥饿感', 'Symptom'),
    ('烧心', 'Symptom'),
    ('寒战', 'Symptom'),
    ('气促', 'Symptom'),
    ('嗜睡', 'Symptom'),
    ('粘便', 'Symptom'),
    ('四肢麻木', 'Symptom'),
    ('腹痛', 'Symptom'),
    ('恶心', 'Symptom'),
    ('胃肠功能紊乱', 'Symptom'),
    ('反流', 'Symptom'),
    ('里急后重', 'Symptom'),
    ('鼻塞', 'Symptom'),
    ('体重下降', 'Symptom'),
    ('贫血', 'Symptom'),
    ('发热', 'Symptom'),
    ('过敏', 'Symptom'),
    ('痉挛', 'Symptom'),
    ('黑便', 'Symptom'),
    ('头晕', 'Symptom'),
    ('乏力', 'Symptom'),
    ('心悸', 'Symptom'),
    ('肠鸣', 'Symptom'),
    ('尿急', 'Symptom'),
    ('细菌感染', 'Symptom'),
    ('喷嚏', 'Symptom'),
    ('腹泻', 'Symptom'),
    ('焦躁', 'Symptom'),
    ('痔疮', 'Symptom'),
    ('精神不振', 'Symptom'),
    ('咳嗽', 'Symptom'),
    ('脱水', 'Symptom'),
    ('消化不良', 'Symptom'),
    ('食欲不振', 'Symptom'),
    ('月经紊乱', 'Symptom'),
    ('背痛', 'Symptom'),
    ('呼吸困难', 'Symptom'),
    ('吞咽困难', 'Symptom'),
    ('水肿', 'Symptom'),
    ('肛周疼痛', 'Symptom'),
    ('呕血', 'Symptom'),
    ('菌群失调', 'Symptom'),
    ('便血', 'Symptom'),
    ('口苦', 'Symptom'),
    ('淋巴结肿大', 'Symptom'),
    ('头痛', 'Symptom'),
    ('尿频', 'Symptom'),
    ('排气', 'Symptom'),
    ('黄疸', 'Symptom'),
    ('呕吐', 'Symptom'),
    ('有痰', 'Symptom'),
    ('打嗝', 'Symptom'),
    ('螺旋杆菌感染', 'Symptom'),
    ('胃复安', 'Medicine'),
    ('泮托拉唑', 'Medicine'),
    ('马来酸曲美布丁', 'Medicine'),
    ('磷酸铝', 'Medicine'),
    ('诺氟沙星', 'Medicine'),
    ('金双歧', 'Medicine'),
    ('人参健脾丸', 'Medicine'),
    ('三九胃泰', 'Medicine'),
    ('泌特', 'Medicine'),
    ('康复新液', 'Medicine'),
    ('克拉霉素', 'Medicine'),
    ('乳果糖', 'Medicine'),
    ('奥美', 'Medicine'),
    ('果胶铋', 'Medicine'),
    ('嗜酸乳杆菌', 'Medicine'),
    ('谷氨酰胺肠溶胶囊', 'Medicine'),
    ('四磨汤', 'Medicine'),
    ('思连康', 'Medicine'),
    ('多潘立酮', 'Medicine'),
    ('得舒特', 'Medicine'),
    ('肠溶胶囊', 'Medicine'),
    ('胃苏', 'Medicine'),
    ('蒙脱石散', 'Medicine'),
    ('益生菌', 'Medicine'),
    ('藿香正气丸', 'Medicine'),
    ('诺氟沙星胶囊', 'Medicine'),
    ('复方消化酶', 'Medicine'),
    ('布洛芬', 'Medicine'),
    ('硫糖铝', 'Medicine'),
    ('乳酸菌素', 'Medicine'),
    ('雷呗', 'Medicine'),
    ('莫沙必利', 'Medicine'),
    ('补脾益肠丸', 'Medicine'),
    ('香砂养胃丸', 'Medicine'),
    ('铝碳酸镁', 'Medicine'),
    ('马来酸曲美布汀', 'Medicine'),
    ('消炎利胆片', 'Medicine'),
    ('多酶片', 'Medicine'),
    ('思密达', 'Medicine'),
    ('阿莫西林', 'Medicine'),
    ('颠茄片', 'Medicine'),
    ('耐信', 'Medicine'),
    ('瑞巴派特', 'Medicine'),
    ('培菲康', 'Medicine'),
    ('吗叮咛', 'Medicine'),
    ('曲美布汀', 'Medicine'),
    ('甲硝唑', 'Medicine'),
    ('胶体果胶铋', 'Medicine'),
    ('吗丁啉', 'Medicine'),
    ('健胃消食片', 'Medicine'),
    ('兰索拉唑', 'Medicine'),
    ('马来酸曲美布汀片', 'Medicine'),
    ('莫沙比利', 'Medicine'),
    ('左氧氟沙星', 'Medicine'),
    ('斯达舒', 'Medicine'),
    ('抗生素', 'Medicine'),
    ('达喜', 'Medicine'),
    ('山莨菪碱', 'Medicine'),
    ('健脾丸', 'Medicine'),
    ('肠胃康', 'Medicine'),
    ('整肠生', 'Medicine'),
    ('开塞露', 'Medicine'),
    ('腹腔镜', 'Test'),
    ('小肠镜', 'Test'),
    ('糖尿病', 'Test'),
    ('CT', 'Test'),
    ('B超', 'Test'),
    ('呼气实验', 'Test'),
    ('肛门镜', 'Test'),
    ('便常规', 'Test'),
    ('尿检', 'Test'),
    ('钡餐', 'Test'),
    ('转氨酶', 'Test'),
    ('尿常规', 'Test'),
    ('胶囊内镜', 'Test'),
    ('肝胆胰脾超声', 'Test'),
    ('胃镜', 'Test'),
    ('结肠镜', 'Test'),
    ('腹部彩超', 'Test'),
    ('胃蛋白酶', 'Test'),
    ('血常规', 'Test'),
    ('肠镜', 'Test'),
    ('性质', 'Attribute'),
    ('诱因', 'Attribute'),
    ('时长', 'Attribute'),
    ('位置', 'Attribute'),
    ('胰腺炎', 'Disease'),
    ('肠炎', 'Disease'),
    ('肝硬化', 'Disease'),
    ('阑尾炎', 'Disease'),
    ('肺炎', 'Disease'),
    ('食管炎', 'Disease'),
    ('便秘', 'Disease'),
    ('胃炎', 'Disease'),
    ('感冒', 'Disease'),
    ('胆囊炎', 'Disease'),
    ('胃溃疡', 'Disease'),
    ('肠易激综合征', 'Disease')
]

c2i = { v:idx  for idx, v in enumerate(i2c) }
c2c = { v[0]:v  for v in i2c }


# 数据生成 来自 MedDG
def _read(file_path, is_pk=False):
    with open('data/160_last_topic2num.pk','rb') as f:
        topic2num = pickle.load(f)

    D = []

    with open(file_path, 'rb') as f:
        if is_pk:
            dataset = pickle.load(f)
        else:
            dataset = json.load(f)
        for dialog in tqdm(dataset):
            new_dialog = []
            history = []
            now_topic = []
            his_topic = []
            for sen in dialog:
                aa = sen['Symptom']+sen['Attribute']+sen['Test']+sen['Disease']+sen['Medicine']
                if len(aa) > 0:
                    if len(history) > 0 and sen['id'] == 'Doctor':
                        new_dialog.append({"history": copy.deepcopy(history), "next_sym": copy.deepcopy(aa), 'now_topic': copy.deepcopy(now_topic)})
                    now_topic.extend(aa)
                    his_topic.extend(aa)
                history.append(sen['Sentence'])
            for dic in new_dialog:
                future = copy.deepcopy(his_topic[len(dic['now_topic']):])
                #dic['future'] = [topic2num[i] for i in future]
                dic['future'] = future
                #dic['next_sym'] = [topic2num[i] for i in dic['next_sym']]
                #yield self.text_to_instance(dic)
                D.append(dic)

    json.dump(
        D,
        open('data/test_data.json', 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    return len(D)



def create_content_label(data_item):
    
    for idx in range(len(data_item)-1, 0, -1):
        
        if data_item[idx]['id'] == 'Doctor':
            
            label = data_item[idx].copy()

            del label['id']
            del label['Sentence']
            
            label = {k:set(v) for k,v in label.items()}

            content = ""
            append_entities = set()

            for s_idx in range(idx-1, -1, -1):
                
                sentence = data_item[s_idx]
                
                for k, vs in sentence.items():
                    if type(vs) == list:
                        for v in vs:
                            append_entities.add((v, k))
                            
                content = sentence['Sentence'] + content

            yield content, append_entities, label


# 数据生成 来自 alala_meddg
def _read2(file_path, is_pk=False):
    with open('data/160_last_topic2num.pk','rb') as f:
        topic2num = pickle.load(f)

    skip_epochs = 50
    epochs = 100
    prob = 1.0
    append_entities_len = 40

    D = []

    with open(file_path, 'rb') as f:
        if is_pk:
            dataset = pickle.load(f)
        else:
            dataset = json.load(f)
        for data_item in tqdm(dataset):
            
            for content, append_entities, label in create_content_label(data_item):
                
                if sum([len(v) for v in label.values()]) == 0:
                    if np.random.uniform() < prob:
                        continue

                #print(content)
                #print(label)
                #print(append_entities)
                
                # change prob
                prob -= (prob / epochs / skip_epochs)

                # 增加出现实体
                #append_entity_ids = [c2i[item] for item in append_entities if item in c2i]
                #append_entity_ids += [0] * (append_entities_len - len(append_entity_ids))


                # 生成答案
                #label_item = [2 for _ in i2c]
                #for k, v in label.items():
                #    for v_item in v:
                #        label_item[c2i[(v_item, k)]] = 1
                
                D.append({
                    #'append_entity_ids' : append_entity_ids,
                    #'label_item' : label_item,
                    'append_entities' : list(append_entities), # 去掉 set
                    'label' : {k:list(v) for k,v in label.items()}, # 去掉 set
                    'content' : content
                })

    json.dump(
        D,
        open('data/test_data2.json', 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    return len(D)


# 数据生成 修改的： 格式按 alala_meddg， 数据按  MedDG
def _read3(file_path, is_pk=False):
    with open('data/160_last_topic2num.pk','rb') as f:
        topic2num = pickle.load(f)

    D = []

    with open(file_path, 'rb') as f:
        if is_pk:
            dataset = pickle.load(f)
        else:
            dataset = json.load(f)
        for dialog in tqdm(dataset):
            new_dialog = []
            history = []
            now_topic = []
            his_topic = []
            for sen in dialog:
                aa = sen['Symptom']+sen['Attribute']+sen['Test']+sen['Disease']+sen['Medicine']
                if len(aa) > 0:
                    if len(history) > 0 and sen['id'] == 'Doctor':
                        new_dialog.append({"history": copy.deepcopy(history), "next_sym": copy.deepcopy(aa), 'now_topic': copy.deepcopy(now_topic)})
                    now_topic.extend(aa)
                    his_topic.extend(aa)
                history.append(sen['Sentence'])
            for dic in new_dialog:
                future = copy.deepcopy(his_topic[len(dic['now_topic']):])
                dic['future'] = future
                content = ''.join(dic['history'])
                append_entities = set()
                for v in dic['now_topic']:
                    append_entities.add(c2c[v])
                label = {
                    "Symptom": set(),
                    "Medicine": set(),
                    "Test": set(),
                    "Attribute": set(),
                    "Disease": set()
                }
                for v in dic['next_sym']:
                    label[c2c[v][1]].add(v)
                D.append({
                    'content' : content,
                    'append_entities' : list(append_entities),
                    'label' : {k:list(v) for k,v in label.items()}, # 去掉 set
                })

    json.dump(
        D,
        open('data/test_data3.json', 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    return len(D)


if __name__ == '__main__':
    #print(_read('../dataset/MedDG/MedDG_train.json'))
    #print(_read2('../dataset/MedDG/MedDG_train.json'))
    #print(_read3('../dataset/MedDG/MedDG_train.json'))
    #print(_read2('data/train_data.pk', True))
    #print(_read3('data/train_data.pk', True))

    print(_read3('data/train_data.pk', True))