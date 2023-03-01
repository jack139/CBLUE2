from tqdm import tqdm
import pickle
import pandas as pd

def load_data():
    f_train=open('./data/train_data.pk','rb')#以二进制的形式读取文件
    train_data=pickle.load(f_train)#读取以上的二进制文件
    f_test=open('./data/dev_data.pk','rb')#以二进制的形式读取文件
    test_data=pickle.load(f_test)#读取以上的二进制文件
    return train_data,test_data

def get_lists(data):
    Symptom_list = []
    Medicine_list = []
    Test_list = []
    Attribute_list = []
    Disease_list = []
    
    for each_data in data:
        for item in each_data:
            Symptom_list += item['Symptom']
            Medicine_list += item['Medicine']
            Test_list += item['Test']
            Attribute_list += item['Attribute']
            Disease_list += item['Disease']
    
    set_lists = dict()
    set_lists['-Symptom'] = list(set(Symptom_list))
    set_lists['-Medicine'] = list(set(Medicine_list))
    set_lists['-Test'] = list(set(Test_list))
    set_lists['-Attribute'] = list(set(Attribute_list))
    set_lists['-Disease'] = list(set(Disease_list))
    
    #额外的excel标签（药物和疾病）
    #df = pd.read_excel('药品-CFDA药品.xlsx',keep_default_na=False)
    #excel_data = df.loc[:,['产品名称','商品名称','通用名称']].values
    #excel_medicine = []
    #for hang in excel_data:
    #    for each in hang:
    #        if(each):
    #            excel_medicine += [each]
    #set_excel_medicine = list(set(excel_medicine))
    #
    #df_2 = pd.read_excel('ICD_10v601.xlsx',keep_default_na=False)
    #excel_data_2 = df_2.loc[:,['霍乱']].values
    #excel_disease = []
    #for hang in excel_data_2:
    #    for each in hang:
    #        if(each):
    #            excel_disease += [each]
    #set_excel_disease = list(set(excel_disease))
    
    #set_lists['-Medicine'] += set_excel_medicine
    #set_lists['-Medicine'] = list(set(set_lists['-Medicine']))
    #set_lists['-Disease'] += set_excel_disease
    #set_lists['-Disease'] = list(set(set_lists['-Disease']))
    
    return set_lists

def dataSort(ele):
    return ele['start']

def returnLength(sentence,index):
    words = ['-Symptom','-Medicine','-Test','-Attribute','-Disease']
    for each in words:
        if(sentence[index].find(each) != -1):
            length = 0
            i = index
            j = index+1
            while(sentence[i].find(each) != -1 and i>-1):
                length += 1
                i -= 1
            while(sentence[j].find(each) != -1 and j<len(sentence)):
                length +=1
                j += 1
            return length
    return 0

def pre_train_data(data,set_lists):
    res = []
    type_words = ['-Symptom','-Medicine','-Test','-Attribute','-Disease']
    for each_data in tqdm(data):
    #     print(index,'/',data_length)
        each_res = []
        for item in each_data:
            dict_list = []
            item_res = []
            sentence = item['Sentence']
            for word in sentence:
                item_res.append(word+' O')
            for type_word in type_words:
                for each in set_lists[type_word]:
                    start = sentence.find(each)
                    end = start + len(each) -1 #包含end位置
                    if(start != -1):
                        flag = False
                        #判断是否已经存在更长的实体
                        for index in range(start,end+1):
                            if returnLength(item_res,index)>len(each):
                                flag = True
                                break
                        if flag:
                            break
                        #不存在更长的实体，将原本存在的实体变成Other
                        for index in range(start,end+1):
                            for each in type_words:
                                if(item_res[index].find(each) != -1):
                                    i = index
                                    j = index+1
                                    while(item_res[i].find(each) != -1 and i>-1):
                                        item_res[i] = sentence[i] + ' O'
                                        i -= 1
                                    while(item_res[j].find(each) != -1 and j<len(item_res)):
                                        item_res[j] = sentence[j] + ' O'
                                        j += 1
                        #分割新实体
                        for index in range(start,end+1):
                            item_res[index] = sentence[index] + (' B' if index==start else ' I') + type_word
                        dict_list.append({'name':each,'start':start,'end':end})
            dict_list.sort(key = dataSort)
            each_res.append(item_res)
        res.append(each_res)
    return res

def pre_test_data(data,set_lists):
    res = []
    type_words = ['-Symptom','-Medicine','-Test','-Attribute','-Disease']
    for each_data in tqdm(data):
        each_res = []
        for item in each_data['history']:
            dict_list = []
            item_res = []
            sentence = item
            for word in sentence:
                item_res.append(word+' O')
            for type_word in type_words:
                for each in set_lists[type_word]:
                    start = sentence.find(each)
                    end = start + len(each) -1 #包含end位置
                    if(start != -1):
                        flag = False
                        #判断是否已经存在更长的实体
                        for index in range(start,end+1):
                            if returnLength(item_res,index)>len(each):
                                flag = True
                                break
                        if flag:
                            break
                        #不存在更长的实体，将原本存在的实体变成Other
                        for index in range(start,end+1):
                            for each in type_words:
                                if(item_res[index].find(each) != -1):
                                    i = index
                                    j = index+1
                                    while(item_res[i].find(each) != -1 and i>-1):
                                        item_res[i] = sentence[i] + ' O'
                                        i -= 1
                                    while(item_res[j].find(each) != -1 and j<len(item_res)):
                                        item_res[j] = sentence[j] + ' O'
                                        j += 1
                        #分割新实体
                        for index in range(start,end+1):
                            item_res[index] = sentence[index] + (' B' if index==start else ' I') + type_word
                        dict_list.append({'name':each,'start':start,'end':end})
            dict_list.sort(key = dataSort)
            each_res.append(item_res)
        res.append(each_res)
    return res


def main():
    train_data,test_data = load_data()
    set_lists = get_lists(train_data)
    res_train_data = pre_train_data(train_data,set_lists)
    res_test_data = pre_train_data(test_data,set_lists)
    
    f_train=open('./data/ner_train','w',encoding='utf-8')#以二进制的形式读取文件
    for each in res_train_data:
        for item in each:
            for sentense in item:
                f_train.write(sentense)
                f_train.write('\r\n')
        f_train.write('\r\n')
    f_train.close()
    
    f_test=open('./data/ner_valid','w',encoding='utf-8')#以二进制的形式读取文件
    for each in res_test_data:
        for item in each:
            for sentense in item:
                f_test.write(sentense)
                f_test.write('\r\n')
        f_test.write('\r\n')
    f_test.close()

if __name__ == '__main__':
    main()