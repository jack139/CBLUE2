#! -*- coding:utf-8 -*-

category = [
    '不标注',
    '阳性', 
    '阴性', 
    '其他',
]

category2 = [i.lower() for i in category]

def category_index(s):
    return category2.index(s.lower())

def category_name(i):
    return category[i]
