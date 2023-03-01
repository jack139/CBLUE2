#! -*- coding:utf-8 -*-

category = [
    '病情诊断', #(diagnosis）
    '病因分析', #(cause)
    '治疗方案', #(method)
    '就医建议', #(advice)
    '指标解读', #(metric_explain)
    '疾病表述', #(disease_express)
    '后果表述', #(result)
    '注意事项', #(attention)
    '功效作用', #(effect)
    '医疗费用', #(price)
    '其他', #(other) 
]

category2 = [i.lower() for i in category]

def category_index(s):
    return category2.index(s.lower())

def category_name(i):
    return category[i]
