import re

i_to_num_dict = {'i':'I', 'ii':'II', 'iii':'III', 'iv':'IV', 'v':'V', 'vi':'VI', 'vii':'VII', 'viii':'VIII'}

def match_itomun(substring):
    abbr = re.search('^v?i+v?', substring.groupdict()['pat'])
    if not abbr:
        abbr = re.search('v?i+v?$', substring.groupdict()['pat'])
    if not abbr:
        return substring.group()
    else:
        abbr = abbr.group()
        matched = re.sub(abbr, i_to_num_dict[abbr], substring.groupdict()['pat'])
        return matched

def i_to_num(string):
    if 'i' in string:
        string = re.sub('(?P<pat>[a-zA-Z]+)', match_itomun, string)
    return string

digit_map = {"Ⅳ":"IV", "Ⅲ":"III", "Ⅱ":"II", "Ⅰ":"I"}
def clean_digit(string):
    # Ⅳ Ⅲ Ⅱ Ⅰ
    # IV III II I
    new_string = ""
    for ch in string:
        if ch.upper() in digit_map:
            new_string = new_string + digit_map[ch.upper()]
        else:
            new_string = new_string + ch
    return new_string

greek_lower = [chr(ch) for ch in range(945, 970) if ch != 962]
greek_upper = [chr(ch) for ch in range(913, 937) if ch != 930]
greek_englist = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda",
                 "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
greek_map = {ch:greek_englist[idx % 24] for idx, ch in enumerate(greek_lower + greek_upper)}
def clean_greek(string):
    new_string = ""
    for ch in string:
        if ch in greek_map:
            new_string = new_string + greek_map[ch]
        else:
            new_string = new_string + ch
    return new_string

prefix_suffix_src = ["部位未特指的", "未特指的", "原因不明的", "意图不确定的", "不可归类在他处", "其他特指的疾患"]
prefix_suffix_tgt = ["部未指", "未指", "不明", "意不", "不归他", "他特指"]
def clean_prefix_suffix(string):
    for idx, replace_str in enumerate(prefix_suffix_src):
        string = string.replace(replace_str, prefix_suffix_tgt[idx])
    return string

other_map = {'＋': '+',
 'pci': '经皮冠状动脉介入治疗',
 'cad': '冠状动脉性心脏病',
 'sle': '系统性红斑狼疮',
 'loa': '左枕前胎位',
 'mp': '支原体',
 'ou': '双眼',
 'mt': '恶性肿瘤',
 'paget': '佩吉特',
 'tpsa': '肿瘤标志物',
 'tc': '血清总胆固醇',
 'pbc': '原发性胆汁型肝硬化',
 'fgr': '胎儿生长受限',
 'barrett': '巴氏',
 'tia': '短暂性脑缺血发作',
 'bowen': '鲍恩',
 'as': '强直性脊柱炎',
 'dic': '弥散性血管内凝血',
 'hcc': '肝细胞癌',
 'ggo': '肺部阴影',
 'cushing': '库欣',
 'ln': '狼疮性肾炎',
 'prl': '泌乳素',
 'copd': '慢性阻塞性肺疾病',
 'mia': '微浸润性腺癌',
 'cea': '癌胚抗原',
 'hpv': '人乳头瘤病毒感染',
 'carcinoma': '恶性上皮肿瘤',
 'iud': '具有子宫内避孕装置',
 'aecopd': '急性加重期慢性阻塞性肺疾病',
 'gvhd': '移植物抗宿主病',
 'crohn': '克罗恩',
 'dixon': '直肠切除术',
 'tsh': '促甲状腺激素',
 'ptca': '冠状动脉腔内血管成形术',
 'ivf': '人工妊娠',
 'rop': '早产儿视网膜病',
 'avnrt': '房室结折返性心动过速',
 'cg': '慢性胃炎',
 'avn': '成人股骨头缺血性坏死',
 'rca': '右冠状动脉',
 'nt': '颈部透明度厚度',
 'nerd': '非糜烂性胃食管反流病',
 'sonk': '自发性膝关节骨坏死',
 'cabg': '冠状动脉搭桥',
 'burrkitt': '伯基特',
 'chd': '冠状动脉粥样硬化性心脏病',
 'hf': '心力衰竭',
 'chdhf': '冠心病心力衰竭',
 'ep': '癫痫',
 'simmond': '西蒙',
 'mgd': '睑板腺功能障碍',
 'fl': '滤泡性淋巴瘤',
 'teson': '特尔松',
 'ra': '类风湿性关节炎',
 'gd': '毒性弥漫性甲状腺肿',
 'poland': '波兰',
 'eb': '疱疹病毒',
 'msi': '微卫星不稳定',
 'pnet': '原始性神经外胚瘤',
 'lutembacher': '卢滕巴赫',
 'acl': '膝关节前交叉韧带',
 'he': '人附睾蛋白',
 'vkh': '伏格特-小柳-原田',
 'le': '红斑狼疮',
 'nyha': '纽约心脏病协会',
 'kt': '克利佩尔-特农纳',
 'rhcc': '复发性肝癌',
 'ige': '免疫球蛋白E',
 'poncet': '篷塞',
 'lst': '大肠侧向发育型肿瘤',
 'cgn': '慢性肾小球肾炎',
 'fsgs': '局灶节段性肾小球硬化',
 'gdm': '妊娠期糖尿病',
 'rsa': '右骶前',
 'htn': '高血压',
 'ncr': '接近完全缓解',
 'hunt': '亨特',
 'ddd': '退变性椎间盘病',
 'alzheimer': '阿尔茨海默',
 'nsclc': '非小细胞肺腺癌',
 'evens': '伊文氏',
 'mikulicz': '米库利奇',
 'ev': '肠病毒',
 'igd': '免疫球蛋白D',
 'chf': '充血性心力衰竭',
 'od': '右眼',
 'ipi': '国际预后指数',
 'dieulafoy': '迪厄拉富瓦',
 'lad': '左前降支',
 'ao': '主动脉',
 'hoffa': '霍法',
 'tunner': '特纳',
 'pagtes': '佩吉特',
 'killip': '基利普',
 'addison': '艾迪生',
 'rett': '雷特',
 'wernicke': '韦尼克',
 'castelman': '卡斯尔曼',
 'goldenhar': '戈尔登哈尔',
 'ufh': '普通肝素',
 'ddh': '发育性髋关节发育不良',
 'stevens': '史蒂文斯',
 'johnson': '约翰逊',
 'athmas': '哮喘',
 'rfa': '射频消融',
 'kippip': '基利普',
 'pancreaticcancer': '胰腺恶性肿瘤',
 'srs': '立体定向放射外科',
 'ama': '抗线粒体抗体',
 'cgd': '慢性肉芽肿病',
 'bmt': '骨髓移植',
 'sd': '脐带血流比值',
 'arnold': '阿诺德',
 'tb': '结核感染',
 'dvt': '下肢深静脉血栓形成',
 'sturge': '斯特奇',
 'weber': '韦伯',
 'smt': '黏膜下肿瘤',
 'ca': '恶性肿瘤',
 'smtca': '粘膜下恶性肿瘤',
 'nse': '神经元特异性烯醇化酶',
 'psvt': '阵发性室上性心动过速',
 'gaucher': '戈谢',
 'fai': '髋关节撞击综合征',
 'lop': '左枕后位',
 'lot': '左枕横位',
 'pcos': '多囊卵巢综合征',
 'sweet': '急性发热性嗜中性皮病',
 'graves': '格雷夫斯',
 'cdh': '先天性髋关节脱位',
 'enneking': '恩内金',
 'leep': '利普',
 'itp': '特发性血小板减少性紫癜',
 'wbc': '白细胞',
 'malt': '粘膜相关淋巴样组织',
 'naoh': '氢氧化钠',
 'fd': '功能性消化不良',
 'ck': '肌酸激酶',
 'hl': '霍奇金淋巴瘤',
 'chb': '慢性乙型肝炎',
 'est': '内镜下十二指肠乳头括约肌切开术',
 'enbd': '内镜下鼻胆管引流术',
 'carolis': '卡罗利斯',
 'lam': '淋巴管肌瘤病',
 'ptcd': '经皮肝穿刺胆道引流术',
 'alk': '间变性淋巴瘤激酶',
 'hunter': '亨特',
 'pof': '卵巢早衰',
 'ems': '子宫内膜异位症',
 'asd': '房间隔缺损',
 'vsd': '室间隔缺损',
 'pda': '动脉导管未闭',
 'stills': '斯蒂尔',
 'ecog': '东部癌症协作组',
 'castlemen': '卡斯尔曼',
 'cgvhd': '慢性移植物抗宿主病',
 'ards': '急性呼吸窘迫综合征',
 'op': '骨质疏松',
 'lsa': '左骶前',
 'afp': '甲胎蛋白',
 'sclc': '小细胞癌',
 'ecg': '心电图',
 'pdl': '细胞程序性死亡配体',
 'mss': '微卫星稳定',
 'masson': '马松',
 'ms': '多发性硬化',
 'tg': '甘油三酯',
 'cmt': '腓骨肌萎缩',
 'ph': '氢离子浓度指数',
 'dlbcl': '弥漫大B细胞淋巴瘤',
 'turner': '特纳',
 'aml': '急性骨髓系白血病',
 'pta': '经皮血管腔内血管成形术',
 'alpers': '阿尔珀斯',
 'tat': '破伤风抗毒素',
 'cavc': '完全性房室间隔缺损',
 'coa': '主动脉缩窄',
 'ggt': '谷氨酰转肽酶',
 'edss': '扩展残疾状态量表',
 'vin': '外阴上皮内瘤变',
 'vini': '外阴上皮内瘤变1',
 'vinii': '外阴上皮内瘤变2',
 'viniii': '外阴上皮内瘤变3',
 'ebv': '疱疹病毒',
 'dcis': '乳腺导管原位癌',
 'gu': '胃溃疡',
 'terson': '特尔松',
 'oa': '骨关节炎',
 'cin': '宫颈上皮内瘤变'
}

def match(substring):
    abbr = re.search('[a-z]+', substring.groupdict()['pat']).group()
    matched = re.sub(abbr, other_map[abbr], substring.groupdict()['pat'])
    return matched

def clean_other(string):
    # oa
    # "＋"="+"
    # aoux not replace ou
    for item in list(other_map.keys()):
        if item == "＋":
            string = re.sub(item, other_map[item], ' '+string+' ')
        else:
            string = re.sub('(?P<pat>[^a-zA-Z]'+item+'[^a-zA-Z])', match, ' '+string+' ')
    return string.strip(' ')

def clean_index(string):
    # 1. 2.
    new_string = ""
    idx = 0
    while idx < len(string):
        ch = string[idx]
        if "1" <= ch <= "9" and idx < len(string) - 1 and string[idx + 1] == ".":
            if ch!= "1":
                new_string += "，"
            else:
                new_string += " "
            idx += 1
        else:
            new_string += ch
        idx += 1
    return new_string

def clean(string):
    string = string.replace("\"", " ").lower()
    string = clean_index(string)
    #string = clean_prefix_suffix(string)
    string = clean_greek(string)
    string = clean_digit(string)
    string = clean_other(string)
    string = i_to_num(string)
    #string = clean_other(string)
    #return string.lower()
    return string

prefix_suffix_src_x = ["恶性","癌", "慢支", "化疗", "皮肤", "胃口", "节育器",
                        "左甲","右甲","腮裂","白内障","小便","停经","积血"]

prefix_suffix_tgt_x = ["恶性肿瘤","癌恶性肿瘤","慢性支气管炎","化学治疗","皮肤和皮下组织", "食欲","避孕环",
                        "左甲状腺","右甲状腺","鳃裂","白内障眼","尿","孕","积血肿"]

def extend_x(string):
    for idx, replace_str in enumerate(prefix_suffix_src_x):
        string = string.replace(replace_str, prefix_suffix_tgt_x[idx])
    return string
