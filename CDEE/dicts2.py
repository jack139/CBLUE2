import json

text2new = {
    '疼痛' : ['痛'],
    '癌'   : ['肿瘤', 'CA', '多发转移'],
    '鳞癌' : ['鳞状细胞癌', '磷状细胞CA'],
    '转移癌' : ['癌转移'],
    '磷状细胞癌' : ['磷状细胞CA'],
    '导管癌' : ['导管内癌'],
    '癌切除术' : ['癌直肠前切除术'],
    '进食' : ['饮食'],
    '睡眠质量' : ['睡眠'],
    '精神状态' : ['精神'],
    '精神可' : ['精神'],
    '食欲' : ['食纳'], # 错别字？
    '咳痰' : ['痰'], 
    '咳嗽' : ['干咳', '偶咳'],
    '大小便' : ['大、小便', '二便'],
    '大便性状' : ['大便', '软便', '便'],
    '大便形状' : ['不成形黑便'],
    '大便颜色' : ['黑色大便'],
    '大便' : ['大小便', '稀水样便'],
    '黑便' : ['黑稀便', '黑大便'],
    '解大便' : ['大便'],
    '大便频率' : ['大便性状及频率'],
    '小便' : ['尿', '小大便', '二便', '大便'],
    '排尿' : ['尿', '小便'],
    '排尿通畅' : ['小便尚通畅'],
    '尿液' : ['尿'],
    '尿线变细' : ['尿线无力、分叉、变细'],
    '尿线分叉' : ['尿线无力、分叉'],
    '间质改变' : ['间质性改变'],
    '听觉下降' : ['听力下降'],
    '嗅觉下降' : ['嗅觉稍下降'],
    '鼻根不适' : ['鼻根胀痛不适'],
    '异形性细胞' : ['异型性细胞'],
    '形态不规则' : ['形状不规则'],
    '非小细胞癌' : ['非小细胞肺癌'],
    '疲劳' : ['乏力'],
    '出血' : ['流血', '触血', '血液流出', '流少量鲜红色血液'],
    '发热' : ['低热', '高热', '热痛'],
    '体重下降' : ['体重未见明显下降', '体重无明显下降', '体重明显下降', '体重近1月下降'],
    '体重变化' : ['体重无明显变化', '体重无异常变化'],
    '体重改变' : ['体重无明显改变', '体重无明显变化', '体重未见明显变化', '体重变化', '体重无明显增减'],
    '体重减轻' : ['体重无明显减轻', '体重减少', '体重明显下降', '体重有减轻', '体重下降', '体重明显减轻'],
    '体重' : ['消瘦'],
    '黄染' : ['身黄', '眼黄'],
    '黑矇' : ['黑朦'], # 错别字？
    '肺大疱' : ['肺大泡'],
    '心脏病' : ['冠心病'],
    '出汗' : ['汗'],
    '症状' : ['[unused1]'],
    '肿' : ['中'], # 错别字？
    '退行性病变' : ['退行性变'], # 漏字
    '尿液颜色' : ['尿黄', '尿色', '小便深黄'],
    '尿量减少' : ['尿量从1500ML左右逐渐减少为300ML左右'],
    '咯血' : ['痰中带血', '咯鲜血'],
    '咳血' : ['咳暗红色血', '咳出10ML鲜血'],
    '呕血' : ['呕鲜红色血液'],
    '瘙痒' : ['痒'],
    '寒颤' : ['寒战'],
    '肾病' : ['肾脏病'],
    '炎性改变' : ['炎症改变'],
    '哽噎感' : ['哽咽感', '梗咽感'],
    '返酸' : ['反酸'],
    '慢阻肺' : ['慢性阻塞性肺疾病'],
    '饱胀感' : ['饱胀'],
    '听觉' : ['听力', '耳聋'],
    '异形细胞' : ['异型性细胞', '异型细胞'],
    '异形细胞团' : ['异型细胞团'],
    '气喘' : ['喘息'],
    '骨密度增强' : ['骨密度明显增强'],
    '疾病进展' : ['疾病已进展'],
    '椎间盘突出' : ['椎间盘变性、突出'],
    '发红' : ['红'],
    '发胀' : ['胀'],
    '体力活动' : ['体力'],
    '软组织块影' : ['软组织块块影'], 
    '肾功能不全' : ['肾功不全'],
    '肾功能衰竭' : ['肾衰竭'],
    '子宫腺肌症' : ['腺肌症'],
    '心力衰竭' : ['心衰'],
    '腹水' : ['腹腔积液'],
    '增大' : ['大'],
    '肿物' : ['息肉'],
    '语言障碍' : ['语言肢体活动障碍'],
    '占位性病变' : ['占位病变', '占位'],
    '包块' : ['包快'], # 错字
    '瘀血' : ['淤血'],
    '血液流出' : ['出血'],
    '支气管炎' : ['慢支炎', '慢支'],
    '腹泻' : ['稀便','泻'], 
    '视力指数' : ['视力数指'], # 错字
    '团块影' : ['团块状影'],
    '听觉减退' : ['听力减退'],
    '骨质吸收' : ['骨质破坏、吸收'],
    '皮肤红肿' : ['皮肤无红肿'],
    '电切除' : ['电切术'],
    '血压波动' : ['血压控制可，波动'],
    '恶性纵裂' : ['恶性肿瘤'], # 错别字？
    '月经周期' : ['月经规则'],
    '经量' : ['量'],
    '步态' : ['跛行'],
    '尿痛' : ['尿末痛', '无痛淡红色血尿'],
    '病变' : ['性变', '病”变', '肺部占位'],
    '体积' : ['大'],
    '紧缩感压榨感' : ['紧缩感、压榨感'],
    '瘀斑' : ['淤斑'],
    '活动耐力下降' : ['活动耐力无明显下降'],
    '咳痰量' : ['痰少'],
    '体力下降' : ['体力逐渐下降'],
    '侧支循环开放' : ['侧枝循环开放'],
    '运动减弱' : ['运动弥漫性减弱'],
    '扩张症' : ['扩张'],
    '颜色' : ['色暗红', '变白变紫又变红'],
    '弯腰活动时明显' : ['弯腰活动及起卧翻身时明显'],
    '神志' : ['神清'],
    '局部按摩后缓解' : ['局部按摩后症状暂时缓解'],
    '梗塞' : ['梗死'],
    '水肿' : ['肿'],
    '呼吸困难' : ['喘累'],
    '红肿热痛' : ['红肿、热痛'],
    '鼻涕' : ['涕'],
    '活动耐量' : ['活动耐力'], # 错字
    '呕吐' : ['吐'],
    '活动受限' : ['运动受限', '活动轻微受限', '活动有轻微受限'],
    '皮肤颜色' : ['身黄'],
    '肝炎' : ['乙肝'],
    '扩张积水' : ['扩张、积水'],
    '语言表达' : ['言语'],
    '脾体积' : ['脾脏增大'],
    '心悸' : ['心慌'],
    '热高' : ['高热'], # 错字
    '黒朦' : ['黑曚'], 
    '黑朦' : ['黑蒙', '黑曚'],
    '黑矇' : ['黑蒙', '黑朦'],
    '肝功能异常' : ['肝功异常', '肝肾功未见异常', '肝肾功能未见明显异常', '肝功轻度异常', '肝功仍轻度异常'],
    '低度鳞状上皮内病变' : ['LSIL'],
    '行走障碍' : ['不能行走'],
    '畏寒' : ['胃寒'], # 错字？
    '打嗝' : ['呃逆'],
    '憋闷' : ['胸闷'],
    '恶心' : ['恶性'], # 错字
    '白昼明显差别' : ['白昼无明显差别'],
    '视物黑矇' : ['无黑蒙'],
    '意识障碍' : ['意识丧失'],
    '血糖正规监测' : ['血糖未正规监测'],
    '尿道感染' : ['尿路感染'],
    '排便' : ['未解大便', '大便频繁'],
    '气促' : ['气短', '气急'],
    '听力下降' : ['听力无明显下降'],
    '渗出' : ['渗液'],
    '流脓' : ['溢脓'],
    '大便' : ['便'],
    '鼻塞' : ['鼻阻'],
    '活动量减少' : ['活动量未见明显减少'],
    '乙型肝炎' : ['乙肝'],
    '营养状态' : ['营养不良'],
    '异常流液' : ['异常阴道流血及流液'],
    '异常流血' : ['异常阴道流血'],
    '偏移' : ['移位'],
    '炎' : ['慢支'],
    '最高体温' : ['最高高体温'],
    '意识恢复' : ['意识逐渐恢复'],
    '流血' : ['流少量鲜红色血液'],
    '结节' : ['姐姐'], # 错字
    '进食量' : ['纳差'], 
    '面色苍白' : ['面色逐渐苍白'],
    '伤口处' : ['伤口换药等处理'],
    '阴道流血' : ['阴道再次出现流血'],
    '神志清晰' : ['神志 清晰'],
    '增厚粗糙' : ['增厚、粗糙'],
    '壁厚' : ['侧壁'],
    '未提及' : ['提示'],
    '肾功能异常' : ['肾功'],
    '贲门粘膜炎' : ['贲门粘膜慢性溃疡性炎'],
    '肺疾病' : ['慢阻肺'],
    '包块增大' : ['包块逐渐增大', '包块未见明显增大'],
    '右乳不适' : ['右乳隐痛不适'],
    '血管充血' : ['血管扩张充血'],
    'FISH：PML/RARA（双色双融合）(15/17)异常' : ['FISH：PML/RARA（双色双融合）(15/17)：未见异常'],
    '排尿踟蹰' : ['排尿滴沥、踟蹰'],
    '阑尾炎' : ['炎性阑尾'],
    '回声异常' : ['低回声低回声区'],
    '胆囊炎' : ['胆囊结石伴炎变'],
    '累及' : ['受累'],
    '改变' : ['性变'],
    '流涕' : ['流清涕'],
    '头不适' : ['头昏不适'],
    '起床困难' : ['起床、睡下困难'],
    '翻身困难' : ['翻身、起床、睡下困难'],
    '全身不适' : ['全身乏力乏力不适'],
    '淋巴结大' : ['大淋巴结'],
    '行走不稳' : ['行走站立不稳'],
    '肿瘤标志物CEA' : ['CEA'],
    '信号异常' : ['异常信号'],
    '支架脱出' : ['支架自行脱出'],
    '大便频率改变' : ['大便性状及频率改变'],
    '大便性状改变' : ['大便性状及频率改变'],
    'HPV16型' : ['HPV：16型'],
    '巧克力囊肿' : ['巧囊'],
    '胸骨后不适' : ['胸骨后疼痛疼痛不适'],
    '肿大' : ['增大'],
    '血肌酐上升趋势' : ['血肌酐呈上升趋势'],
    '体力改变' : ['体力、体重无明显变化'],
    '视物倾斜' : ['视物双影，伴轻度倾斜'],
    '排便困难' : ['排大便困难', '排便稍困难', '排便、排尿困难'],
    '尿量异常' : ['尿量、尿液颜色异常'],
    '精神障碍' : ['精神意识障碍'],
    '软组织增多' : ['增多软组织'],
    '吞咽困难' : ['吞咽梗阻、困难'],
    '面色潮红' : ['面色苍白、潮红'],
    '小便异常' : ['小便未见明显异常'],
    '手足不适' : ['手足麻木麻木不适'],
    '腹部不适' : ['腹部阵发隐痛不适'],
    '脾大' : ['脾稍大'],
    '血常规异常' : ['血常规、肝肾功未见异常'],
    '肾功异常' : ['查C'],
    '肝功异常' : ['查C', '肝功能异常'],
    '骨融合' : ['融合骨'],
    '尿频' : ['小便次数增加'],
    '上腹部不适' : ['上腹部隐痛不适'],
    '排便费力' : ['排便稍费力'],
    '体重减少' : ['体重较少'], # 错字？
    '肿块增大' : ['肿块逐渐增大'],
    '血红蛋白降低' : ['血红蛋白及血小板降低'],
    '欠光滑' : ['前光滑'], # 错字？
    '胃肠道反应' : ['胃肠道反'], # 缺字
    '右纵隔炎' : ['右纵膈慢性肉芽肿性炎'],
    '右侧胸壁炎' : ['右侧胸壁、右纵膈慢性肉芽肿性炎'],
    '右肺炎' : ['右肺、右侧胸壁、右纵膈慢性肉芽肿性炎'],
    '包块缩小' : ['包块稍有缩小'],
    '宫颈HPV16型' : ['宫颈HPV仍提示16型'],
    '情况稳定' : ['情况相对稳定'],
    '十二指肠球隆起' : ['十二指肠球部隆起'],
    '淋巴细胞异常' : ['异常淋巴细胞'],
    '切口不适' : ['切口疼痛不适'],
    '大小便异常' : ['大小便无明显异常', '大小便未见明显异常'],
    '肺不张' : ['右肺下叶实变不张'],
    '脾增大' : ['脾脏增大'],
    '白细胞升高' : ['白细胞长长期处于升高状态', '白细胞逐渐升高'],
    '肺间质炎' : ['肺间质慢性炎'],
    '骨显影异常' : ['骨显影未见明显异常'],
    '脾脏缩小' : ['脾脏逐渐缩小'],
    '视物变形' : ['视物遮挡、变形'],
    '视力下降' : ['视力逐渐下降', '视力明显下降'],
    '视物模糊' : ['视近物模糊'],
    '全身肌肉不适' : ['全身肌肉酸痛不适'],
    '淋巴细胞及单核细胞增高' : ['淋巴细胞及单核细胞比例略增高'],
    '胃肠反应' : ['胃肠道反应'],
    '胸背部不适' : ['胸背部背部隐痛不适'],
    '甲状腺功能抗体异常' : ['甲功抗体未见异常'],
    '甲状腺功能异常' : ['甲功及甲功抗体未见异常'],
    '功能失调性子宫出血' : ['功血'],
}   



value2new ={
    '回盲壁' : ['回盲部壁'], 
    '右肺门淋巴结' : ['右肺门'],
    '纵隔淋巴结' : ['纵隔多发淋巴结', '纵隔', '纵膈' ],
    '宫颈管' : ['颈管'],
    '黑矇' : ['黑朦'],
    '发射' : ['放射'],
    '放射性' : ['放射'],
    '发射性' : ['放射'],
    '原发性，极高危' : ['原发性高血压3级高血压3级极高危极'],
    '带血' : ['血'],
    '好' : ['可'],
    '肝结节' : ['肝内多发稍低密度小结节'],
    '右髂总淋巴结' : ['右髂总'],
    '左髂总淋巴结' : ['左髂总'],
    '右盆腔淋巴结' : ['右盆腔）淋巴结'],
    '左盆腔淋巴结' : ['左盆腔'],
    '子宫内膜样，侵及深肌层（近浆膜层），宫颈内口及双侧宫旁未见癌累及' : ['子宫内膜样腺癌（3级），侵及深肌层（近浆膜层），宫颈内口及双侧宫旁未见癌累及'],
    '子宫腔后壁底部' : ['子宫腔后壁及底部'],
    '右下叶支气管' : ['右下叶下叶支气管'],
    '胰头上方淋巴结' : ['<胰头上方>淋巴结'],
    '劳累性' : ['劳力性'],
    '正常' : ['如常'],
    '纵膈淋巴结' : ['纵膈多处淋巴结', '纵膈多发淋巴结', '纵膈'],
    '食道中段下段' : ['食道中下段'],
    '溃疡型' : ['溃疡性'],
    '全身' : ['身', '感冒受凉', '[unused1]'],
    '全身性' : ['全身', '四肢', '身'],
    '量少' : ['少'],
    '减轻' : ['减少', '下降', '消瘦'],
    '增加' : ['升高', '增高'],
    '变重' : ['加重'],
    '升高' : ['发热', '大于', '均高于'],
    '非霍奇金氏' : ['非霍奇金'],
    '骨髓检查可见' : ['骨髓检查：可见'],
    '服用中药后稍好转' : ['服用中药后疼痛稍好转'],
    '加重' : ['右肺腺癌伴纵膈、锁骨上淋巴结转移、胸膜、心包、颅内转移', '增加'],
    '偶尔' : ['偶'],
    '胸椎骶髂关节' : ['骶髂关节'],
    '肠周脂肪淋巴结' : ['肠周脂肪中淋巴结'],
    '右卵巢' : ['（右）卵巢'],
    '左卵巢' : ['（左）卵巢'],
    '吻合口淋巴结' : ['吻合口'],
    '双侧下颌下淋巴结' : ['双侧下颌下下颌下淋巴结'],
    '囊状' : ['囊性'],
    '减弱' : ['低', '弱'],
    '叶片团' : ['叶片状'],
    '肠系膜根淋巴结' : ['<肠系膜根部>淋巴结', '肠系膜根部淋巴结'],
    '肠系膜根部淋巴结' : ['<肠系膜根部>淋巴结'],
    '右侧盆壁淋巴结' : ['右侧盆壁'],
    '肠周淋巴结' : ['肠周脂肪组织中淋巴结', '（肠周）淋巴结'],
    '直肠双侧切缘' : ['双侧切缘'],
    '肩部' : ['肩背部', '肩、背部'],
    '减少' : ['纳差'],
    '胸后' : ['胸'],
    '眼' : ['视物'],
    '夜间' : ['夜尿'],
    '次数增多' : ['增多', '尿频', '次数多', '次数增加'],
    '次数增' : ['尿频'],
    '腹部' : ['腹'],
    '增厚' : ['肥厚'],
    '连续发作' : ['连续2年以上发作'],
    '冠状动脉' : ['冠心病', '冠脉'],
    '硬化性，粥样' : ['冠心病'],
    '硬化性' : ['冠心病'],
    '粥样' : ['冠心病'],
    '增多' : ['大汗', '较多', '异常信号', '量多', '多汗'],
    '偶有，尿线' : ['偶有尿线'],
    '浸润性，非特殊类型' : ['浸润性癌，非特殊类型'],
    '骨组织' : ['左侧髂骨', '骨'],
    '骨组织转移' : ['肿瘤骨转移'],
    '双侧大腿后' : ['双侧大腿前侧'],
    '间断性，疼痛尚可忍受' : ['间断性痛，疼痛尚可忍受'],
    '左侧第1跖趾关节' : ['左侧第一跖趾关节'],
    '左乳左右上后切缘' : ['左乳左、右、上、下、后切缘'],
    '不良' : ['差'],
    '腹膜后左侧' : ['腹膜后偏左侧'],
    '子宫内' : ['宫内'],
    '宫颈右侧穹隆' : ['右侧穹隆'],
    '宫颈前穹隆' : ['前穹隆'],
    '接触性' : ['触血'],
    '明显，反复' : ['胸闷仍反复发作，现患者胸闷仍明显'],
    '休息后好转' : ['休息后胸闷好转', '休息后眩晕好转'],
    '无诱因' : ['无明显诱因'],
    '多发性' : ['多发'],
    '活动后胀痛，平卧位可明显减轻' : ['活动后伴右侧腹股沟区胀痛，取平卧位可明显减轻'],
    '活动后胀痛' : ['活动后伴右侧腹股沟区胀痛'],
    '一大拇指大小，质地较软，移动可，可回纳' : ['一大拇指大小包块，质地较软，移动可，可回纳'],
    '粘性' : ['粘痰', '黏痰'],
    '冠状动脉LAD开口' : ['LAD开口'],
    '含服硝酸甘油缓解' : ['含服硝酸甘油后症状可立即缓解'],
    '间隔时间缩短' : ['间隔时间减短'],
    '输液治疗缓解' : ['输液治疗（具体用药及诊断不详）后腰痛缓解'],
    '黄色' : ['黄'],
    '缓解，明显' : ['明显缓解'],
    '明显缓解' : ['缓解不明显'],
    '头部' : ['头痛'],
    '阵发性，胀痛，夜间采取侧卧位时疼痛明显，休息后自行缓解' : ['阵发性胀痛，于夜间采取侧卧位时疼痛明显，休息后自行缓解'],
    '无明显下降' : ['未见明显下降'],
    '左侧腹股沟淋巴结' : ['左侧腹股沟可见多发淋巴结'],
    '盆腔淋巴结' : ['盆腔'],
    '腹膜后淋巴结' : [ '腹膜后多发淋巴结', '腹膜后'],
    '干性' : ['干咳'],
    '有进展' : ['进展'],
    '弥漫性' : ['弥漫'],
    '食管' : ['食道'],
    '欠佳' : ['不佳'],
    '鲜红色' : ['带血'],
    '红色' : ['血'],
    '变软，缩小' : ['变软、缩小'],
    '输液后' : ['输液3天后'],
    '腰椎L2L4椎体前缘' : ['平腰2-腰44椎体前缘'],
    '双侧腹股沟淋巴结' : ['双侧腹股沟腹股沟淋巴结'],
    '肺鳞癌' : ['（右上叶后段）鳞癌'],
    '肺右上叶后段' : ['右上叶后段'],
    '右侧面部淋巴组织' : ['（右侧面部）送检少许组织示淋巴组织'],
    '进食后，平卧位，身体前驱时出现，影响夜间休息，持续时间数小时不等' : ['进食后及平卧位、身体前驱时出现，影响夜间休息，持续时间数小时不等'],
    '逐渐，增大' : ['逐渐开始增大'],
    '食管旁淋巴结' : ['（食管旁）淋巴结', '食管旁'],
    '黄色，软' : ['黄色软'],
    '两肺门淋巴结' : ['两肺门'],
    '两肺上叶胸膜' : ['两肺上叶上叶胸膜'],
    '右肺上叶尖段水平裂' : ['右肺上叶尖段陈旧结核'],
    '急性发作，慢性，阻塞性' : ['慢性阻塞性肺疾病急性发作'],
    '阻塞性' : ['慢阻肺'],
    '锁骨下淋巴结' : ['锁骨下多发淋巴结'],
    '右肺门周围' : ['右肺门占位伴周围'],
    '明显，端坐位后可缓解' : ['夜间明显，自诉端坐位后可缓解'],
    '红系旺盛' : ['红系增生旺盛'],
    '减退' : ['下降'],
    '发黄' : ['尿黄'],
    '右肺中' : ['右肺'],
    '右腕部' : ['右侧腕部'],
    '腹主动脉旁旁淋巴结' : ['腹主动脉旁'],
    '门静脉旁淋巴结' : ['门静脉旁'],
    '腹腔干旁淋巴结' : ['腹腔干旁>淋巴结'],
    '非典型' : ['不典型'],
    '1-6肋骨' : ['1-6肋'],
    '反复，进行性加重，排气后可暂时缓解' : ['反复右下腹隐痛，无进行性加重，排气后可暂时缓解'],
    '双肺中叶' : ['双肺肺中叶'],
    '胃部' : ['胃'],
    '胸壁皮肤' : ['胸壁及左侧大腿内侧等处皮肤'],
    '直肠癌术后，转移性' : ['直肠癌术后伴肝转移，小肠转移转移性'],
    '量多，糊状，粘液' : ['量多，呈糊状，偶有粘液'],
    '前列腺中央及外周' : ['前列腺增大，中央带及外周带'],
    '右侧8-10后肋' : ['右侧第8-10后肋'],
    '膀胱腔内' : ['膀胱腔腔内'],
    '交界处浆膜面' : ['浆膜面'],
    '交界处管腔' : ['管腔'],
    '右侧左侧阴道前壁前壁' : ['左侧阴道前壁、右侧阴道前壁'],
    '双侧髂血管旁' : ['双侧髂血管血管旁'],
    '左侧肺门淋巴结' : ['左侧肺门肺门淋巴结'],
    '夜间增多' : ['夜间较多'],
    '坐位，改变体位' : ['坐位、改变体位'],
    '骨头' : ['骨'],
    '鲜，无血凝块、食物残渣。' : ['鲜血，无血凝块、食物残渣'],
    '受凉、天气变化大时即发作，性质同前，冬春季节多发，迁延反复' : ['受凉、天气变化大时即发作，每次性质同前，冬春季节多发，每年累积发作时间均超过3个月，期间病情迁延反复'],
    '左乳局部' : ['左乳'],
    '肝胃间隙淋巴结' : ['肝胃间隙多发肿大淋巴结'],
    '腰后' : ['腰'], 
    '实性，占位性' : ['实性伴占位性'],
    '右侧胸后' : ['右侧胸'], 
    '带血，为暗红色，不与大便混合' : ['带血，为暗红色不与大便混合'],
    '明显活跃，形态异常，粒系，未见有核红细胞' : ['粒系增生明显活跃伴形态异常，未见有核红细胞'],
    '口腔粘膜' : ['口腔黏膜'],
    '次数增多，明显' : ['尿频'],
    '脓性' : ['脓'],
    '周边探及' : ['周边未探及'],
    '左肾肾盏' : ['左肾肾盂肾盏'],
    '环状' : ['环形'],
    '右肺上叶斜裂' : ['右肺上叶及斜裂'],
    '轻微，胀痛' : ['轻微胀痛'],
    '无明显变化' : ['无明显改变'],
    '明显' : ['明确'],
    '淋巴转移' : ['淋巴结转移'],
    '右肱骨上段局部' : ['右肱骨上段'],
    '爬坡、爬楼、快走、劳动等中等量体力活动后加重，休息后缓解' : ['爬坡、爬楼、快走、劳动等中等量体力活动后出现上述不适加重，每次持续约5-10分钟，休息后可缓解'],
    '活动后加重' : ['活动后出现上述不适加重', '活动后明显加重', '活动后咳嗽加重', '活动后尤甚', '患者活动后（平路行走）心累、气促加重'],
    '缓解，反复' : ['缓解，但疼痛反复'],
    '缓解' : ['疗效评价PR', '明显好转'],
    '左侧下磨牙' : ['左侧一颗下磨牙'],
    '双肺门淋巴结' : ['双肺门多发淋巴结', '双肺门'],
    '胸后后' : ['胸'],
    '胀痛，明显' : ['胀痛，疼痛明显'],
    '休息后缓解，长时间说话加重' : ['长时间说话加重，休息后声嘶可稍缓解'],
    '腰部' : ['腰臀部', '腰背部', '腰腹部'],
    '腰背部' : ['腰'],
    '久行弯腰可诱发加重' : ['久行、弯腰可诱发疼痛加重'],
    '急性，普通' : ['急性普通'],
    '周围肝组织内' : ['周围肝组织部分区域'],
    '右肺旁下下叶背段' : ['右肺下叶背段'],
    '右下肺门' : ['右下肺肺门'],
    '纵膈内淋巴结' : ['纵膈内见稍大淋巴结', '纵膈内肿大淋巴结'],
    '左侧顶叶' : ['左侧额、顶叶'],
    '左侧额叶' : ['左侧额'],
    '右上叶支气管周围粘膜' : ['右上叶支气管被新生物完全阻塞，新生物表现血管迂曲、丰富，周围粘膜'],
    '累及右侧亚隆突、中间支气管' : ['累计右侧亚隆突、中间支气管'],
    '血管丰富' : ['血管迂曲、丰富'],
    '上述' : ['上诉'], # 错字
    '夜间明显' : ['夜间头昏症状最明显'],
    '1-2小时后加重' : ['1-2小时后头昏症状逐渐加重'],
    '早晨午睡后减轻' : ['早晨及午睡后头昏较轻'],
    '障碍' : ['心悸', '不能'],
    '肝门淋巴结' : ['肝门部淋巴结', '肝门区淋巴结'],
    '直肠壁深肌层' : ['肠壁深肌层'],
    '腰椎L2、3椎体' : ['腰2、3椎体'],
    '腰椎' : ['腰33/4椎'],
    '颈椎' : ['颈33/4-6/7椎'],
    '腰椎L4-5水平椎管内上方椎管内' : ['腰4-5水平椎管内占位性病变，其上方椎管内'],
    '腰椎L4-5水平椎管内' : ['腰4-5水平椎管内'],
    '身体' : ['身'],
    '肝后' : ['肝'],
    '左上腹部' : ['左上腹上腹部'],
    '胸部' : ['胸'],
    '癌结节，微小' : ['微小癌癌结节'],
    '无隐血' : ['无大便隐血'],
    '右手食指' : ['右手拇指及食指'],
    '胸椎' : ['胸腰椎'],
    '右侧第4肋骨' : ['右侧第4及左侧第9、10肋骨'],
    '全身，明显缓解' : ['全身乏力症状未见明显缓解'],
    '髓样' : ['髓系'],
    '夜间，阵发性' : ['夜间阵发性'],
    '劳力性，以上楼、上坡或剧烈活动后明显' : ['劳力性呼吸困难，以上楼、上坡或剧烈活动后明显'],
    '粘膜' : ['黏膜'],
    '非咖啡色内容物' : ['非咖啡色胃内容物'],
    '右侧锁骨淋巴结' : ['右侧锁骨锁骨区淋巴结'],
    '左肾门淋巴结' : ['<左肾门>淋巴结'],
    '膀胱基底部' : ['膀胱肿瘤及基底部'],
    '乳头状，尿路上皮' : ['乳头状尿路上皮'],
    '高级别，尿路上皮' : ['高级别尿路上皮'],
    '发红' : ['红'],
    '粘稠液' : ['粘稠痰'],
    '粘液' : ['粘稠痰', '黏液'],
    '轻度，倾斜' : ['轻度倾斜'],
    '有重影' : ['视物双影'],
    '萎靡' : ['萎'],
    '脓，粘液' : ['粘液脓血'],
    '肺门淋巴结' : ['右肺门结节影，淋巴结', '肺门'],
    '持续性隐痛程' : ['持续性隐痛'],
    '右侧背部' : ['右侧肩背部'],
    '腹腔后淋巴结' : ['腹腔后多发淋巴结', '腹腔后增多软组织影，肿大淋巴结'],
    '腹腔淋巴结' : ['腹腔及腹腔后多发淋巴结', '腹腔及腹膜后多发淋巴结'],
    '强化，小' : ['强化小'],
    '肝细胞，包膜见癌组织侵犯，切缘未见癌组织累及' : ['分化细胞肝癌，包膜见癌组织侵犯；切缘未见癌组织累及'],
    '肢体后' : ['肢体'],
    '子宫肌层' : ['肌层'],
    '不均' : ['欠均质'],
    '胃内容物，胆汁' : ['胃内容物或胆汁'],
    '持续性，不剧烈，可忍受' : ['持续性右上腹痛，程度不剧烈，可忍受'],
    '指关节' : ['指间关节'],
    '心脏' : ['心'],
    '双侧掌关节' : ['双侧掌指、近端指间关节'],
    '下颌下' : ['[unused1]'],
    '左颌' : ['[unused1]'],
    '肾上腺后' : ['肾上腺'],
    '逐渐增大，无局部红，无局部肿，无局部热，无局部痒' : ['逐渐增大 ， 无局部红、肿、热、痒'],
    '右卵巢后' : ['右卵巢'], 
    '右半结肠两切端' : ['右半结肠圆形细胞恶性肿瘤，两切端'],
    '右肺中叶下叶' : ['右肺中叶及下叶'],
    '皮肤无破溃' : ['皮肤无红肿、破溃'],
    '颈部淋巴结' : ['颈部、腋窝淋巴结'],
    '性交时' : ['同房'],
    '质硬，皮肤受累' : ['质硬肿块，皮肤受累'],
    '右枕叶室管膜' : ['（右枕叶）室管膜'],
    '转移性' : ['转移可能大'],
    '胸椎T3椎体' : ['T3椎体'],
    '右肩部皮下' : ['右肩部肩部皮下'],
    '右肩部皮下脂肪层' : ['右肩部肩部皮下皮下脂肪层'],
    '囊实性' : ['囊实混合性'],
    '子宫旁' : ['右侧宫旁'],
    '肝左叶' : ['肝右后叶占位及左叶'],
    '不规则，低' : ['不规则低'],
    '进行性，增大' : ['进行性增大'],
    '次数减少' : ['次数及尿量均减少'],
    '夜间及活动后加重' : ['夜间及活动后尤甚'],
    '面色，突发性' : ['突发性面色'],
    '未分化型' : ['未分化'],
    '平静休息后缓解' : ['平静休息后可缓解'],
    '无皮肤肿' : ['皮肤无红、肿'],
    '无皮肤红' : ['皮肤无红'],
    '低分化' : ['低分'],
    '质硬，无压痛' : ['质硬,无压痛'],
    '十二指肠球粘膜下方' : ['（十二指肠球部）送检粘膜下方'],
    '左侧输尿管口附近' : ['左输尿管口附近'],
    '非浸润性，尿路上皮' : ['非浸润性尿路上皮'],
    '膀胱左侧顶壁' : ['膀胱左侧顶顶壁'],
    '左侧输尿管口' : ['左侧输尿管输尿管口'],
    '瘢痕组织旁性' : ['瘢痕组织旁'],
    '活动后，较前加重，少于之前活动量即可发作，症状持续时间较前延长' : ['活动后胸痛、心悸、胸闷较前加重，表现为少于之前活动量即可发作，症状持续时间较前延长'],
    '右腋窝淋巴结' : ['<右腋窝>淋巴结', '（右腋窝蓝染）淋巴结'],
    '黄色，粘稠' : ['黄白色粘稠'],
    '控制不详' : ['控制情况不详'],
    '干结，颗粒状' : ['干结，6-7天/次，颗粒大便'],
    '间断，暗红色' : ['间断可咳暗红色'],
    '左肺下叶外基底段' : ['下叶外基底段'],
    '右前2肋' : ['右前第2肋'],
    '大量，有活动性出血' : ['有活动性出血'],
    '大量，呈鲜红色，一天可浸湿三张卫生巾' : ['大量流血，呈鲜红色，一天可浸湿三张卫生巾'],
    '直肠周围淋巴结' : ['直肠新生物伴周围淋巴结', '直肠周围多发淋巴结'],
    '持续性，隐痛' : ['持续性隐痛'],
    '入侵' : ['侵及'],
    '阴道壁组织上皮' : ['（阴道壁组织）鳞状上皮'],
    '间歇发作' : ['间断发作'],
    '无肩背部反射痛' : ['无肩背部放射痛'],
    '低密度弱' : ['低密度'],
    '支气管' : ['支'],
    '慢性' : ['慢'],
    '多，分散' : ['多且较分散'],
    '奈达铂化疗药红' : ['奈达铂化疗药引起等过敏反应'],
    '左心脏' : ['左心'],
    '慢性，肾衰竭期' : ['肾衰竭期'],
    '肝右叶膈顶区' : ['肝右叶膈顶顶区'],
    '切端，累及' : ['切端未见癌累及'],
    '中分化，含粘液腺癌成份，浸及肠壁全层' : ['中分化腺癌（含粘液腺癌成份）浸及肠壁全层'],
    '约苹果大小，进行性增大' : ['约苹果大小，无进行性增大'],
    '非霍奇金型' : ['非霍奇金'],
    '发黑' : ['黑色', '黑'],
    '腹壁淋巴结' : ['腹壁>送检淋巴结'],
    '右侧髂血管旁淋巴结' : ['右侧髂血管血管旁多发淋巴结', '右侧髂血管血管旁', ],
    '左侧髂血管旁淋巴结' : ['左侧髂血管血管旁'],
    '直肠远切' : ['远切'],
    '直肠近切' : ['近切'],
    '右眼球后' : ['右眼急性球后'],
    '盆腔周围' : ['周围'],
    '盆腔左侧' : ['盆腔偏左侧'],
    '化放疗后' : ['放化疗后'],
    '明显增多' : ['大汗'],
    '结肠管腔' : ['所累及处管腔'],
    '冠状动脉硬化性' : ['冠心病'],
    '双乳房' : ['双乳'],
    '乳头皮肤' : ['乳头及皮肤', '皮肤'],
    '肝脏' : ['肝'],
    '膀胱病损，硬膜外，麻醉下' : ['硬膜外麻醉下行膀胱病损'],
    '面部' : ['面色'],
    '隆突下淋巴结' : ['隆突下）淋巴结'],
    '胃左淋巴结' : ['胃左'],
    '胃周围淋巴结' : ['胃周围'],
    '片状' : ['大片'],
    '右下腹部' : ['右下腹'],
    '卧床休息后可缓解' : ['卧床休息可稍缓解'],
    '弯腰、负重及久坐后加重' : ['久站、久坐及长距离行走后加重'],
    '包绕，跨叶生长' : ['包绕下叶支气管跨叶生长'],
    '左胸双侧胸壁皮下' : ['双侧胸壁皮下'],
    '左胸纵隔皮下' : ['纵隔'],
    '左胸皮下' : ['左胸'],
    '回肠近端' : ['近端'],
    '不完全性' : ['不全性'],
    '回肠末端' : ['回肠末段'],
    '全身多淋巴结转移' : ['全身多淋巴结、骨转移'],
    '肺内转移' : ['肺内、全身多淋巴结、骨转移'],
    '炎症细胞，少量，慢性' : ['少量慢性炎症细胞'],
    '衬覆少量尿路上皮，炎症细胞浸润' : ['衬覆少量尿路上皮伴个别炎症细胞浸润'],
    '右乳4点7点' : ['右乳4点、7点'],
    '与前片比较变化不大' : ['于前片（2014.12.29）比较变化不大'],
    '夜间，次数增多' : ['夜尿次数增多'],
    '子宫颈' : ['宫颈'],
    '酸痛' : ['酸疼'],
    '持续性，胀痛，疼痛可忍受，休息后可稍缓解，无放射痛' : ['持续性胀痛，疼痛可忍受，休息后可稍缓解，无放射痛'],
    '受凉后，白色，粘' : ['白色粘'],
    '减轻明显' : ['减轻'],
    '变换体位时明显' : ['变换体位时疼痛明显'],
    '周围淋巴结' : ['周围多发淋巴结'],
    '多发，转移' : ['多发淋巴结肿大淋巴结肿大转移'],
    '邻近上下颌骨' : ['邻近上、下颌骨'],
    '桃大小，表面有溃烂面，疼痛明显，质地硬，活动度差' : ['核桃大小的肿物，表面有溃烂面，疼痛明显，质地硬，活动度差'],
    '量增多' : ['量逐渐增多'],
    '肝淋巴结' : ['肝十二指肠肠淋巴结'],
    '右上肺' : ['右肺', '（右上）肺'],
    '改善明显' : ['改善不明显'],
    '颈椎间盘' : ['颈腰椎间盘'],
    '块状，软组织，分叶状' : ['块状软组织密度影，呈分叶状'],
    '和体位活动无关' : ['和体位、活动无关'],
    '左肺' : ['左 肺'],
    '腰2椎间盘' : ['腰2-骶1椎间盘'],
    '腹膜内淋巴结' : ['腹腔内多发肿大淋巴结'],
    '左乳11点钟腺体内' : ['左乳11点钟距乳头约3CM处腺体层内'],
    '管腔表面' : ['管腔全周，表面'],
    '稍微活动后' : ['稍活动'],
    '以左侧为甚，夜间未影响睡眠' : ['以左侧为甚，为持续胀痛，夜间未影响睡眠'], 
    '左回旋支近段管腔' : ['左回旋支近段管壁可见非钙化斑块，管腔'],
    '稀便' : ['稀大便'],
    '少量' : ['少许'],
    '白色' : ['色白', '白'],
    '牙龈黏膜' : ['牙龈和口腔黏膜'],
    '大肠' : ['肠'],
    '小肠' : ['肠'],
    '明显，化疗期间，化疗后' : ['化疗期间及化疗后无明显'],
    '右肺中叶及左肺下叶' : ['右肺中叶及左肺下'],
    '剧烈时出汗' : ['剧烈时汗出'],
    '活动时' : ['稍微活动'],
    '肠壁切缘' : ['肠壁全层，两切缘'],
    '近左下肺门' : ['近左下肺肺门处'],
    '反复间断' : ['反复出现间断'],
    '双侧腰部' : ['双侧腰腰部', '双侧腰'],
    '欠光滑' : ['前光滑'], # 错字
    '双肺后' : ['双肺'],
    '未见新生物' : ['未见充血水肿及新生物'],
    '双侧肺门角淋巴结' : ['双侧肺门见多发肿大淋巴结'],
    '后心膈角淋巴结' : ['后心膈角'],
    '主肺动脉窗淋巴结' : ['主肺动脉窗'],
    '气管前间隙淋巴结' : ['气管前间隙'],
    '右胸腔' : ['右侧少量胸腔'],
    '粘' : ['黏'],
    '右肺上中下叶' : ['右肺上、中、下叶'],
    '隐痛持续性加重' : ['隐痛，持续性加重'],
    '腰椎L5双侧' : ['腰5双侧椎'],
    '右上腹腹膜' : ['右上腹上腹部腹膜'],
    '边界清晰' : ['边界较清晰'],
    '双侧输尿管中下段' : ['双侧输尿管中上段扩张，中下段'],
    '原发性，很高危' : ['原发性高血压3级很高危'],
    '直肠管腔' : ['直肠包块伴管腔'],
    '宫内膜腺体' : ['局灶腺体'],
    '不典型性' : ['不典型'],
    '阵发性，喜屈曲卧位，肚脐周围疼痛为主' : ['肚脐周围疼痛为主，为阵发性绞痛，喜屈曲卧位'],
    '与活动饮食无关' : ['饮食无关'],
    '双侧输卵管' : ['双侧慢性输卵管'],
    '右腋窝深组淋巴结' : ['右腋窝深组无蓝染）淋巴结'],
    '食管壁淋巴结' : ['食管壁壁淋巴结'],
    '肝右后叶下段' : ['肝右后叶右后叶下段'],
    '肝左外叶上段' : ['肝左外叶左外叶上段'],
    '右侧腹股沟区淋巴结' : ['右侧腹股沟区可见肿大淋巴结'],
    '右侧大腿' : ['右大腿'],
    '术后恢复可' : ['术后患者恢复可'],
    '休息后' : ['休息2天后'],
    '脑后' : ['脑'],
    '硬化性，粥样，心绞痛型' : ['冠心病（心绞痛型）'],
    '逐渐增大，久站及剧烈咳嗽时突出较明显，间歇性胀痛，可回纳' : ['逐渐增大，久站及剧烈咳嗽时突出较明显，仍伴有间歇性胀痛，可回纳'],
    '横结肠远端切缘' : ['横结肠近端及远端切缘'],
    '横结肠近端切缘' : ['横结肠近端及远端切缘'],
    '前列腺后' : ['前列腺'],
    '胃小弯' : ['小弯侧'],
    '胃贲门区' : ['胃贲门贲门区'],
    '黑色' : ['黑'],
    '鸡鸣样，昼夜均较频繁' : ['鸡鸣样咳嗽，昼夜均较频繁'],
    '阵发性，非高调金属声样' : ['阵发性咳嗽，非高调金属声样'],
    '右肺中叶内侧段上叶下舌段' : ['右肺中叶内侧段及左肺上叶下舌段'],
    '左手2、3、4、5指关节' : ['左手第2、3、4、5指关节'],
    '左足趾' : ['左足第一足趾'],
    '排气后缓解' : ['排气后腹胀可缓解'],
    '性质待定，异常' : ['异常回声，性质待定'],
    '双侧腋窝淋巴结' : ['双侧腋窝、颈部淋巴结'],
    '黄豆大小' : ['“黄豆”大小'],
    '季节变化时加重' : ['季节变化时症状加重'],
    '冬春季节加重' : ['冬春季节、季节变化时症状加重'],
    '冷空气刺激后加重' : ['冷空气刺激后、活动后咳嗽加重'],
    '侵犯' : ['受侵'],
    '左侧腰部' : ['左侧腰腿腿部'],
    '久站久坐久行' : ['久站、久坐及久行'],
    '左侧肾盏' : ['左侧肾盂肾盏'],
    '左肾静脉' : ['左肾动静脉'],
    '左肾动脉' : ['左肾动静脉'],
    '腹主动脉动脉' : ['腹主动脉'],
    '左肺上叶下叶背段中叶' : ['左肺上叶、下叶背段及右肺中叶'],
    '肌层2\\/3' : ['肌层2/3'],
    '胸骨中段下段' : ['胸骨中下段'],
    '带血，回吸性' : ['回吸性涕血'],
    '左上肺前段舌段' : ['左上肺前段+舌段'],
    '绞痛，餐后开始出现，无放射痛，程度剧烈，难以忍受，每次约持续5秒后自行缓解' : ['餐后开始出现中上腹疼痛，为绞痛，无放射痛，程度剧烈，难以忍受，每次约持续5秒后自行缓解'],
    '发作，上述' : ['1-2年发作一次'],
    '明显，体位改变时加重' : ['体位改变时症状明显加重'],
    '右侧肾周间隙' : ['右侧肾肾周间隙'],
    '直肠上皮' : ['（直肠）腺上皮'],
    '异型，中-重度，腺上皮' : ['腺上皮中-重度异型'],
    '环状，直肠距肛7~12CM，表面溃疡，肠腔狭窄' : ['直肠距肛7~12CM见一个环状新生物，表面溃疡，肠腔狭窄'],
    '盆腔内淋巴结' : ['盆腔内内淋巴结'],
    '不规则，点滴' : ['不规则点滴'],
    '腰椎L5椎体' : ['腰5椎体'],
    '腰椎L5椎' : ['腰55椎'],
    'L5S1椎间盘' : ['腰5-骶1椎间盘'],
    '骶椎' : ['骶1椎'],
    '自行缓解' : ['自行缓'],
    '肝细胞癌，术后' : ['肝细胞癌切除术后'],
    '解便缓解' : ['解便后可缓解'],
    '肛门排气后缓解' : ['肛门排气及解便后可缓解'],
    '左侧盆腔腹膜后' : ['左侧盆腔盆腔腹膜腹膜后'],
    '幽门下方大网膜淋巴结' : ['幽门下方大网膜，有髂外，左盆腔，右盆腔）淋巴结'],
    '左右卵巢' : ['左、右卵巢'],
    '脾脏' : ['脾'],
    '胃窦平滑肌' : ['胃窦粘膜下肿瘤（来源第四层，平滑肌'],
    '右冠状动脉近段' : ['右冠近段'],
    '回旋支中段' : ['回旋支管腔细小，中段'],
    '左冠状动脉主干' : ['左主干'],
    '右冠状动脉' : ['右冠'],
    '左前哨淋巴结' : ['左前哨前哨淋巴结'],
    '肾盂' : ['肾孟'], # 错字 
    '阵阵，干咳为主' : ['阵阵咳嗽，以干咳为主'],
    '发病过程不能准确描述' : ['发病形式、过程不能准确描述'],
    '发病形式不能准确描述' : ['发病形式、过程不能准确描述'],
    '受凉后易诱发' : ['受凉后上述症状易诱发'],
    '乳腺后' : ['乳腺'],
    '结肠环周切缘' : ['环周切缘'],
    '呈咖啡色，偶有异味，同房后症状加重，增多' : ['增多，呈咖啡色，偶有异味，同房后症状加重'],
    '左心室下壁' : ['左室下壁'],
    '右心房' : ['右房'],
    '左心室' : ['左室'],
    '左心房' : ['左房'],
    '进行性，面色' : ['进行性面色'],
    '左侧冠状动脉前降支远端壁' : ['左侧冠状动脉钱简直远端壁'], # 错字
    '乙状结肠交界' : ['直乙交界'],
    '带脓血' : ['脓血'],
    '增多，量少，呈糊状' : ['增多，约5-6次/天，量少呈糊状'],
    '中分化，侵及全层，近远端未见癌累及' : ['中分化腺癌，侵及全层，近远端未见癌累及'],
    '肿瘤性，累及' : ['肿瘤性病变，累及'],
    '化疗后，缩小' : ['化疗后患者面部包块缩小'],
    '左侧颈部' : ['左侧面颈部'],
    '胸膜腔内' : ['胸膜腔腔内'],
    '丧失' : ['聋'],
    '腰1\\/2、3\\/4、4\\/5、腰5\\/骶1椎间盘' : ['腰11/2、3/4、4/5、腰5/骶1椎间盘'],
    '腰椎L3椎体' : ['腰33椎体'],
    '腰椎L1椎体' : ['腰1椎体'],
    '向右右侧' : ['向后右侧'],
    '腰椎L4椎体' : ['腰44椎体'],
    '为持续性胀痛，久坐久站后加重，卧位休息可缓解' : ['为持续性胀痛，久坐、久站后加重，卧位休息可缓解'],
    '豌豆大小' : ['“豌豆”大小'],
    '进食过多过快时发生' : ['进食过多、过快时发生'],
    '类圆形，软组织' : ['类圆形软组织'],
    '右肺中叶内侧段下叶背段' : ['右肺中叶内侧段、下叶背段'],
    '左侧腹股沟区淋巴结' : ['（左侧腹股沟）淋巴结'] ,
    '稍增大' : ['略有增大'],
    '双前臂尺侧' : ['双手前臂尺侧'],
    '动脉区' : ['动脉期'],
    '长期，升高' : ['长期处于升高状态'],
    'SD' : ['稳定'],
    '进食半流质食物明显' : ['半流质食物明显'],
    '进食固体食物明显' : ['进食固体食物及半流质食物明显'],
    '颜色深黄' : ['深黄'],
    '脂肪组织淋巴结' : ['脂肪组织中未发现明显肿大淋巴结'],
    '粘液，结节' : ['结节黏液'],
    '局部，复发，远处转移' : ['局部复发及远处转移'],
    '第77组淋巴结' : ['第77组组淋巴结'],
    '肠系膜淋巴结' : ['肠系膜多发淋巴结', '肠系膜可见肿大淋巴结'],
    '半流质、流质食物' : ['半流质、流质饮食'],
    '结干' : ['干燥'],
    '腹腔' : ['腹盆腔'],
    '同房后' : ['同房'],
    '双侧第1跖趾关节' : ['双侧第一跖趾关节'],
    '腹腔1212、16组' : ['腹腔第1212、16组'],
    '吻合口上方下方右侧腰大肌旁' : ['吻合口上方、下方右侧腰大肌旁'],
    '无回声区内透声好' : ['无回声区区内透声好'],
    '尤以T2-4、T7、T8椎体严重' : ['尤以T2-4、T7椎体，T8椎体严重'],
    '模糊' : ['眼花'],
    '夜间平卧时明显，放疗后' : ['以夜间平卧时明显'],
    '控制尚可' : ['控制血糖尚可'],
    '食管中段下段' : ['食管中下段'],
    '腰椎L4-L5椎体右侧' : ['原L4-L5椎体右侧'],
    '尺桡骨中段下段' : ['左尺桡骨中下段'],
    '踇指' : ['拇指'], # 错字
    '左足背' : ['左足足背'],
    '甲状软骨水平气管' : ['甲状软骨水平-气管'],
    '腰椎椎体' : ['腰椎多椎体'],
    '胸椎椎体' : ['胸腰椎多椎体'],
    '实性肿物，经彩超引导下' : ['经彩超引导下左肺实性肿物'],
    '治疗后缓解' : ['治疗后有所缓解'],
    '腰5椎体骶棘' : ['腰55椎体骶棘'], 
    '双乳囊内' : ['双乳多发囊肿，部分囊内'],
    '鲜红色，痰' : ['鲜红色血痰'],
    '带鲜红色血丝' : ['带血丝，为鲜红色血丝'],
    '乳房基底' : ['基底'],
    '乳房皮肤' : ['皮肤'], 
    '中上腹部' : ['中上腹上腹部'],
    '左腋窝淋巴结' : ['（左腋窝）淋巴结'],
    '腋窝淋巴结' : ['腋窝有5个淋巴结'],
    '胸12椎体' : ['胸12'],
    '非典型性' : ['不典型'],
    '十二指肠乳头周围' : ['十二指肠乳头'],
    '旋转' : ['眩晕'],
    '左颧弓' : ['左颞骨及颧弓'],
    '暗红色' : ['暗红'],
    '带血丝' : ['少量血丝'],
    '左肺下叶远端' : ['左肺下叶肿块伴远端'],
    '活动后或夜间加重，休息后可自行缓解' : ['活动后或夜间加重，每次持续时间约20-30秒，休息后可自行缓解'],
    '胸椎T10椎体' : ['胸10椎体'],
    '胸椎体周围' : ['椎体周围'],
    '双肾周' : ['双肾肾周'],
    '无咖啡样' : ['无咖啡色样'],
    '易咳出' : ['易咯出'],
    'RCA内皮中远段' : ['RCA近段见原支架内再狭窄85%，中段原支架内血流通畅，中远段'],
    'RCA中段' : ['RCA近段见原支架内再狭窄85%，中段'],
    '冠状动脉OM远段' : ['OM远段'],
    '冠状动脉远段' : ['远段'],
    '冠状动脉中段' : ['中段'],
    '冠状动脉LCX' : ['LCX'],
    '冠状动脉中远段' : ['中远段'],
    '冠状动脉中段分叉后' : ['中段分叉后'],
    '冠状动脉LAD近段' : ['LAD近段'],
    '冠状动脉LM分叉处' : ['LM分叉处'],
    '右上肺右上叶开口右上叶开口' : ['右上叶开口'],
    '右上肺右上叶开口' : ['右上叶开口'],
    '密度低' : ['低密度'],
    '如上诉' : ['如上述'],
    '无血凝块' : ['无鲜血、血凝块'],
    '继发性' : ['继发型'],
    '同房后，少量，呈鲜红色' : ['同房后有少量阴道流血，呈鲜红色'],
    '子宫腔底部' : ['子宫腔后壁及底部'],
    '左侧腹股沟淋巴结组织' : ['左侧腹股沟腹股沟区淋巴结'],
    '左侧腹股沟淋巴结组织结构' : ['<左腹股沟淋巴结穿刺活检标本>淋巴组织结构'],
    '矫正' : ['纠正'],
    '食管静脉' : ['食道胃底静脉'],
    '7、8、9、10、11、12组淋巴结': ['<7、8、9、10、11、12组>淋巴结'],
    '腹膜后平腰椎L2L4椎体前缘' : ['腹膜后约平腰2-腰44椎体前缘'],
    '13组淋巴结' : ['13组组淋巴结'],
    '进食后、平卧位、身体前驱时出现' : ['进食后及平卧位、身体前驱时出现'],
    '闻及刺激性气味无诱发或加重' : ['闻及刺激性气味咳嗽无诱发或加重'],
    '增强扫描可强化' : ['增强扫描可见强化'],
    '腹主动脉旁淋巴结' : ['腹主动脉旁'],
    '持续时间约1小时才缓解缓解' : ['持续时间约1小时才缓解'],
    '前列腺中央带及外周' : ['前列腺增大，中央带及外周'],
    '左侧输尿管上端' : ['左侧输尿管及膀胱腔腔内双J管影，上端'],
    '乙状结肠降结肠交界处浆膜面' : ['乙状结肠降结肠交界处局部肠壁增厚，管腔狭窄，浆膜面'],
    '乙状结肠降结肠交界处管腔' : ['乙状结肠降结肠交界处局部肠壁增厚，管腔'],
    '右侧左侧阴道前壁前壁血管' : ['左侧阴道前壁、右侧阴道前壁'],
    '受凉后' : ['因受凉'],
    '站立、坐位、改变体位时加重' : ['站立、坐位、改变体位时腹痛加重'],
    '平卧时好转' : ['平卧时胀痛好转'],
    '贲门胃壁' : ['贲门部胃壁'],
    '受凉后加重' : ['受凉后呼吸困难加重', '受凉后再次出现咳嗽咳痰症状加重'],
    '粘液性' : ['粘液便'],
    '胸椎放射性浓聚区' : ['胸椎及肋骨放射性浓聚区'],
    '右肱骨上段局部骨皮质' : ['右肱骨上段放射性浓聚区'],
    '休息后缓解' : ['休息后可缓解'],
    '爬坡、爬楼、快走、劳动等中等量体力活动后加重' : ['爬坡、爬楼、快走、劳动等中等量体力活动后出现上述不适加重'],
    '与腹泻交替' : ['腹泻与便秘交替'],
    '休息后稍缓解' : ['休息后声嘶可稍缓解', '休息后可稍缓解', '休息后双下肢疼痛可缓解，乏力稍缓解'],
    '双侧盆壁淋巴结' : ['双侧盆壁、腹股沟区淋巴结'],
    '约1分钟后自行缓解' : ['约1分1分钟后自行缓解'],
    '夜间最明显' : ['夜间头昏症状最明显'],
    '此后1-2小时后逐渐加重' : ['此后1-2小时后头昏症状逐渐加重'],
    '早晨及午睡后较轻' : ['早晨及午睡后头昏较轻'],
    '近邻右侧膈肌' : ['邻近右侧膈肌'],
    '右侧腋窝下' : ['右侧腋窝腋窝下'],
    '腰椎L3/4椎间盘' : ['腰33/4椎间盘'],
    '颈椎C3/4-6/7椎间盘' : ['颈33/4-6/7椎间盘'],
    '腰椎L4-5水平上方椎管内' : ['腰4-5水平椎管内'],
    '肠系膜下淋巴结' : ['肠系膜下）淋巴结'],
    '左侧肾脏组织未见癌累及' : ['<左侧>肾脏组织未见癌累及'],
    '表面或大便中暗红色血液' : ['表面或大便中混有暗红色血液'],
    '明显变形' : ['明显变细、变形'],
    '皮肤黏膜' : ['皮肤粘膜'],
    '代谢异常增高' : ['代谢未见异常增高'],
    '8组淋巴结' : ['8组组淋巴结'],
    '占位性' : ['占位'],
    '右肩部皮下皮下脂肪层' : ['右肩部肩部皮下皮下脂肪层'],
    '弯曲体位时可稍缓' : ['弯曲体位时腹痛可稍缓解'],
    '粘痰' : ['黏痰'],
    '休息后可自行缓解' : ['休息后头昏及头痛症状可自行缓解'],
    '绿豆大小' : ['“绿豆”大小'],
    '右下肢从足底沿胫前至大腿中部' : ['右下肢从足底沿胫前放射至大腿中部'],
    '左下肢从足底沿胫前区至膝部' : ['左下肢从足底沿胫前区放射至膝部'],
    '活动后明显' : ['活动后喘累明显'],
    '干性为主' : ['干咳为主'],
    '支气管黏膜' : ['支气管粘膜'],
    '次数正常' : ['大便次数及性状正常'],
    '进食干硬大块食物明显' : ['进食干硬及大块食物时明显'],
    '进食加重' : ['进食哽咽加重'],
    '左肺下叶外基底段胸膜下' : ['下叶外基底段胸膜下'],
    '右乳小叶' : ['（右乳）浸润性小叶'],
    '持续4分钟' : ['持续约4分4分钟'],
    '双乳房皮肤' : ['双乳胀痛，局部皮肤'],
    '服用头痛粉' : ['服用“头痛粉”'],
    '胸膜转移IV期' : ['胸膜转移（IV期，T4NXM1）'],
    '5、10组，6组，7组，9组，9组，10组，12组淋巴结' : ['<5、10组，6组，7组，9组，9组，10组，12组>淋巴结'],
    '较前明显增大' : ['与前次CT比较肝内病灶明显增大'],
    '3极' : ['3级'],
    '摔倒后加重' : ['摔倒后胸痛加重'],
    '系膜内淋巴结' : ['系膜内未扪及确切肿大淋巴结'],
    '较前增大' : ['较前有所增大'],
    '如常' : ['如上述'], # 错字？
    '右侧小脑血管' : ['右侧小脑占位性病变海绵状血管'],
    '尿道' : ['尿路'],
    '侧支循环' : ['侧枝循环'],
    '肝脏表面' : ['肝脏色暗红，表面'],
    '轻微压痛' : ['轻压痛'],
    '持续约2分钟' : ['持续约2分2分钟'],
    '休息后无缓解' : ['休息2天后腰痛无缓解'],
    '偶有' : ['偶感'],
    '胃贲门区胃壁' : ['胃贲门贲门区胃壁'],
    '左侧腿' : ['左侧腰腿'],
    '左侧腰腿部' : ['左侧腰腿腿部'],
    '久站久坐久行后加重' : ['久站、久坐及久行后加重'],
    '逐渐变细' : ['渐行变细'],
    '右侧盆腔病变组织' : ['右侧盆腔盆腔病变组织'],
    '体位改变时明显加重' : ['体位改变时症状明显加重'],
    '右侧肾周' : ['右侧肾肾周'],
    '腹' : ['泻'],
    '腰L4椎体' : ['腰44椎体'],
    '口服药物退热治疗后降至正常' : ['口服药物退热治疗后体温降至正常'],
    '口服药物后缓解' : ['口服药物后疼痛缓解'],
    '持续10秒左右自行缓解' : ['持续10秒左右自行缓'],
    '胃' : ['烧心'],
    '腹水见' : ['（腹水）涂片见'],
    '髂外淋巴结' : ['髂外，左盆腔，右盆腔）淋巴结'],
    '行走时加重' : ['行走时疼痛加重'],
    '稍好转' : ['稍有好转'],
    '平路行走后加重' : ['（平路行走）心累、气促加重'],
    '活动后反复发作' : ['活动后心累气促仍旧反复发作'],
    '停止活动可缓解' : ['停止活动疼痛可缓解'],
    '体位变化时加重' : ['体位变化时头晕加重'],
    '腰1/2、3/4、4/5、腰5/骶1椎间盘' : ['腰11/2、3/4、4/5、腰5/骶1椎间盘'],
    '腰3椎体' : ['腰33椎体'],
    '腰L3椎体' : ['腰33椎体'],
    '腰L1椎体' : ['腰1椎体'],
    '久坐久站后加重' : ['久坐、久站后加重'],
    '右肺胸膜' : ['右肺右胸膜'],
    '下腹部皮肤' : ['下腹部'],
    '双前臂尺侧皮肤' : ['双手前臂尺侧'],
    '治疗后好转' : ['治疗后骨髓抑制好转'],
    '左下肺1区' : ['左下肺门区'],
    '腰5椎' : ['腰55椎'],
    '1111组淋巴结' : ['1111组组淋巴结'],
    '气管中段淋巴组织' : ['（气管中段）淋巴组织'],
    '吻合口上方、下方右侧腰大肌旁腹腔1212、16组淋巴结' : ['吻合口上方、下方右侧腰大肌旁及腹腔第1212、16组多发淋巴结'],
    '右乳乳腺' : ['<右乳>乳腺'],
    '停止2天' : ['停止排气、排便2天'],
    '甲状软骨水平气管壁' : ['甲状软骨水平-气管分叉水平气管壁'],
    '乳头周围' : ['乳头肿大，尖端周围'],
    '左输尿管上段腔内' : ['左输尿管上段扩张伴腔内'],
    '无回声区内' : ['无回声区区内'],
    '子宫右前壁' : ['右前壁'],
    '每次持续时间约20-29秒' : ['每次持续时间约20-30秒'],
    '无恶性细胞' : ['无性细胞'],
    'RCA中远段内皮' : ['中远段支架内可见内皮'],
    '冠脉OM远段' : ['OM远段'],
    '冠脉LCX远段' : ['LCX弥漫斑块，中段狭窄60%-70%，可见溃疡和瘤样扩张，远段'],
    '冠脉LCX中段' : ['LCX弥漫斑块，中段'],
    '冠脉LCX' : ['LCX'],
    '冠脉LAD中远段' : ['LAD近段原支架内血流通畅，中段分叉后狭窄70%,中远段'],
    '冠脉LAD中段分叉后' : ['LAD近段原支架内血流通畅，中段分叉后'],
    '冠脉LAD近段' : ['LAD近段'],
    '冠脉LM分叉处' : ['LM分叉处'],
    '被新生物' : ['被一菜花样新生物'],
    '右肺下周围' : ['右肺下叶片块影并周围'],
    '乙型' : ['乙肝'],
    '1期间程度无明显变化' : ['期间乏力程度无明显变化'],
    '休息后可缓解' : ['休息后双下肢疼痛可缓解'],
    '子宫' : ['功血'],

    ####  for tendency
    '否定' : ['不伴', '未见', '未诉', '无', '愈合良好', '为见', '未提示', '否认', 
            '未达', '未再', '未述', '不能', '未', '不明显'],
    '不确定' : ['可能大', '考虑', '可能', '待排', '待查', '倾向', '可疑', '不除外', 
            '不完全除外', '支持', '似见', '如为', '疑为', '待定', '？', '?', '可', '以', 
            '拟', '未见确切', '进一步确诊'],
    '["加重"]' : ['转移'],
}


new2text = {}
for k in text2new.keys():
    for v in text2new[k]:
        if v in new2text.keys():
            new2text[v].append(k)
        else:
            new2text[v] = [k]

new2value = {}
for k in value2new.keys():
    for v in value2new[k]:
        if v in new2value.keys():
            new2value[v].append(k)
        else:
            new2value[v] = [k]

new2tendency = {}
for k in ['否定', '不确定', '["加重"]']:
    for v in value2new[k]:
        if v in new2tendency.keys():
            new2tendency[v].append(k)
        else:
            new2tendency[v] = [k]


if __name__ == '__main__':
    print('new2text = ', json.dumps(new2text, indent=4, ensure_ascii=False, sort_keys=True))
    print('\n')
    print('new2value = ', json.dumps(new2value, indent=4, ensure_ascii=False, sort_keys=True))
    print('\n')
    print('new2tendency = ', json.dumps(new2tendency, indent=4, ensure_ascii=False, sort_keys=True))
