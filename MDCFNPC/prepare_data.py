import json
import copy
import codecs
from tqdm import tqdm


def convert_data(infile, outfile):

    maxlen = 0

    submit_result = []

    with codecs.open(infile, mode='r', encoding='utf8') as f:
        reader = f.readlines(f)    

    data_list = []

    for dialogue_ in tqdm(reader):
        dialogue_ = json.loads(dialogue_)
        for content_idx_, contents_ in enumerate(dialogue_['dialog_info']):

            terms_ = contents_['ner']

            if len(terms_) != 0:
                idx_ = 0
                for _ner_idx, term_ in enumerate(terms_):

                    entity_ = dict()

                    entity_['dialogue'] = dialogue_
                    
                    _text = dialogue_['dialog_info'][content_idx_]['text']
                    _text_list = list(_text)
                    _text_list.insert(term_['range'][0], '[unused1]')
                    _text_list.insert(term_['range'][1]+1, '[unused2]')
                    _text = ''.join(_text_list)

                    if content_idx_ - 1 >= 0 and len(dialogue_['dialog_info'][content_idx_-1]) < 40:
                        forward_text = dialogue_['dialog_info'][content_idx_-1]['sender'] + ':' + dialogue_['dialog_info'][content_idx_-1]['text'] + ';'
                    else:
                        forward_text = ''

                    if contents_['sender'] == '医生':

                        if content_idx_ + 1 >= len(dialogue_['dialog_info']):
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                        else:
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text + ';'
                            temp_index = copy.deepcopy(content_idx_) + 1

                            speaker_flag = False
                            sen_counter = 0
                            while True:

                                if dialogue_['dialog_info'][temp_index]['sender'] == '患者':
                                    sen_counter += 1
                                    speaker_flag = True
                                    entity_['text_a'] += dialogue_['dialog_info'][temp_index]['sender'] + ':' + dialogue_['dialog_info'][temp_index]['text'] + ';'

                                if sen_counter > 3:
                                    break

                                temp_index += 1
                                if temp_index >= len(dialogue_['dialog_info']):
                                    break

                    elif contents_['sender'] == '患者':
                        if content_idx_ + 1 >= len(dialogue_['dialog_info']):
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                        else:
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text + ';'
                            temp_index = copy.deepcopy(content_idx_) + 1

                            speaker_flag = False
                            sen_counter = 0
                            while True:

                                sen_counter += 1
                                speaker_flag = True
                                entity_['text_a'] += dialogue_['dialog_info'][temp_index]['sender'] + ':' + dialogue_['dialog_info'][temp_index]['text'] + ';'

                                if sen_counter > 3:
                                    break

                                temp_index += 1
                                if temp_index >= len(dialogue_['dialog_info']):
                                    break
                    else:
                        entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                            
                        
                    if term_['name'] == 'undefined':
                        add_text = '|没有标准化'
                    else:
                        add_text = '|标准化为' + term_['name']

                    entity_['text_b'] = term_['mention']  + add_text
                    entity_['start_idx'] = term_['range'][0]
                    entity_['end_idx'] = term_['range'][1] - 1

                    entity_['label'] = term_['attr']
                    idx_ += 1
                    
                    #if dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] != tm_predictor_instance.predict_one_sample([entity_['text_a'], entity_['text_b']]):
                    #    dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = ''

                    submit_result.append({
                        'text_a' : entity_['text_a'],
                        'text_b' : entity_['text_b'],
                        'label'  : entity_['label'] if entity_['label']!='' else '不标注'
                    })

                    if len(entity_['text_a'])+len(entity_['text_b'])>512:
                        maxlen += 1
        #submit_result.append(dialogue_)
        
    #with open(outfile, 'w') as output_data:
    #    for json_content in submit_result:
    #        output_data.write(json.dumps(json_content, ensure_ascii=False) + '\n')

    json.dump(
        submit_result,
        open(outfile, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    print('total: %d\tmaxlen>512: %d'%(len(submit_result), maxlen))

if __name__ == '__main__':
    convert_data('../dataset/CHIP-MDCFNPC/CHIP-MDCFNPC_train.jsonl', 'data/filter_train.json')
    convert_data('../dataset/CHIP-MDCFNPC/CHIP-MDCFNPC_dev.jsonl', 'data/filter_dev.json')