#! -*- coding:utf-8 -*-
# 通过梯度惩罚增强模型的泛化性能
# 适用于Keras 2.3.1

import json
import numpy as np
import codecs
import copy
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from sklearn.metrics import f1_score
from tqdm import tqdm
from category import category_index, category_name

num_classes = 4
maxlen = 512
batch_size = 8
learning_rate = 5e-6 # 8e-6 

# BERT base
config_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    for l in json.load(open(filename)):
        D.append((l['text_a'], l['text_b'], category_index(l['label'])))
    return D


# 加载数据集
train_data = load_data('data/filter_train.json')
valid_data = load_data('data/filter_dev.json')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()


def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
    return loss + 0.5 * epsilon * gp


model.compile(
    loss=loss_with_gradient_penalty,
    optimizer=Adam(learning_rate),
    metrics=['sparse_categorical_accuracy'],
)


def evaluate(data):
    f1_true = np.array([], 'int32')
    f1_pred = np.array([], 'int32')
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        f1_true = np.append(f1_true, y_true)
        f1_pred = np.append(f1_pred, y_pred)
    return f1_score(f1_true, f1_pred, average='macro')

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = evaluate(valid_generator)
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            model.save_weights('mdcfnpc_gp_best_f1_%.5f.weights'%val_f1)
        print(
            u'val_f1: %.5f, best_val_f1: %.5f\n' %
            (val_f1, self.best_val_f1)
        )


def predict_to_file0(in_file, out_file):
    """预测到文件
    """
    data = json.load(open(in_file))
    for l in tqdm(data, ncols=100):
        text1 = l['text_a']
        text2 = l['text_b']
        token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
        label = model.predict([[token_ids], [segment_ids]])[0].argmax()
        l['label'] = category_name(label)

    json.dump(
        data,
        open(out_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )



def predict_to_file(infile, outfile):

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

                    text1 = entity_['text_a']
                    text2 = entity_['text_b']
                    token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
                    label = model.predict([[token_ids], [segment_ids]])[0].argmax()
                    dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = category_name(label)
                    
                    #entity_['label'] = term_['attr']
                    idx_ += 1
                    
                    #if dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] != tm_predictor_instance.predict_one_sample([entity_['text_a'], entity_['text_b']]):
                    #    dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = ''

                    #submit_result.append({
                    #    'text_a' : entity_['text_a'],
                    #    'text_b' : entity_['text_b'],
                    #    'label'  : entity_['label'] if entity_['label']!='' else '不标注'
                    #})

        submit_result.append(dialogue_)
        
    with open(outfile, 'w') as output_data:
        for json_content in submit_result:
            output_data.write(json.dumps(json_content, ensure_ascii=False) + '\n')


if __name__ == '__main__':

    evaluator = Evaluator()

    model.load_weights("./mdcfnpc_gp_best_f1_0.76633.weights")

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=30,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./mdcfnpc_gp_best_f1_0.76633.weights')
    predict_to_file('../dataset/CHIP-MDCFNPC/CHIP-MDCFNPC_test.jsonl', 'CHIP-MDCFNPC_test.jsonl')
