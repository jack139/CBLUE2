python3 Text2DT_TreeDecoder.py \
    --config_file config.yml \
    --save_dir ckpt/Text2DT \
    --data_dir data/Text2DT \
    --bert_model_name "../../nlp_model/chinese-bert-wwm-ext_pytorch" \
    --epochs 100 \
    --fine_tune \
    --device 0