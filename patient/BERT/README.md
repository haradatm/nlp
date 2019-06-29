### Put BERT Pretrained model here

Downlod [BERT Japanese Pretrained model](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT日本語Pretrainedモデル) and extract them here.

- Japanese_L-12_H-768_A-12_E-30_BPE/arrays_bert_model.ckpt.npz
- Japanese_L-12_H-768_A-12_E-30_BPE/bert_config.json
- Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt

In advance, you need to convert a BERT TensorFlow checkpoint in a Chainer save file by using the convert_tf_checkpoint_to_chainer.py script. [See also](https://github.com/soskek/bert-chainer)

```
export BERT_BASE_DIR=BERT/Japanese_L-12_H-768_A-12_E-30_BPE

python convert_tf_checkpoint_to_chainer.py \
  --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
  --npz_dump_path $BERT_BASE_DIR/arrays_bert_model.ckpt.npz

```

[See also] https://github.com/soskek/bert-chainer