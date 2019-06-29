### Put BERT Pretrained model here

Downlod [BERT Japanese Pretrained model](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT日本語Pretrainedモデル) and extract them here.

- uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz
- uncased_L-12_H-768_A-12/bert_config.json
- uncased_L-12_H-768_A-12/vocab.txt

In advance, you need to convert a BERT TensorFlow checkpoint in a Chainer save file by using the convert_tf_checkpoint_to_chainer.py script.

[See also] https://github.com/soskek/bert-chainer