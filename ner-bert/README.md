# BERT-NER (Neural Architectures for Named Entity Recognition using BERT fine-tuning)

### Description

This example code is a named entity recognition using BERT fine-tuning.

- ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Jacob Devlin, et al.,](https://arxiv.org/abs/1810.04805)

### Dependencies
- python 3.7
- chainer 5.4

In addition, please add the project folder to PYTHONPATH and `conca install` the following packages:
- `matplotlib`
- `seqeval`

### Usage ###

### Preparation ###

***BERT Pretrained model***

  - Downlod [Pretrained model](https://github.com/google-research/bert) and extract them in "BERT".

***Data***

  - Downlod [CoNLL-2003 Datasets](https://github.com/glample/tagger/tree/master/dataset) and put them in the appropriate place.

  - Convert from BIO to IOBES format as follows.

```
cd datasets
python ../tools/conv_iobes.py --file eng.train > eng.train.bioes
python ../tools/conv_iobes.py --file eng.testa > eng.testa.bioes
python ../tools/conv_iobes.py --file eng.testb > eng.testb.bioes
```


***Run and Evaluate***

```
python train_ner-bert.py \
--gpu 0 \
--batchsize 64 \
--learnrate 5e-05 \
--weightdecay 0.01 \
--epoch 30 \
--train datasets/eng.train.bioes \
--test  datasets/eng.testb.bioes \
--init_checkpoint  BERT/uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz \
--bert_config_file BERT/uncased_L-12_H-768_A-12/bert_config.json \
--vocab_file       BERT/uncased_L-12_H-768_A-12/vocab.txt \
--out results_ner-bert \
2>&1 | tee train_ner-bert.log
```


***Input***

- format
```
[word] [tag]
[word] [tag]
...
```

- eng.train  (**BIO** format)
```
EU   B-ORG
rejects O
German  B-MISC
call    O
to  O
boycott O
British B-MISC
lamb    O
.   O

Peter   B-PER
Blackburn   I-PER
...
```

- eng.train.bioes  (**IOBES** format)
```
EU	S-ORG
rejects	O
German	S-MISC
call	O
to	O
boycott	O
British	S-MISC
lamb	O
.	O

Peter	B-PER
Blackburn	E-PER
...
```

***Output***

- train_ner-bert.log (use **train_ner-bert.py**)
```
2019-08-13 01:48:53,295 - load_data - INFO - Loading dataset ... done.
2019-08-13 01:48:54,492 - load_data - INFO - Loading dataset ... done.
# train: 14986, test: 3683, class: 20
# vocab: 30522
2019-08-13 01:54:55,990 - main - INFO - [  1] T/loss=0.140780 T/f1=0.820586 T/acc=0.962188 T/sec= 334.482679 D/loss=0.101339 D/f1=0.875239 D/acc=0.972458 D/sec= 14.611923 lr=0.779989 eta=0.000050
saving early-stopped model (loss) at epoch 1
saving early-stopped model (f1) at epoch 1

==== Classification report (early-stopped model) ====

           precision    recall  f1-score   support

      ORG       0.82      0.85      0.83      1661
      LOC       0.87      0.92      0.89      1668
      PER       0.97      0.96      0.96      1617
     MISC       0.71      0.78      0.75       702

micro avg       0.86      0.89      0.88      5648
macro avg       0.86      0.89      0.88      5648

2019-08-13 02:02:27,434 - main - INFO - [  2] T/loss=0.038447 T/f1=0.943426 T/acc=0.989142 T/sec= 436.854918 D/loss=0.100838 D/f1=0.895060 D/acc=0.975641 D/sec= 14.589645 lr=0.920128 eta=0.000048
saving early-stopped model (loss) at epoch 2
saving early-stopped model (f1) at epoch 2

==== Classification report (early-stopped model) ====

           precision    recall  f1-score   support

      PER       0.97      0.96      0.96      1617
     MISC       0.73      0.81      0.77       702
      ORG       0.85      0.87      0.86      1661
      LOC       0.91      0.93      0.92      1668

micro avg       0.89      0.90      0.90      5648
macro avg       0.89      0.90      0.90      5648

2019-08-13 02:10:02,165 - main - INFO - [  3] T/loss=0.021613 T/f1=0.967268 T/acc=0.994080 T/sec= 439.982089 D/loss=0.131141 D/f1=0.888384 D/acc=0.973491 D/sec= 14.748159 lr=0.969505 eta=0.000047
2019-08-13 02:16:47,199 - main - INFO - [  4] T/loss=0.014813 T/f1=0.976431 T/acc=0.995868 T/sec= 390.369368 D/loss=0.127803 D/f1=0.885682 D/acc=0.974168 D/sec= 14.664813 lr=0.988170 eta=0.000045
2019-08-13 02:23:30,766 - main - INFO - [  5] T/loss=0.008865 T/f1=0.983228 T/acc=0.997313 T/sec= 388.851013 D/loss=0.144984 D/f1=0.893967 D/acc=0.975912 D/sec= 14.715833 lr=0.995384 eta=0.000043
2019-08-13 02:30:12,162 - main - INFO - [  6] T/loss=0.008053 T/f1=0.986839 T/acc=0.997768 T/sec= 386.888935 D/loss=0.163425 D/f1=0.879993 D/acc=0.973660 D/sec= 14.507427 lr=0.998195 eta=0.000042
```

- result_ner-bert.png (use **train_ner-bert.py**)
