# Chainer example code to write a Japanese paraphrasing using attention mechanism

## Requirements

- chainer==1.18.0
- nltk==3.1

## Description

This is an example code to write a to write a Japanese paraphrasing using attention mechanism.

```
python2.7 train_attention-concat.py --gpu 0 --dim 100 --unit 200 --batchsize 50 --epoch 10 --train data/ogura/train.txt --test data/ogura/test.txt
python2.7 train_attention-dot.py    --gpu 0 --dim 100 --unit 200 --batchsize 50 --epoch 10 --train data/ogura/train.txt --test data/ogura/test.txt
```
