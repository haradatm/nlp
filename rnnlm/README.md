# RNNLM (Recurrent Net Language Model) example

### Description

This example code is a recurrent net for language modeling using three kinds of approaches, `Simple LSTM`, `NStep LSTM`.

### Dependencies
- python 3.6
- chainer 3.4

In addition, please add the project folder to PYTHONPATH and `conca install` the following packages:
- `matplotlib`

### Usage ###

***Data***

  - Downlod [青空文庫](https://www.aozora.gr.jp/cards/000148/card789.html) and put them in the appropriate place.
  - Create train and test datasets and put them in the appropriate place.

```
cd datasets/soseki
cat neko.txt | sed -e "s/\(.\)/\1 /g" | sed -e "s/ $//g" > neko-char.txt
cat neko.txt | mecab -b 81920 -Owakati > neko-word.txt
word2vec -train neko-word.txt -output neko_w2v.bin -cbow 0 -size 200 -window 5 -negative 1 -hs 1 -sample 0.001 -threads 4 -min-count 1 -binary 1

head -n 8951 neko-word.txt > neko-word-train.txt
tail -n 100  neko-word.txt > neko-word-test.txt

wc -l neko-word*.txt
    8951 neko-word-train.txt
     100 neko-word-test.txt
    9051 neko-word.txt
   18102 total

cd ../../
```

***Run and Evaluate***

```
python train_rnnlm.py       --train datasets/soseki/neko-word-train.txt --test  datasets/soseki/neko-word-test.txt --w2v datasets/soseki/neko_w2v.bin --gpu 0 --epoch 300 --batchsize 100 --unit 200 --bproplen 35 --out result_rnnlm-w2v       | tee train_rnnlm-w2v.log 2>&1
python train_rnnlm-nstep.py --train datasets/soseki/neko-word-train.txt --test  datasets/soseki/neko-word-test.txt --w2v datasets/soseki/neko_w2v.bin --gpu 0 --epoch 300 --batchsize 100 --unit 200               --out result_rnnlm-w2v-nstep | tee train_rnnlm-w2v-nstep.log 2>&1
```

***Input***

- format (space-separated test)
```
[token] [token] ... [token]
[token] [token] ... [token]
 :
```

- neko-word-train.txt
```
吾輩 は 猫 で ある 。
名前 は まだ 無い 。
どこ で 生れ た か とんと 見当 が つか ぬ 。
 :
```

- neko-word-test.txt
```
こんな 豪傑 が すでに 一 世紀 も 前 に 出現 し て いる なら 、 吾輩 の よう な
碌 で なし は とうに 御 暇 を 頂戴 し て 無 何 有 郷 に 帰臥 し て も いい はず で あっ た 。
主人 は 早晩 胃病 で 死ぬ 。
金田 の じいさん は 慾 で もう 死ん で いる 。
 :
```

***Output***

- train_rnnlm-w2v.log (use **train_rnnlm.py**)
```
2018-08-18 06:09:43,306 - main - INFO - vocabulary size: 13948
2018-08-18 06:09:43,311 - main - INFO - train data size: 208502
2018-08-18 06:09:43,312 - main - INFO - train data starts with: 吾輩 は 猫 で ある 。 ...
2018-08-18 06:09:43,319 - main - INFO - test  data size: 1735
Initialize the embedding from word2vec model: datasets/soseki/neko_w2v.bin
going to train 625500 iterations (300 epochs)
2018-08-18 06:15:17,925 - main - INFO - [  1] T/loss=7.314021 T/acc=0.030782 T/perp=3182.100133 T/sec= 332.478771 D/loss=6.096238 D/acc=0.050000 D/perp=444.183391 D/sec= 1.517477 lr=0.000700
SAMPLE #=> 吾輩は猫である。をが名さを鼠ことごとくもだろ。のかくし事がし奴黒い使て
2018-08-18 06:21:13,628 - main - INFO - [  2] T/loss=6.378471 T/acc=0.041856 T/perp=614.686833 T/sec= 354.200411 D/loss=6.348036 D/acc=0.180000 D/perp=571.370132 D/sec= 1.502880 lr=0.000696
SAMPLE #=> 吾輩は猫である。同もで</s>置い延びる手方法富子て安心ないか迷亭をしおめでたけりゃも上、
 :
```

- train_rnnlm-w2v-nstep.log (use **train_rnnlm-nstep.py**)
```
2018-08-18 06:09:37,145 - main - INFO - vocabulary size: 13948
2018-08-18 06:09:37,146 - main - INFO - train data size: 8951
2018-08-18 06:09:37,152 - main - INFO - train data starts with: 吾輩 は 猫 で ある 。 ...
2018-08-18 06:09:37,157 - main - INFO - test  data size: 100
Initialize the embedding from word2vec model: datasets/soseki/neko_w2v.bin
2018-08-18 06:09:53,117 - main - INFO - [  1] T/loss=153.408735 T/acc=0.050951 T/perp=2161.037404 T/sec= 15.201595 D/loss=100.005348 D/acc=0.067890 D/perp=453.291382 D/sec= 0.044296 lr=0.000700
SAMPLE #=> 吾輩は猫である。ねえさに音なり武ないと当る。間もなくものこのののにようをと聖書
2018-08-18 06:10:34,729 - main - INFO - [  2] T/loss=137.093931 T/acc=0.058924 T/perp=468.661673 T/sec= 41.567921 D/loss=99.580887 D/acc=0.070336 D/perp=441.674988 D/sec= 0.044635 lr=0.000696
SAMPLE #=> 吾輩は猫である。てか。磁力</s>ががにななろ云うたて否や遊弋よう全くて。た
 :
```
