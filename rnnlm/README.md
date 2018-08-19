# RNNLM (Recurrent Net Language Model) example

### Description

This example code is a recurrent net for language modeling using three kinds of approaches, `BPTT (back-propagation through time) LSTM`, `NStep LSTM`.

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
- training

```
python train_rnnlm-bptt.py  --train datasets/soseki/neko-word-train.txt --test datasets/soseki/neko-word-test.txt --w2v datasets/soseki/neko_w2v.bin --gpu 0 --epoch 300 --batchsize 100 --unit 200 --bproplen 35 --out results/rnnlm-w2v-bptt  2>&1 | tee results/train_rnnlm-w2v-bptt.log
python train_rnnlm-nstep.py --train datasets/soseki/neko-word-train.txt --test datasets/soseki/neko-word-test.txt --w2v datasets/soseki/neko_w2v.bin --gpu 0 --epoch 300 --batchsize 100 --unit 200               --out results/rnnlm-w2v-nstep 2>&1 | tee results/train_rnnlm-w2v-nstep.log 
```

- test (your own text-generating)
```
python  test_rnnlm-bptt.py  --model results/rnnlm-w2v-bptt/early_stopped.model  --text "吾輩 は 猫 で ある 。" 2>&1 | tee results/test_rnnlm-w2v-bptt.log 
python  test_rnnlm-nstep.py --model results/rnnlm-w2v-nstep/early_stopped.model --text "吾輩 は 猫 で ある 。" 2>&1 | tee results/test_rnnlm-w2v-nstep.log
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

- train_rnnlm-w2v-bptt.log (use **train_rnnlm-bptt.ipynb** on **Google Colaboratory**)
```
2018-08-19 11:27:32,705 - main - INFO - vocabulary size: 13948
2018-08-19 11:27:32,707 - main - INFO - train data size: 208502
2018-08-19 11:27:32,713 - main - INFO - train data starts with: 吾輩 は 猫 で ある 。 ...
2018-08-19 11:27:32,715 - main - INFO - test  data size: 1735
Initialize the embedding from word2vec model: datasets/soseki/neko_w2v.bin
going to train 625500 iterations (300 epochs)
2018-08-19 11:28:34,263 - main - INFO - [  1] T/loss=7.321579 T/acc=0.034657 T/perp=3177.996886 T/sec= 36.996148 D/loss=6.078034 D/acc=0.050000 D/perp=436.170881 D/sec= 0.061185 lr=0.000700
SAMPLE #=> 吾輩は猫である。、しからながらため月並手云う顔が堂々たる現に出来かと時魂消るた動か用
 :
2018-08-19 14:20:45,239 - main - INFO - [300] T/loss=4.009955 T/acc=0.314206 T/perp=58.074400 T/sec= 24.630806 D/loss=6.091312 D/acc=0.227647 D/perp=442.883241 D/sec= 0.065305 lr=0.000156
SAMPLE #=> 吾輩は猫である。</s>純然たら蒟蒻べからんがあ、えらいな。</s>知らないが、すべてで妙
loading early stopped-model at epoch 276
吾輩は猫である。
そのあてつけるに行き休むだ。
誰を知り出すか、あの事は私はとうてい駄目か。と医者が満更首を返らんて付をなる。
少し食べていただきたい。
それで何だか叔母さんだ。と云う。
2018-08-19 14:21:09,672 - main - INFO - time spent: 10434.439452 sec
```

- train_rnnlm-w2v-nstep.log (use **train_rnnlm-nstep.ipynb** on **Google Colaboratory**)
```
2018-08-18 12:36:25,397 - main - INFO - vocabulary size: 13948
2018-08-18 12:36:25,398 - main - INFO - train data size: 8951
2018-08-18 12:36:25,403 - main - INFO - train data starts with: 吾輩 は 猫 で ある 。 ...
2018-08-18 12:36:25,405 - main - INFO - test  data size: 100
Initialize the embedding from word2vec model: datasets/soseki/neko_w2v.bin
2018-08-18 12:36:50,768 - main - INFO - [  1] T/loss=153.004472 T/acc=0.047866 T/perp=2103.527864 T/sec= 21.964320 D/loss=100.016502 D/acc=0.070948 D/perp=453.600800 D/sec= 0.120360 lr=0.000700
SAMPLE #=> 吾輩は猫である。。つい、上押し返しとがた</s>に今でそんな</s>はなけれ</s>よりはある
 :
2018-08-18 14:19:30,912 - main - INFO - [300] T/loss=98.843602 T/acc=0.374162 T/perp=84.322790 T/sec= 14.485338 D/loss=89.144028 D/acc=0.201835 D/perp=233.278687 D/sec= 0.043545 lr=0.000156
SAMPLE #=> 吾輩は猫である。</s>に話しかける。</s>は始まる宴会の泥棒が筧に目だたである天下煖ほどで
loading early stopped-model at epoch 73
吾輩は猫である。
の気になってるものじゃない。寒月君は読売新聞のあたりへ京焼杯入れて鞍何らしいものが想像します。
から妻が聖書掌に行って彼だって近日的に帰るんでございませんなどだから宜だ。
と聞かれたから、ほかにその客なんぞ先月の？
生きますから返す返すまで裸体とか大変、まだいい車屋が鋭目へ呼びかけて、冷笑してくれが吾輩は撮っるを似合？
2018-08-18 14:20:05,497 - main - INFO - time spent: 6233.629851 sec
```

- train_rnnlm-bptt-w2v.png (use **train_rnnlm-bptt.py**)

<img src="results/train_rnnlm-bptt-w2v.png" width="262px" height="261px"/> <img src="results/02-turn-1.png" width="262px" height="261px"/> <img src="results/03-turn-1.png" width="262px" height="261px"/>

- train_rnnlm-nstep-w2v.png (use **test_rnnlm-nstep.py**)

<img src="results/train_rnnlm-nstep-w2v.png" width="262px" height="261px"/> <img src="results/02-turn-2.png" width="262px" height="261px"/> <img src="results/03-turn-2.png" width="262px" height="261px"/>
