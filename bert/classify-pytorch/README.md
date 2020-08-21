# BERT-Classification (PyTorch example code for Text Classification using BERT fine-tuning)

### Description

This example code is a text entclassification using BERT fine-tuning.

### Dependencies
- Python 3.7
- PyTorch 1.6.0
- Transformers 3.0.2
- MeCab 0.996 (for Japanese)

In addition, please add the project folder to PYTHONPATH and `conca install` the following packages:
- `matplotlib`
- `filelock`
- `tqdm`
- `mecab-python3` (for Japanese)

### Usage ###

### Preparation ###

***Data***

  - [Scale Movie Review Dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/) (rt-polarity): Predict its sentiment (positive/negative) from a review about a movie.
  - [Road Transport Bureau of MLIT](http://carinf.mlit.go.jp/jidosha/carinf/opn/index.html)
  - Create train and test datasets and put them in the appropriate place.

```
wc -l datasets/rt-polarity/04-{train,test}.txt
    9596 datasets/rt-polarity/04-train.txt
    1066 datasets/rt-polarity/04-test.txt
   10662 total

head -n 3 datasets/rt-polarity/04-train.txt
==> datasets/rt-polarity/04-train.txt <==
0	simplistic , silly and tedious .
0	it's so laddish and juvenile , only teenage boys could possibly find it funny .
0	exploitative and largely devoid of the depth or sophistication that would make watching such a graphic treatment of the crimes bearable .

head -n 3 datasets/rt-polarity/04-test.txt
0	a visually flashy but narratively opaque and emotionally vapid exercise in style and mystification .
0	while the performances are often engaging , this loose collection of largely improvised numbers would probably have worked better as a one-hour tv documentary .
0	on a cutting room floor somewhere lies . . . footage that might have made no such thing a trenchant , ironic cultural satire instead of a frustrating misfire .
```

```
wc -l datasets/mlit/04-{train,test}.txt
   46743 datasets/mlit/04-train.txt
    5197 datasets/mlit/04-test.txt
   51940 total

head -n 3 datasets/mlit/04-train.txt
エンジン	車庫にいれるために、右後方縦列駐車でバックしていたところ、アクセル操作をしていないのに車が急加速し、右後方の壁に激突した。
エンジン	一般道路を走行中、突然エンジンが停止した。
制動装置	高速道路を１００ｋｍ／ｈくらいで走行中、ＡＢＳのマーク、サイドブレーキのマークが表示された。

head -n 3 datasets/mlit/04-test.txt
車枠・車体	ダッシュボードが溶けてベトベトしている。
排ｶﾞｽ･騒音	ＮＯＸセンサーの不良により、エンジン警告灯が点きっぱなしになった。
車枠・車体	電動オープンのルーフを閉じるときに、エラーメッセージが出て幌が閉まらなくなった。
```

***Run and Evaluate***

- training for rt-polarity datasets (for English)

```
python train_bert.py \
--train datasets/rt-polarity/04-train.txt \
--valid datasets/rt-polarity/04-test.txt \
--pretrained "bert-base-uncased" \
--batchsize 64 \
--learnrate 2e-05 \
--epoch 10 \
--out results_bert-rt \
2>&1 | tee results/results_bert-rt.log

2020-08-01 04:53:19,574 - main - INFO - {
  "train": "datasets/rt-polarity/04-train.txt",
  "valid": "datasets/rt-polarity/04-test.txt",
  "pretrained": "bert-base-uncased",
  "batchsize": 64,
  "learnrate": 2e-05,
  "epoch": 10,
  "out": "results_bert-rt",
  "noplot": false
}
# train: 9596, valid: 1066
# class: 2, labels: {'0': 0, '1': 1}
# vocab: 30522
Linear(in_features=768, out_features=2, bias=True)
2020-08-20 04:39:42,234 - main - INFO - [  1] T/loss=0.433948 T/acc1=0.798875 T/acc2=0.000000 T/sec= 54.470282 D/loss=0.324746 D/acc1=0.869606 D/acc2=0.869606 D/sec= 2.155193 
saving early-stopped model (loss) at epoch 1
saving early-stopped model (uar) at epoch 1
2020-08-20 04:41:45,919 - main - INFO - [  2] T/loss=0.255309 T/acc1=0.902876 T/acc2=0.000000 T/sec= 121.530358 D/loss=0.300297 D/acc1=0.894934 D/acc2=0.894934 D/sec= 2.154646 
saving early-stopped model (loss) at epoch 2
saving early-stopped model (uar) at epoch 2
2020-08-20 04:43:49,857 - main - INFO - [  3] T/loss=0.150620 T/acc1=0.947582 T/acc2=0.000000 T/sec= 121.789301 D/loss=0.397301 D/acc1=0.873358 D/acc2=0.873358 D/sec= 2.148116 
2020-08-20 04:44:54,141 - main - INFO - [  4] T/loss=0.086604 T/acc1=0.972072 T/acc2=0.000000 T/sec= 62.131293 D/loss=0.464844 D/acc1=0.881801 D/acc2=0.881801 D/sec= 2.152615 
 :
2020-08-20 04:51:16,605 - main - INFO - [ 10] T/loss=0.022666 T/acc1=0.994268 T/acc2=0.000000 T/sec= 60.405383 D/loss=0.806572 D/acc1=0.876173 D/acc2=0.876173 D/sec= 2.175515 

==== Classification report (early_stopped-uar) ====

              precision    recall  f1-score   support

           0       0.90      0.89      0.89       533
           1       0.89      0.90      0.90       533

    accuracy                           0.89      1066
   macro avg       0.89      0.89      0.89      1066
weighted avg       0.89      0.89      0.89      1066
```

- training for mlit datasets (for Japanese)

```
python train_bert.py \
--train datasets/mlit/04-train.txt \
--valid datasets/mlit/04-test.txt \
--pretrained "cl-tohoku/bert-base-japanese-whole-word-masking" \
--batchsize 64 \
--learnrate 2e-05 \
--epoch 10 \
--out results_bert-mlit \
2>&1 | tee results/results_bert-mlit.log

2020-08-20 02:01:11,039 - main - INFO - {
  "train": "datasets/mlit/04-train.txt",
  "valid": "datasets/mlit/04-test.txt",
  "pretrained": "cl-tohoku/bert-base-japanese-whole-word-masking",
  "batchsize": 64,
  "learnrate": 2e-05,
  "epoch": 10,
  "out": "results_bert-mlt",
  "noplot": false
}
# train: 46742, eval: 5197,
# class: 16, labels: {'エンジン': 0, '制動装置': 1, '動力伝達': 2, '排ｶﾞｽ･騒音': 3, '乗車装置': 4, '保安灯火': 5, '車枠・車体': 6, 'かじ取り': 7, '電気装置': 8, '燃料装置': 9, 'その他': 10, '電動機(モーター)': 11, '緩衝装置': 12, '走行装置': 13, '装置その他': 14, '非装置': 15}
# vocab: 32005
Linear(in_features=768, out_features=16, bias=True)
2020-08-20 02:08:08,041 - main - INFO - [  1] T/loss=0.772249 T/acc1=0.785247 T/acc2=0.000000 T/sec= 360.639226 D/loss=0.481126 D/acc1=0.862998 D/acc2=0.715137 D/sec= 12.366193 
saving early-stopped model (loss) at epoch 1
saving early-stopped model (uar) at epoch 1
2020-08-20 02:16:52,732 - main - INFO - [  2] T/loss=0.428546 T/acc1=0.876214 T/acc2=0.000000 T/sec= 512.311893 D/loss=0.448293 D/acc1=0.874735 D/acc2=0.727704 D/sec= 12.378893 
saving early-stopped model (loss) at epoch 2
saving early-stopped model (uar) at epoch 2
2020-08-20 02:24:11,761 - main - INFO - [  3] T/loss=0.341935 T/acc1=0.899726 T/acc2=0.000000 T/sec= 426.646790 D/loss=0.458531 D/acc1=0.877044 D/acc2=0.723688 D/sec= 12.383017 
2020-08-20 02:30:32,634 - main - INFO - [  4] T/loss=0.273988 T/acc1=0.920671 T/acc2=0.000000 T/sec= 368.496383 D/loss=0.465648 D/acc1=0.873196 D/acc2=0.731104 D/sec= 12.375902 
saving early-stopped model (uar) at epoch 4
 :
2020-08-20 03:10:54,654 - main - INFO - [ 10] T/loss=0.073781 T/acc1=0.978863 T/acc2=0.000000 T/sec= 394.504868 D/loss=0.747309 D/acc1=0.868193 D/acc2=0.740413 D/sec= 12.368943 

==== Classification report (early_stopped-uar) ====

              precision    recall  f1-score   support

        かじ取り       0.89      0.87      0.88       299
         その他       0.81      0.71      0.76       300
        エンジン       0.88      0.91      0.89      1494
        乗車装置       0.90      0.95      0.92       290
        保安灯火       0.90      0.95      0.92       363
        制動装置       0.93      0.93      0.93       477
        動力伝達       0.91      0.91      0.91       814
     排ｶﾞｽ･騒音       0.93      0.85      0.89       122
        燃料装置       0.86      0.88      0.87       232
        緩衝装置       0.86      0.82      0.84       129
       装置その他       0.56      0.42      0.48        12
        走行装置       0.88      0.85      0.86       127
       車枠・車体       0.73      0.73      0.73       323
   電動機(モーター)       0.81      0.50      0.62        34
        電気装置       0.79      0.69      0.73       177
         非装置       0.00      0.00      0.00         4

    accuracy                           0.87      5197
   macro avg       0.79      0.75      0.76      5197
weighted avg       0.87      0.87      0.87      5197
```

- Learning Curve (train_bert.py on Google Colaboratory ([code](train_bert.ipynb)))

|rt-polarity|mlit| 
|---|---|
![](results/results_bert-rt/results_bert-rt.png)|![](results/results_bert-mlit/results_bert-mlit.png)

<img src="results/accuracy-rt.png"/> <img src="results/accuracy-mlit.png"/>

See also: [Chainer classification examples](/bert/classify) <br>
See also: [other classification experiments](/classify)

### Explaining text classifiers using LIME

```
pip install lime
```

- for rt-polarity datasets (subword, en)

```
python lime_bert-subword.py \
--test datasets/rt-polarity/04-test.txt \
--pretrained "bert-base-uncased" \
--model "results_bert-4-rt/early_stopped-uar.pth.tar" \
--batchsize 64 \
--topN 2 \
--epoch 10 \
--out results_lime-bert-subword-rt \
2>&1 | tee results/results_lime-bert-subword-rt.log

2020-08-21 07:49:27,136 - main - INFO - {
  "test": "datasets/rt-polarity/04-test.txt",
  "pretrained": "bert-base-uncased",
  "model": "results_bert-4-rt/early_stopped-uar.pth.tar",
  "batchsize": 64,
  "topN": 2,
  "out": "results_lime-bert-subword-rt"
}

Linear(in_features=768, out_features=2, bias=True)
2020-08-21 07:49:38,163 - load_data - INFO - Loading dataset ... done.
2020-08-21 07:49:38,164 - main - DEBUG - ['a visually flash ##y but narrative ##ly opaque and emotionally va ##pid exercise in style and my ##sti ##fication .', 'while the performances are often engaging , this loose collection of largely improvised numbers would probably have worked better as a one - hour tv documentary .', 'on a cutting room floor somewhere lies … footage that might have made no such thing a trench ##ant , ironic cultural satire instead of a frustrating mis ##fire .']
2020-08-21 07:49:38,164 - main - DEBUG - [0, 0, 0]
# test  X: 1066, y: 1066, class: 2

Enter document ID [0..1066]=> 0
Document id: 0
True class: 0
Probability (0) = 0.988464
Probability (1) = 0.011536

Explanation of document id 0 for class 0
('pid', 0.2638781076812105)
('opaque', 0.19493093394162633)
('va', 0.14111772448297713)
('and', -0.07000921762623179)
('emotionally', -0.05775203060488132)
('in', -0.05058837417097684)
('but', -0.041236925791098965)
('a', -0.04027007239259117)
('my', -0.0382126038056275)
('narrative', -0.03251564607900992)
('style', -0.029193711154148292)
('fication', 0.022344754395777006)
('y', -0.01620165847804197)
('exercise', 0.00865502179491645)
('ly', -0.008565875780031547)
('sti', 0.0059054230905909635)
('flash', 0.00462206699393289)
('visually', 0.0020202570469299666)

Explanation of document id 0 for class 1
('pid', -0.26387810905681125)
('opaque', -0.19493093216008053)
('va', -0.14111772756665797)
('and', 0.07000921555062066)
('emotionally', 0.05775203402314305)
('in', 0.05058837948515464)
('but', 0.04123692663095498)
('a', 0.04027007116764118)
('my', 0.03821260301878023)
('narrative', 0.03251564294832647)
('style', 0.029193713060257784)
('fication', -0.022344755084083213)
('y', 0.016201661409198386)
('exercise', -0.008655024132245425)
('ly', 0.008565875890614204)
('sti', -0.005905422756075881)
('flash', -0.004622067486400607)
('visually', -0.0020202582566745224)

Enter document ID [0..1066]=> 
```

- for mlit datasets (subword, ja)

```
python lime_bert-subword.py \
--test datasets/mlit/04-test.txt \
--pretrained "cl-tohoku/bert-base-japanese-whole-word-masking" \
--model "results_bert-4-mlt/early_stopped-uar.pth.tar" \
--batchsize 64 \
--topN 2 \
--out results_lime-bert-subword-mlit \
2>&1 | tee results/results_lime-bert-subword-mlit.log

2020-08-21 07:51:30,867 - main - INFO - {
  "test": "datasets/mlit/04-test.txt",
  "pretrained": "cl-tohoku/bert-base-japanese-whole-word-masking",
  "model": "results_bert-4-mlt/early_stopped-uar.pth.tar",
  "batchsize": 64,
  "topN": 2,
  "out": "results_lime-bert-subword-mlit"
}

Linear(in_features=768, out_features=16, bias=True)
2020-08-21 07:52:21,185 - load_data - INFO - Loading dataset ... done.
2020-08-21 07:52:21,186 - main - DEBUG - ['ダッシュ ##ボード が 溶け て ベ ##ト ##ベト し て いる 。', 'NO ##X センサー の 不良 により 、 エンジン 警告 灯 が 点 ##き っ ##ぱ ##なし に なっ た 。', '電動 オープン の ルー ##フ を 閉じる とき に 、 エラー ##メッセ ##ージ が 出 て 幌 が 閉 ##まら なく なっ た 。']
2020-08-21 07:52:21,187 - main - DEBUG - [6, 3, 6]
# test  X: 5197, y: 5197, class: 16

Enter document ID [0..5197]=> 0
Document id: 0
True class: 車枠・車体
Probability (車枠・車体) = 0.871476
Probability (乗車装置) = 0.092782

Explanation of document id 0 for class 車枠・車体
('ボード', 0.5319475699273315)
('いる', 0.16350428116925098)
('ベ', 0.1198598983893719)
('ダッシュ', -0.09467334673550995)
('て', 0.08459260780356471)
('が', -0.07742958854689783)
('し', -0.0444915403293281)
('ト', -0.04122271503574596)
('溶け', 0.04102362816418765)
('ベト', -0.028252250025748566)

Explanation of document id 0 for class 乗車装置
('ボード', -0.21920380921004942)
('ダッシュ', 0.14143110765333594)
('が', 0.07171206685385868)
('溶け', 0.04524682789971206)
('ベト', -0.036533317109689004)
('し', 0.024203641701843935)
('ベ', -0.02047974956688656)
('て', -0.011450973202305084)
('いる', 0.005466010772075001)
('ト', -0.003695830917293938)

Enter document ID [0..5197]=> 
```

|LIME output (rt-polarity, docid=0)|
|---|
|![](results/results_lime-bert-subword-rt/exp_show-docid_0.png)|

|LIME output (mlit, docid=0)|
|---|
|![](results/results_lime-bert-subword-mlt/exp_show-docid_0.png)|


- for rt-polarity datasets (subword, en)

```
python lime_bert-word.py \
--test datasets/rt-polarity/04-test.txt \
--pretrained "bert-base-uncased" \
--model "results_bert-4-rt/early_stopped-uar.pth.tar" \
--batchsize 64 \
--topN 2 \
--epoch 10 \
--out results_lime-bert-word-rt \
2>&1 | tee results/results_lime-bert-word-rt.log

2020-08-20 04:52:33,165 - main - INFO - {
  "test": "datasets/rt-polarity/04-test.txt",
  "pretrained": "bert-base-uncased",
  "model": "results_bert-rt/early_stopped-uar.pth.tar",
  "batchsize": 64,
  "topN": 2,
  "out": "results_lime-bert-word-rt"
}
Linear(in_features=768, out_features=2, bias=True)
2020-08-20 23:35:44,728 - load_data - INFO - Loading dataset ... done.
2020-08-20 23:35:44,729 - main - DEBUG - ['a visually flashy but narratively opaque and emotionally vapid exercise in style and mystification .', 'while the performances are often engaging , this loose collection of largely improvised numbers would probably have worked better as a one-hour tv documentary .', 'on a cutting room floor somewhere lies … footage that might have made no such thing a trenchant , ironic cultural satire instead of a frustrating misfire .']
2020-08-20 23:35:44,730 - main - DEBUG - [0, 0, 0]
# test  X: 1066, y: 1066, class: 2

Enter document ID [0..1066]=> 0
Document id: 0
True class: 0
Probability (0) = 0.988464
Probability (1) = 0.011536

Explanation of document id 0 for class 0
('vapid', 0.46518849978191323)
('opaque', 0.12733099672951603)
('but', -0.09200529605150436)
('and', -0.08672848652423029)
('a', -0.05784081703979195)
('in', -0.04808162436542701)
('emotionally', -0.0378265070070972)
('mystification', -0.022240540125408555)
('style', -0.020163725401766794)
('narratively', -0.013351168644917269)
('exercise', 0.01012951382868599)
('visually', -0.007953168914141733)
('flashy', 0.001322723187597232)

Explanation of document id 0 for class 1
('vapid', -0.4651884985572376)
('opaque', -0.12733099599497824)
('but', 0.09200529330199993)
('and', 0.08672848534497804)
('a', 0.05784081362391625)
('in', 0.04808162965169284)
('emotionally', 0.037826511182361806)
('mystification', 0.022240543049856277)
('style', 0.020163723382124098)
('narratively', 0.013351166942152949)
('exercise', -0.010129513850034378)
('visually', 0.007953165758780485)
('flashy', -0.0013227238000519308)

Enter document ID [0..1066]=> 
```

- for mlit datasets (subword, ja)

```
python lime_bert-word.py \
--test datasets/mlit/04-test.txt \
--pretrained "cl-tohoku/bert-base-japanese-whole-word-masking" \
--model "results_bert-4-mlt/early_stopped-uar.pth.tar" \
--batchsize 64 \
--topN 2 \
--out results_lime-bert-word-mlit \
2>&1 | tee results/results_lime-bert-word-mlit.log

2020-08-20 04:04:30,989 - main - INFO - {
  "test": "datasets/mlit/04-test.txt",
  "pretrained": "cl-tohoku/bert-base-japanese-whole-word-masking",
  "model": "results_bert-mlt/early_stopped-uar.pth.tar",
  "batchsize": 64,
  "topN": 2,
  "out": "results_lime-bert-word-mlit"
}
Linear(in_features=768, out_features=16, bias=True)
2020-08-20 04:04:41,343 - load_data - INFO - Loading dataset ... done.
2020-08-20 04:04:41,345 - main - DEBUG - ['ダッシュボード が 溶け て ベトベト し て いる 。', 'ＮＯ Ｘ センサー の 不良 により 、 エンジン 警告 灯 が 点き っぱなし に なっ た 。', '電動 オープン の ルーフ を 閉じる とき に 、 エラーメッセージ が 出 て 幌 が 閉まら なく なっ た 。']
2020-08-20 04:04:41,346 - main - DEBUG - [6, 3, 6]
# test  X: 5197, y: 5197, class: 16

Enter document ID [0..5197]=> 0
Document id: 0
True class: 車枠・車体
Probability (車枠・車体) = 0.871476
Probability (乗車装置) = 0.092782

Explanation of document id 0 for class 車枠・車体
('ダッシュボード', 0.47517304130341503)
('いる', 0.20711680367757068)
('て', 0.2039136100643194)
('ベトベト', 0.045422222680755575)
('溶け', 0.04019130026466325)
('し', -0.03657148397356947)
('が', 0.004224280842913079)

Explanation of document id 0 for class 乗車装置
('溶け', 0.023207162919567825)
('ダッシュボード', 0.022304686324583035)
('ベトベト', -0.01962039614972773)
('し', 0.014460980225919596)
('いる', -0.011011642279823207)
('て', 0.0012605664979242369)
('が', -2.5085145319950024e-05)

Enter document ID [0..5197]=> 
```

|LIME output (rt-polarity, docid=0)|
|---|
|![](results/results_lime-bert-word-rt/exp_show-docid_0.png)|

|LIME output (mlit, docid=0)|
|---|
|![](results/results_lime-bert-word-mlt/exp_show-docid_0.png)|
