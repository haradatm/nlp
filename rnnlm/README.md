# SER (Speech Emotion Recognition) example

### Description

This example code is a turn-based speech emotion recognition using three kinds of approaches, `Simple LSTM`, `NStep LSTM`, and `NStep BiLSTM`.

### Dependencies
- python 3.6
- chainer 3.4

In addition, please add the project folder to PYTHONPATH and `conca install` the following packages:
- `matplotlib`

### Usage ###

***Data***

  - Downlod [WAV files](http://voice-statistics.github.io/) and put them in the appropriate place.

```
mkdir -p datasets && cd datasets

# Normal
wget https://github.com/voice-statistics/voice-statistics.github.com/raw/master/assets/data/tsuchiya_normal.tar.gz	
wget https://github.com/voice-statistics/voice-statistics.github.com/raw/master/assets/data/uemura_normal.tar.gz
wget https://github.com/voice-statistics/voice-statistics.github.com/raw/master/assets/data/fujitou_normal.tar.gz

# Hppy
wget https://github.com/voice-statistics/voice-statistics.github.com/raw/master/assets/data/tsuchiya_happy.tar.gz
wget https://github.com/voice-statistics/voice-statistics.github.com/raw/master/assets/data/uemura_happy.tar.gz
wget https://github.com/voice-statistics/voice-statistics.github.com/raw/master/assets/data/fujitou_happy.tar.gz
 
# Angry
wget https://github.com/voice-statistics/voice-statistics.github.com/raw/master/assets/data/tsuchiya_angry.tar.gz
wget https://github.com/voice-statistics/voice-statistics.github.com/raw/master/assets/data/uemura_angry.tar.gz
wget https://github.com/voice-statistics/voice-statistics.github.com/raw/master/assets/data/fujitou_angry.tar.gz
```

  - Extract features using [OpenSMILE](https://audeering.com/technology/opensmile/) and put them in the appropriate place.

```
tar zxvf tsuchiya_normal.tar.gz
tar zxvf uemura_normal.tar.gz
tar zxvf fujitou_normal.tar.gz
tar zxvf tsuchiya_happy.tar.gz
tar zxvf uemura_happy.tar.gz
tar zxvf fujitou_happy.tar.gz
tar zxvf tsuchiya_angry.tar.gz
tar zxvf uemura_angry.tar.gz
tar zxvf fujitou_angry.tar.gz

mkdir -p wav
mv fujitou_angry/*.wav   wav/.
mv fujitou_happy/*.wav   wav/.
mv fujitou_normal/*.wav  wav/.
mv tsuchiya_angry/*.wav  wav/.
mv tsuchiya_happy/*.wav  wav/.
mv tsuchiya_normal/*.wav wav/.
mv uemura_angry/*.wav    wav/.
mv uemura_happy/*.wav    wav/.
mv uemura_normal/*.wav   wav/.

mkdir -p arff
for f in wav/*.wav; do /Data/haradatm/src/opensmile-2.3.0/SMILExtract -C /Data/haradatm/src/opensmile-2.3.0/config/IS09_emotion.conf -I $f -O arff/`basename $f .wav`.arff ; done
grep -E "^\'unknown" arff/fujitou*.arff  | tr "," "\t" | cut -f 2-385 > feature-fujitou.txt
grep -E "^\'unknown" arff/tsuchiya*.arff | tr "," "\t" | cut -f 2-385 > feature-tsuchiya.txt
grep -E "^\'unknown" arff/uemura*.arff   | tr "," "\t" | cut -f 2-385 > feature-uemura.txt
ls -1 arff/fujitou*.arff  | cut -d "_" -f 2 > class-fujitou.txt
ls -1 arff/tsuchiya*.arff | cut -d "_" -f 2 > class-tsuchiya.txt
ls -1 arff/uemura*.arff   | cut -d "_" -f 2 > class-uemura.txt
paste feature-fujitou.txt  class-fujitou.txt  > train-fujitou.txt
paste feature-tsuchiya.txt class-tsuchiya.txt > train-tsuchiya.txt
paste feature-uemura.txt   class-uemura.txt   > train-uemura.txt
```

  - Create pseudo data with 3 turn each continued and put them in the appropriate place.

```
python ../tools/sampling.py train-fujitou.txt  300 > rand-fujitou.txt
python ../tools/sampling.py train-tsuchiya.txt 300 > rand-tsuchiya.txt
python ../tools/sampling.py train-uemura.txt   300 > rand-uemura.txt
python ../tools/prepare.py rand-fujitou.txt  > 01-test.txt
python ../tools/prepare.py rand-tsuchiya.txt > 02-test.txt
python ../tools/prepare.py rand-uemura.txt   > 03-test.txt
cat 02-test.txt 03-test.txt > 01-train.txt
cat 03-test.txt 01-test.txt > 02-train.txt
cat 01-test.txt 02-test.txt > 03-train.txt
```

***Run and Evaluate***

```
python train_turn-mlp.py    --batchsize 100 --epoch 300           --unit 256 --train ../datasets/01-train.txt --test ../datasets/01-test.txt --out 01-turn-1 2>&1 | tee 01-turn-1.log
python train_turn-lstm.py   --batchsize 100 --epoch 300           --unit 256 --train ../datasets/01-train.txt --test ../datasets/01-test.txt --out 01-turn-2 2>&1 | tee 01-turn-2.log
python train_turn-nstep.py  --batchsize 100 --epoch 300 --layer 3 --unit 256 --train ../datasets/01-train.txt --test ../datasets/01-test.txt --out 01-turn-3 2>&1 | tee 01-turn-3.log
python train_turn-bilstm.py --batchsize 100 --epoch 300 --layer 1 --unit 256 --train ../datasets/01-train.txt --test ../datasets/01-test.txt --out 01-turn-4 2>&1 | tee 01-turn-4.log
python train_both-bilstm.py --batchsize 100 --epoch 300 --burnin 100 --layer_t 1 --unit_t 256 --layer_c 1 --unit_c 64 --train ../datasets/01-train.txt --test ../datasets/01-test.txt --out 01-both-1 2>&1 | tee 01-both-1.log
```

***Input***

- format
```
[#trun] [feature 1] ... [feature 384] [class of turn 1] [class of turn N] [class of ALL]
```

- 01-test.txt
```
3	1.793738e-02	6.840575e-05	1.786898e-02	･･･	-1.245339e-01	4.882097e+00	happy	happy	happy	happy
```


***Output***

- 01-turn-1.log (use **train_both-bilstm.py**)
```
2018-07-05 12:22:05,491 - main - INFO - [  1] T/loss=1.111592 T/acc1=0.363333 T/acc2=0.000000 T/sec= 0.385696 D/loss=1.060488 D/acc1=0.370000 D/acc2=0.280000 D/sec= 1.075764 lr=0.001000
 :
2018-07-05 12:31:59,461 - main - INFO - [300] T/loss=0.090317 T/acc1=0.980000 T/acc2=0.965000 T/sec= 0.724062 D/loss=0.650186 D/acc1=0.766667 D/acc2=0.790000 D/sec= 0.937508 lr=0.001000
2018-07-05 12:32:00,059 - main - INFO - time spent: 597.156162 sec
```

- 0[1-3]-turn-1.png (use **train_mlp.py**)

<img src="results/01-turn-1.png" width="262px" height="261px"/> <img src="results/02-turn-1.png" width="262px" height="261px"/> <img src="results/03-turn-1.png" width="262px" height="261px"/>

- 0[1-3]-turn-2.png (use **train_turn-lstm.py**)

<img src="results/01-turn-2.png" width="262px" height="261px"/> <img src="results/02-turn-2.png" width="262px" height="261px"/> <img src="results/03-turn-2.png" width="262px" height="261px"/>

- 0[1-3]-turn-3.png (use **train_turn-nstep.py**)

<img src="results/01-turn-3.png" width="262px" height="261px"/> <img src="results/02-turn-3.png" width="262px" height="261px"/> <img src="results/03-turn-3.png" width="262px" height="261px"/>

- 0[1-3]-turn-4.png (use **train_turn-bilstm.py**)

<img src="results/01-turn-4.png" width="262px" height="261px"/> <img src="results/02-turn-4.png" width="262px" height="261px"/> <img src="results/03-turn-4.png" width="262px" height="261px"/>

- 0[1-3]-both-1.png (use **train_both-bilstm.py**)

<img src="results/01-both-1.png" width="262px" height="261px"/> <img src="results/02-both-1.png" width="262px" height="261px"/> <img src="results/03-both-1.png" width="262px" height="261px"/>
