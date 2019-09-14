# Python example code to write for Text Classification research and evaluation

### Requirements

- Python=3.6
- nltk==3.2

### Description

This is example codes to write for Information Retrieval research and evaluation.

***Run and Evaluate***

```
python scikit-ngram.py --type lsvc --analyzer word --train dataset-rt/01-train.txt --test dataset-rt/01-test.txt 2>&1 | tee results/rt-ngram-lsvc-word.log
python scikit-ngram.py --type lgb  --analyzer word --train dataset-rt/01-train.txt --test dataset-rt/01-test.txt 2>&1 | tee results/rt-ngram-lgb_-word.log
python scikit-ngram.py --type xgb  --analyzer word --train dataset-rt/01-train.txt --test dataset-rt/01-test.txt 2>&1 | tee results/rt-ngram-xgb_-word.log
python scikit-ngram.py --type lsvc --analyzer word --train dataset-mlit/01-train-wakachi.txt --test dataset-mlit/01-test-wakachi.txt 2>&1 | tee results/mlit-ngram-lsvc-word.log
python scikit-ngram.py --type lgb  --analyzer word --train dataset-mlit/01-train-wakachi.txt --test dataset-mlit/01-test-wakachi.txt 2>&1 | tee results/mlit-ngram-lgb_-word.log
python scikit-ngram.py --type xgb  --analyzer word --train dataset-mlit/01-train-wakachi.txt --test dataset-mlit/01-test-wakachi.txt 2>&1 | tee results/mlit-ngram-xgb_-word.log
```

***Data***

  - For English dataset: Downlod [sentence polarity dataset v1.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz) and put them in the appropriate place.
  - For Japanese dataset: Collect from [Ministry of Land, Infrastructure, Transport and Tourism](http://carinf.mlit.go.jp/jidosha/carinf/opn/index.html) and put them in the appropriate place.


***Results (dataset-rt)***


```
grep -E "train X|test  X|f1-score|avg / total|time spent" results/rt-ngram-lsvc-word.log
# train X: 9114, y: 9114, class: 2
# test  X: 1068, y: 1068, class: 2
             precision    recall  f1-score   support
dev  / total       0.80      0.80      0.80       480
test / total       0.80      0.80      0.80      1068
time spent:  30.62774395942688

grep -E "train X|test  X|f1-score|avg / total|time spent" results/rt-ngram-lgb_-word.log
# train X: 9114, y: 9114, class: 2
# test  X: 1068, y: 1068, class: 2
             precision    recall  f1-score   support
dev  / total       0.75      0.75      0.75       480
test / total       0.75      0.75      0.75      1068
time spent:  137.2329158782959

grep -E "train X|test  X|f1-score|avg / total|time spent" results/rt-ngram-xgb_-word.log
# train X: 9114, y: 9114, class: 2
# test  X: 1068, y: 1068, class: 2
             precision    recall  f1-score   support
dev  / total       0.66      0.66      0.66       480
test / total       0.69      0.69      0.69      1068
time spent:  211.02406811714172

grep -E "train X|test  X|f1-score|avg / total|time spent" results/rt-ngram-lsvc-word-svd.log
# train X: 9114, y: 9114, class: 2
# test  X: 1068, y: 1068, class: 2
             precision    recall  f1-score   support
dev  / total       0.76      0.76      0.76       480
test / total       0.74      0.74      0.73      1068
time spent:  729.1524150371552
```

***Results (dataset-mlit, word)***

```
# train X: 38902, y: 38902, class: 16
# test  X: 4557, y: 4557, class: 16
             precision    recall  f1-score   support
dev  / total       0.83      0.83      0.83      2048
test / total       0.83      0.83      0.83      4557
time spent:  138.16021704673767

grep -E "train X|test  X|f1-score|avg / total|time spent" results/mlit-ngram-lgb_-word.log
# train X: 38902, y: 38902, class: 16
# test  X: 4557, y: 4557, class: 16
             precision    recall  f1-score   support
dev  / total       0.81      0.81      0.81      2048
test / total       0.81      0.81      0.81      4557
time spent:  3647.5329928398132

grep -E "train X|test  X|f1-score|avg / total|time spent" results/mlit-ngram-xgb_-word.log
# train X: 38902, y: 38902, class: 16
# test  X: 4557, y: 4557, class: 16
             precision    recall  f1-score   support
dev  / total       0.80      0.78      0.78      2048
test / total       0.77      0.76      0.75      4557
time spent:  4461.006906032562

```