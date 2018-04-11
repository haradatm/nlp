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
```

***Results (dataset-rt)***


```
cat results/rt-ngram-lsvc-word.log
# train X: 9114, y: 9114, class: 2
# test  X: 1068, y: 1068, class: 2
             precision    recall  f1-score   support
avg / train       0.80      0.80      0.80       480
avg / test        0.78      0.78      0.78      1068
time spent:  59.76116991043091

cat results/rt-ngram-lsvc-word.log
# train X: 9114, y: 9114, class: 2
# test  X: 1068, y: 1068, class: 2
             precision    recall  f1-score   support
avg / train       0.78      0.77      0.77       480
avg / test        0.71      0.71      0.71      1068
time spent:  59.76116991043091

cat results/rt-ngram-xgb_-word.log
# train X: 9114, y: 9114, class: 2
# test  X: 1068, y: 1068, class: 2
             precision    recall  f1-score   support
avg / train       0.74      0.74      0.74       480
avg / test        0.68      0.68      0.67      1068
time spent:  59.76116991043091

```

***Results (dataset-mlit)***

```
cat results/mlit-ngram-lsvc-word.log
# train X: 39056, y: 38907, class: 16
# test  X: 4575, y: 4575, class: 16
             precision    recall  f1-score   support
avg / total       0.83      0.83      0.83      2056
avg / total       0.84      0.84      0.84      4575
time spent:  239.10893321037292

cat results/mlit-ngram-lsvc-word.log
# train X: 38907, y: 38907, class: 16
# test  X: 4575, y: 2048, 4575: 16
             precision    recall  f1-score   support
avg / total       0.80      0.80      0.80      2056
avg / total       0.82      0.81      0.81      4575
time spent:  4820.543439865112

```