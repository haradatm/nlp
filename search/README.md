# Python example code to write for Information Retrieval research and evaluation

## Requirements

- Python=3.6
- nltk==3.2

## Description

This is example codes to write for Information Retrieval research and evaluation.

```
python3 search.py --type tfidf --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 1000  2>&1 | tee log_tfidf-1000-2-1-0.txt
python3 search.py --type bm25  --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 1000  2>&1 | tee log_bm25_-1000-2-1-0.txt
python3 search.py --type w2v   --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 1000  2>&1 | tee log_w2v__-1000-2-1-0.txt
```
