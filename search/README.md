# Python example code to write for Information Retrieval research and evaluation

### Requirements

- Python=3.6
- nltk==3.2

### Description

This is example codes to write for Information Retrieval research and evaluation.

***Run***

```
python3 search.py --type tfidf --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee log_tfidf-100-2-1-0.txt
python3 search.py --type bm25  --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee log_bm25_-100-2-1-0.txt
python3 search.py --type w2v   --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee log_w2v__-100-2-1-0.txt
python3 search_hybrid.py       --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee log_hybrid-100-2-1-0.txt
```

***Evaluate***

Download [trec_eval](https://trec.nist.gov/trec_eval/trec_eval_latest.tar.gz) evaluation tool and put them in the appropriate place.

```
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results-bm25.txt  | grep ndcg > ndcg-bm25.txt 
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results-tfidf.txt | grep ndcg > ndcg-tfidf.txt
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results-w2v.txt   | grep ndcg > ndcg-w2v.txt  
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results-hybrid.txt | grep ndcg > ndcg-hybrid.txt 
```

***NDCG Score***

```
grep 100 ndcg-*.txt

ndcg-bm25.txt:  ndcg_cut_100          	all	0.2773
ndcg-tfidf.txt: ndcg_cut_100          	all	0.2668
ndcg-w2v.txt:   ndcg_cut_100          	all	0.2104
ndcg-hybrid.txt:ndcg_cut_100          	all	0.2917

ndcg-bm25.txt:  ndcg_cut_1000         	all	0.3907
ndcg-tfidf.txt: ndcg_cut_1000         	all	0.3797
ndcg-w2v.txt:   ndcg_cut_1000         	all	0.3284
ndcg-hybrid.txt:ndcg_cut_1000         	all	0.4016
```