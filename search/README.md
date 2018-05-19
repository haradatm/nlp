# Python example code to write for Information Retrieval research and evaluation

### Requirements

- Python=3.6
- nltk==3.2

### Description

This is example codes to write for Information Retrieval research and evaluation.

***Run***

```
python3 search.py        --type tfidf --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log_tfidf-100-2-1-0.txt
python3 search.py        --type bm25  --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log_bm25_-100-2-1-0.txt
python3 search.py        --type w2v   --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log_w2v__-100-2-1-0.txt
python3 search.py        --type fast  --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log_fast_-100-2-1-0.txt
python3 search_hybrid.py --type w2v   --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log_hybrid_w2v_-100-2-1-0.txt
python3 search_hybrid.py --type fast  --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log_hybrid_fast-100-2-1-0.txt
```

***Evaluate***

Download [trec_eval](https://trec.nist.gov/trec_eval/trec_eval_latest.tar.gz) evaluation tool and put them in the appropriate place.

```
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-bm25.txt        | grep ndcg > results/ndcg-bm25.txt       
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-tfidf.txt       | grep ndcg > results/ndcg-tfidf.txt      
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-w2v.txt         | grep ndcg > results/ndcg-w2v.txt        
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-fast.txt        | grep ndcg > results/ndcg-fast.txt       
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-hybrid_w2v.txt  | grep ndcg > results/ndcg-hybrid_w2v.txt 
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-hybrid_fast.txt | grep ndcg > results/ndcg-hybrid_fast.txt
```

***NDCG Score***

```
grep "_20" ndcg-*.txt

ndcg-bm25.txt:ndcg_cut_20          all 0.2557		# Okapi BM25
ndcg-tfidf.txt:ndcg_cut_20         all 0.2475		# TF-IDF
ndcg-w2v.txt:ndcg_cut_20           all 0.1753		# word2vec
ndcg-fast.txt:ndcg_cut_20          all 0.1686		# fastText
ndcg-hybrid_w2v.txt:ndcg_cut_20    all 0.2673		# Okapi BM25 + word2vec
ndcg-hybrid_fast.txt:ndcg_cut_20   all 0.2704		# Okapi BM25 + fastText

ndcg-bm25.txt:ndcg_cut_200         all 0.3076
ndcg-tfidf.txt:ndcg_cut_200        all 0.2943
ndcg-w2v.txt:ndcg_cut_200          all 0.2321
ndcg-fast.txt:ndcg_cut_200         all 0.2208
ndcg-hybrid_w2v.txt:ndcg_cut_200   all 0.3167
ndcg-hybrid_fast.txt:ndcg_cut_200  all 0.3179
```