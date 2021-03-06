# Python example code to write for Information Retrieval research and evaluation

### Requirements

- Python=3.6
- nltk==3.2

### Description

This is example codes to write for Information Retrieval research and evaluation.

***Run***

```
python3 search.py        --type tfidf              --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-tfidf-100-2-1-0.txt
python3 search.py        --type bm25 --bm_type qtf --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-bm25_qtf-100-2-1-0.txt
python3 search.py        --type bm25 --bm_type qbm --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-bm25_qbm-100-2-1-0.txt
python3 search.py        --type w2v                --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-w2v.txt-100-2-1-0.txt
python3 search.py        --type fast               --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-fast.txt-100-2-1-0.txt
python3 search.py        --type bert               --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-bert.txt-100-2-1-0.txt
python3 search_hybrid.py --type w2v  --bm_type qtf --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-hybrid_qtf_w2v.txt-100-2-1-0.txt
python3 search_hybrid.py --type w2v  --bm_type qbm --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-hybrid_qbm_w2v.txt-100-2-1-0.txt
python3 search_hybrid.py --type fast --bm_type qtf --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-hybrid_qtf_fast.txt-100-2-1-0.txt
python3 search_hybrid.py --type fast --bm_type qbm --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-hybrid_qbm_fast.txt-100-2-1-0.txt
python3 search_hybrid.py --type bert --bm_type qtf --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-hybrid_qtf_bert.txt-100-2-1-0.txt
python3 search_hybrid.py --type bert --bm_type qbm --docs datasets/nfcorpus/test.docs --queries datasets/nfcorpus/test.nontopic-titles.queries --qrels datasets/nfcorpus/test.2-1-0.qrel --K 100  2>&1 | tee results/log-hybrid_qbm_bert.txt-100-2-1-0.txt
```

***Evaluate***

Download [trec_eval](https://trec.nist.gov/trec_eval/trec_eval_latest.tar.gz) evaluation tool and put them in the appropriate place.

```
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-tfidf.txt           | grep ndcg > results/ndcg-tfidf.txt          
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-bm25_qtf.txt        | grep ndcg > results/ndcg-bm25_qtf.txt       
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-bm25_qbm.txt        | grep ndcg > results/ndcg-bm25_qbm.txt       
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-w2v.txt             | grep ndcg > results/ndcg-w2v.txt            
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-fast.txt            | grep ndcg > results/ndcg-fast.txt           
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-hybrid_qtf_w2v.txt  | grep ndcg > results/ndcg-hybrid_qtf_w2v.txt 
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-hybrid_qbm_w2v.txt  | grep ndcg > results/ndcg-hybrid_qbm_w2v.txt 
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-hybrid_qtf_fast.txt | grep ndcg > results/ndcg-hybrid_qtf_fast.txt
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-hybrid_qbm_fast.txt | grep ndcg > results/ndcg-hybrid_qbm_fast.txt
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-bert.txt            | grep ndcg > results/ndcg-bert.txt            
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-hybrid_qtf_bert.txt | grep ndcg > results/ndcg-hybrid_qtf_bert.txt 
trec_eval -m all_trec datasets/nfcorpus/test.2-1-0.qrel results/results-hybrid_qbm_bert.txt | grep ndcg > results/ndcg-hybrid_qbm_bert.txt 
```

***NDCG Score***

```
grep "_20 " ndcg-*.txt

ndcg-tfidf.txt:ndcg_cut_20                     	all	0.2475	# TF-IDF
ndcg-bm25_qbm.txt:ndcg_cut_20                  	all	0.2557	# Okapi BM25 (w/bm25 query vector)
ndcg-bm25_qtf.txt:ndcg_cut_20                  	all	0.2657	# Okapi BM25 (w/tf query vector)
ndcg-w2v.txt:ndcg_cut_20                       	all	0.1752	# word2vec
ndcg-fast.txt:ndcg_cut_20                      	all	0.1687	# fastText
ndcg-bert.txt:ndcg_cut_20                      	all	0.1860	# Bert
ndcg-hybrid_qbm_w2v.txt:ndcg_cut_20            	all	0.2676	# Okapi BM25 (w/bm25 query vector) + word2vec
ndcg-hybrid_qtf_w2v.txt:ndcg_cut_20            	all	0.2650	# Okapi BM25 (w/tf query vector)   + word2vec
ndcg-hybrid_qbm_fast.txt:ndcg_cut_20           	all	0.2706	# Okapi BM25 (w/bm25 query vector) + fastText
ndcg-hybrid_qtf_fast.txt:ndcg_cut_20           	all	0.2719	# Okapi BM25 (w/tf query vector)   + fastText
ndcg-hybrid_qbm_bert.txt:ndcg_cut_20           	all	0.2603	# Okapi BM25 (w/bm25 query vector) + Bert
ndcg-hybrid_qtf_bert.txt:ndcg_cut_20           	all	0.2700	# Okapi BM25 (w/tf query vector)   + Bert
```

<img src="results/ndcg-020.png"/>

```
grep "_200 " ndcg-*.txt

ndcg-tfidf.txt:ndcg_cut_200                    	all	0.2897
ndcg-bm25_qbm.txt:ndcg_cut_200                 	all	0.2989
ndcg-bm25_qtf.txt:ndcg_cut_200                 	all	0.3055
ndcg-w2v.txt:ndcg_cut_200                      	all	0.2344
ndcg-fast.txt:ndcg_cut_200                     	all	0.2213
ndcg-bert.txt:ndcg_cut_200                     	all	0.2116
ndcg-hybrid_qbm_w2v.txt:ndcg_cut_200           	all	0.3157
ndcg-hybrid_qtf_w2v.txt:ndcg_cut_200           	all	0.3194
ndcg-hybrid_qtf_fast.txt:ndcg_cut_200          	all	0.3196
ndcg-hybrid_qbm_fast.txt:ndcg_cut_200          	all	0.3174
ndcg-hybrid_qbm_bert.txt:ndcg_cut_200          	all	0.3004
ndcg-hybrid_qtf_bert.txt:ndcg_cut_200          	all	0.3096
```

<img src="results/ndcg-200.png"/>
