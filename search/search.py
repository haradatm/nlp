#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, time, logging
import numpy as np
np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

from six.moves import xrange
import collections


def pp(obj):
    import pprint
    pp = pprint.PrettyPrinter(indent=1, width=160)
    print(pp.pformat(obj))


start_time = time.time()


def load_data(filename):

    logger.info('loading dataset: {}'.format(filename))
    sys.stdout.flush()

    text, labels = [], []

    for i, line in enumerate(open(filename, 'rU')):
        # if i == 0:
        #     continue

        try:
            line = line.strip()
        except UnicodeDecodeError:
            logger.error('invalid record: {}\n'.format(line))
            continue

        if line == '':
            continue

        line = line.replace(u'. . .', u'…')

        row = line.split(u'\t')
        if len(row) < 2:
            logger.error('invalid record: {}\n'.format(line))
            continue

        # label
        labels.append(row[0])

        # text
        text.append(row[1])

    return text, labels


def load_qrels(filename):
    qrels = collections.defaultdict(lambda: [])

    for i, line in enumerate(open(filename, 'rU')):
        # if i == 0:
        #     continue

        try:
            line = line.strip()
        except UnicodeDecodeError:
            logger.error('invalid record: {}\n'.format(line))
            continue

        if line == '':
            continue

        line = line.replace(u'. . .', u'…')

        row = line.split(u'\t')
        if len(row) < 2:
            logger.error('invalid record: {}\n'.format(line))
            continue

        qrels[row[0]].append((row[2], int(row[3])))

    return qrels


def mean_average_precision(queries, results, qrels, k):

    num_queries = len(queries)
    out = 0.

    for q_id, doc_ids in zip(queries, results):

        candidates = []
        for doc_id in doc_ids:
            b = False
            for d, r in qrels[q_id]:
                if d == doc_id:
                    candidates.append(r)
                    b = True
                    break
            if not b:
                candidates.append(0)

        num_correct = 0.
        precisions = []
        for j in xrange(min(k, len(candidates))):
            if int(candidates[j]) >= 1:
                num_correct += 1
                precisions.append(num_correct / (j + 1))

        avg_prec = 0.
        if len(precisions) > 0:
            avg_prec = sum(precisions) / len(precisions)
        out += avg_prec

        # logger.debug('map q_id={:}, avg_prec={:.6f}'.format(q_id, avg_prec))

    return out / float(num_queries)


def n_discount_cumulative_count(queries, results, qrels, k):

    num_queries = len(queries)
    out = 0.

    for q_id, doc_ids in zip(queries, results):

        candidates = []
        for doc_id in doc_ids:
            b = False
            for d, r in qrels[q_id]:
                if d == doc_id:
                    candidates.append(r)
                    b = True
                    break
            if not b:
                candidates.append(0)

        dcg = 0
        for j in xrange(min(k, len(candidates))):
            dcg += (2 ** candidates[j] - 1.) / np.log2(j + 2)

        candidates = []
        for d, r in sorted(qrels[q_id], key=lambda x: x[1], reverse=True):
            candidates.append(r)

        if len(doc_ids) > len(candidates):
            candidates.extend([0] * (len(doc_ids) - len(candidates)))

        ideal_dcg = 0
        for j in xrange(min(k, len(candidates))):
            ideal_dcg += (2 ** candidates[j] - 1.) / np.log2(j + 2)

        ndcg = 0.
        if ideal_dcg != 0. and dcg != 0.:
            ndcg = dcg / ideal_dcg
        out += ndcg

        # logger.debug('q_id={} ndcg={} dcg={} ideal_dcg={}'.format(q_id, ndcg, dcg, ideal_dcg))

    return out / float(num_queries)


from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('--docs',    default='', type=str, help='document data file (.txt)')
    parser.add_argument('--queries', default='', type=str, help='query data file (.txt)')
    parser.add_argument('--qrels',   default='', type=str, help='query relevance file (.qrel)')
    parser.add_argument('--type',    default='w2v', choices=['tfidf', 'bm25', 'w2v', 'fast'], help='type of vectorizer')
    parser.add_argument('--K', default=20, type=int, help='number of evaluations')
    args = parser.parse_args()

    # データの読み込み
    docs, doc_labels = load_data(args.docs)

    # データの読み込み
    queries, que_labels = load_data(args.queries)

    # 正解データの読み込み
    qrels = load_qrels(args.qrels)

    # confidence = 0.6

    # ベクトライザの定義
    import importlib
    module = importlib.import_module(args.type)
    vectorizer = module.Vectorizer()

    # データのベクトル化
    vector = vectorizer.fit_transform(docs)

    with open('results-{:}.txt'.format(args.type), 'w') as f:

        sim_doc_labels = []
        for idx, text in enumerate(queries):

            item = vectorizer.transform([text])

            # calculate cosine similarities
            similarities = cosine_similarity(item, vector)

            for i, j in enumerate(similarities.argsort()[0][::-1]):
                f.write("{}\tQ0\t{}\t{}\t{:.6f}\tSTANDARD\n".format(que_labels[idx], doc_labels[j], i, similarities[0][j]))

            # sort in descending order
            similarities_idx = similarities.argsort()[0][-1:-1001:-1]
            sim_doc_labels.append([doc_labels[x] for x in similarities_idx])

    acc_map = mean_average_precision(que_labels, sim_doc_labels, qrels, k=args.K)
    acc_ndcg = n_discount_cumulative_count(que_labels, sim_doc_labels, qrels, k=args.K)

    print('map@{:}={:.6f}\tndcg@{}={:.6f}'.format(args.K, acc_map, args.K, acc_ndcg))

logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
