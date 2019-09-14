#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#
# usege: python3.6
#

__version__ = '0.0.1'

import sys, time, logging
import numpy as np
np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def pp(obj):
    import pprint
    pp = pprint.PrettyPrinter(indent=1, width=160)
    print(pp.pformat(obj))


start_time = time.time()


import numpy as np
import collections
np.set_printoptions(precision=20)

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx

from nlplib3 import word_segmenter, sent_splitter, cleanse


def lexrank(sentences, continuous=True, sim_threshold=0.1, alpha=0.9):

    ranker_params = {'max_iter': 10000}
    ranker = nx.pagerank_scipy
    ranker_params['alpha'] = alpha

    graph = nx.DiGraph()

    # sentence -> tf
    sent_tf_list = []
    for sent in sentences:
        words = word_segmenter(sent, plus='名詞|動詞|形容詞|未知語', minus='助詞|助動詞|記号', type='base')
        if len(words) == 0:
            words = ['']
        tf = collections.Counter(words)
        sent_tf_list.append(tf)

    sent_vectorizer = DictVectorizer(sparse=True)
    sent_vecs = sent_vectorizer.fit_transform(sent_tf_list)

    # compute similarities between senteces
    sim_mat = 1 - pairwise_distances(sent_vecs, sent_vecs, metric='cosine')

    if continuous:
        linked_rows, linked_cols = np.where(sim_mat > 0)
    else:
        linked_rows, linked_cols = np.where(sim_mat >= sim_threshold)

    # create similarity graph
    graph.add_nodes_from(range(sent_vecs.shape[0]))
    for i, j in zip(linked_rows, linked_cols):
        if i == j:
            continue
        weight = sim_mat[i, j] if continuous else 1.0
        # graph.add_edge(i, j, {'weight': weight})
        graph.add_edge(i, j)

    scores = ranker(graph, **ranker_params)
    return scores, sim_mat


def select(text, sent_limit=None, char_limit=None, imp_require=None):

    sentences = list(sent_splitter(text))

    if len(sentences) > 1:
        scores, sim_mat = lexrank(sentences)

        if scores:
            sum_scores = sum(scores)
            acc_scores = 0.0
            indexes = set()
            num_sent, num_char = 0, 0

            for i in sorted(scores, key=lambda i: scores[i], reverse=True):
                num_sent += 1
                num_char += len(sentences[i])
                if sent_limit is not None and num_sent > sent_limit:
                    break
                if char_limit is not None and num_char > char_limit:
                    break
                if imp_require is not None and acc_scores / sum_scores >= imp_require:
                    break
                indexes.add(i)
                acc_scores += scores[i]
        else:
            indexes = []

    else:
        indexes = [0]

    if len(indexes) > 0:
        summary_sents = [sentences[i] for i in sorted(indexes)]
    else:
        summary_sents = []

    # pp({'sentences': sentences, 'scores': scores})

    return summary_sents


if __name__ == '__main__':
    text = '前線を伴った低気圧が、日本の東を東北東へ進んでいます。高気圧が大陸 から日本海に張り出しています。  ' \
           '【関東甲信地方】  関東甲信地方は、曇りの所が多く、雨の降っている所があります。   ' \
           '29日は、気圧の谷の影響により、曇りの所が多く、雨の降る所があるでしょう。   ' \
           '30日は、気圧の谷の影響により、曇りで雨の降る所もありますが、日中 は次第に晴れる所が多いでしょう。   ' \
           '関東近海では、29日は波が高い見込みです。30日はしける所があるでしょう。船舶は高波に注意してください。  ' \
           '【東京地方】 29日は、曇りで、雨の降る所があるでしょう。  ' \
           '30日は、曇り昼過ぎから時々晴れで、朝まで雨の降る所がある見込みで す'

    ret = select(cleanse(text), sent_limit=3, char_limit=None)
    pp(ret)

