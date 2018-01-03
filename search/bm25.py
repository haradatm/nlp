#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

import numpy as np
from scipy import sparse as sp

import re
import nltk
token_pattern = re.compile(r"(?u)\b\w\w+\b")
lemmatizer = nltk.WordNetLemmatizer()


def stopwords():
    symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')', '*', '--', '\\']
    stopwords = nltk.corpus.stopwords.words('english')
    return stopwords + symbols


def analyzer(text, stopwords=[]):
    words = []
    for word in token_pattern.findall(text):
        word = lemmatizer.lemmatize(word.lower())
        if word not in stopwords:
            words.append(word)
    return words


class Vectorizer:
    def __init__(self, **params):
        self.K1, self.B, self.delta = 2.0, 0.75, 1.0    # 定数
        self.idf = np.array([])                         # IDF
        self.vocab = {}                                 # 語彙とインデックスの辞書
        self.feature_names = []                         # 単語名のリスト
        self.avg_words = 0                              # ドキュメント内の単語数の平均
        self.stopwords = stopwords()                    # ストップワード
        self.analyzer = analyzer

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    def fit(self, documents):

        for document in documents:
            searched = {}
            words = self.analyzer(document, self.stopwords)
            self.avg_words += len(words)

            for word in words:
                # このドキュメント内で始めて出た単語
                if word not in searched:

                    searched[word] = True

                    # 他のドキュメントですでに出た単語
                    # if word in self.vocab and self.vocab[word] != 0:
                    if word in self.vocab:
                        self.idf[self.vocab[word]] += 1.0

                    # 初めて出現する単語
                    else:
                        self.feature_names.append(word)
                        self.vocab[word] = len(self.vocab)
                        self.idf = np.append(self.idf, [1.0])

        self.idf = np.log2((len(documents) - self.idf + 0.5) / (self.idf + .5))
        self.avg_words = self.avg_words / len(documents)

    def transform(self, documents):

        # 疎行列オブジェクト生成
        scores = sp.lil_matrix((len(documents), len(self.vocab)))

        for i, document in enumerate(documents):

            # TF
            tf = {}
            words = self.analyzer(document, self.stopwords)
            for word in words:
                if word in self.vocab:
                    if self.vocab[word] in tf:
                        tf[self.vocab[word]] += 1.0
                    else:
                        tf[self.vocab[word]] = 1.0

            # BM25 スコア
            for key in tf:
                scores[i, key] = self.idf[key] * (self.delta + (tf[key] * (self.K1 + 1.0)) / (tf[key] + self.K1 * (1.0 - self.B + self.B * (len(words) / self.avg_words))))

        return scores

    def get_feature_names(self):
        return self.feature_names
