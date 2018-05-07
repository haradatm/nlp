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

import sys
import numpy as np

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
UNK_VEC = None
PAD_VEC = None


def analyzer(text):
    words = []
    for word in text.split(' '):
        if word:
            words.append(word)
    return words


def seeded_vector(fastText, seed_string):
    once = np.random.RandomState(hash(seed_string) & 0xffffffff)
    return (once.rand(fastText.vector_size) - 0.5) / fastText.vector_size


from gensim.models import KeyedVectors
# w2v_model_path = 'models/nfc_train_w2v.bin'
# w2v_model_path = 'models/nfcorpus-all_w2v.bin'
fastText_model_path = 'models/nfcorpus-all_fast.vec'


def load_fastText_model(filename):

    logger.info('loading fastText model: {}'.format(filename))
    sys.stdout.flush()

    # w2v = KeyedVectors.load_word2vec_format(filename, binary=True)
    fastText = KeyedVectors.load_word2vec_format(filename, binary=False)

    global UNK_VEC, PAD_VEC
    UNK_VEC = seeded_vector(fastText, UNK_TOKEN)
    PAD_VEC = seeded_vector(fastText, PAD_TOKEN)

    return fastText


class Vectorizer():
    def __init__(self, **params):
        self.fastText = load_fastText_model(fastText_model_path)
        self.feature_names = []
        self.analyzer = analyzer
        self.alpha, self.beta = 0.001, 4.0
        self.tf = {}
        self.count = 0

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    def fit(self, documents):
        self.count = 0
        for document in documents:
            words = self.analyzer(document)
            for word in words:
                if word in self.tf:
                    self.tf[word] += 1.0
                else:
                    self.tf[word] = 1.0
            self.count += len(words)

    def transform(self, documents):
        results = []
        for document in documents:
            vec = []
            words = self.analyzer(document)
            for word in words:
                try:
                    # v = self.alpha * self.w2v[word] / (self.alpha + self.tf[word] / self.count)
                    p = (self.tf[word] / self.count) if word in self.tf else (1. / self.count)
                    v = self.alpha * self.fastText[word] / (self.alpha + p)
                    vec.append(v)
                except KeyError:
                    logger.warning('unk: {}'.format(word))
                    vec.append(UNK_VEC)
            results.append(np.average(np.asarray(vec, dtype=np.float32), axis=0))
        return np.asarray(results)

    def get_feature_names(self):
        return self.feature_names
