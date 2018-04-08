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


def seeded_vector(w2v, seed_string):
    once = np.random.RandomState(hash(seed_string) & 0xffffffff)
    return (once.rand(w2v.vector_size) - 0.5) / w2v.vector_size


from gensim.models import KeyedVectors
# w2v_model_path = 'models/nfc_train_w2v.bin'
w2v_model_path = 'models/nfcorpus-all_w2v.bin'


def load_w2v_model(filename):

    logger.info('loading word2vec model: {}'.format(filename))
    sys.stdout.flush()

    w2v = KeyedVectors.load_word2vec_format(filename, binary=True)

    global UNK_VEC, PAD_VEC
    UNK_VEC = seeded_vector(w2v, UNK_TOKEN)
    PAD_VEC = seeded_vector(w2v, PAD_TOKEN)

    return w2v


class Vectorizer():
    def __init__(self, **params):
        self.w2v = load_w2v_model(w2v_model_path)
        self.feature_names = []
        self.analyzer = analyzer

    def fit_transform(self, documents):
        return self.transform(documents)

    def fit(self, documents):
        pass

    def transform(self, documents):
        results = []
        for document in documents:
            vec = []
            words = self.analyzer(document)
            for word in words:
                try:
                    vec.append(self.w2v[word])
                except KeyError:
                    logger.warning('unk: {}'.format(word))
                    vec.append(UNK_VEC)
            results.append(np.average(np.asarray(vec, dtype=np.float32), axis=0))
        return np.asarray(results)

    def get_feature_names(self):
        return self.feature_names
