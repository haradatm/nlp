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

import sys, struct
import numpy as np

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
UNK_VEC = None
PAD_VEC = None


class Word2Vec:
    def __init__(self, path):

        with open(path, 'rb') as f:
            self.vocab, self.index2word = {}, {}
            self.vocab_size, self.vector_size = map(int, f.readline().split())
            self.vectors = np.empty((self.vocab_size, self.vector_size), dtype=np.float32)

            for i in range(self.vocab_size):
                chs = b''
                while True:
                    ch = struct.unpack('c', f.read(1))[0]
                    if ch == b' ':
                        break
                    chs += ch

                word = chs.decode('utf-8')
                try:
                    self.vocab[word] = i
                    self.index2word[i] = word

                except RuntimeError:
                    logging.error('Error unicode(): %s', word)
                    self.vocab[word] = i
                    self.index2word[i] = word

                self.vectors[i] = np.zeros(self.vector_size)
                for j in range(self.vector_size):
                    self.vectors[i][j] = struct.unpack('f', f.read(struct.calcsize('f')))[0]

                # 改行を strip する
                assert f.read(1) == b'\n'

    def __getitem__(self, word):
        return self.vectors[self.vocab[word]]


def analyzer(text):
    words = []
    for word in text.split(' '):
        if word:
            words.append(word)
    return words


def seeded_vector(w2v, seed_string):
    once = np.random.RandomState(hash(seed_string) & 0xffffffff)
    return (once.rand(w2v.vector_size) - 0.5) / w2v.vector_size


# w2v_model_path = 'models/nfc_train_w2v.bin'
w2v_model_path = 'models/nfcorpus-all_w2v.bin'


def load_w2v_model(filename):

    logger.info('loading word2vec model: {}'.format(filename))
    sys.stdout.flush()

    # from gensim.models import KeyedVectors
    # w2v = KeyedVectors.load_word2vec_format(filename, binary=True)
    w2v = Word2Vec(filename)

    global UNK_VEC, PAD_VEC
    UNK_VEC = seeded_vector(w2v, UNK_TOKEN)
    PAD_VEC = seeded_vector(w2v, PAD_TOKEN)

    return w2v


class Vectorizer():
    def __init__(self, **params):
        self.w2v = load_w2v_model(w2v_model_path)
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
                    v = self.alpha * self.w2v[word] / (self.alpha + p)
                    vec.append(v)
                except KeyError:
                    logger.warning('unk: {}'.format(word))
                    vec.append(UNK_VEC)
            results.append(np.average(np.asarray(vec, dtype=np.float32), axis=0))
        return np.asarray(results)

    def get_feature_names(self):
        return self.feature_names
