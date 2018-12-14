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


def seeded_vector(bert, seed_string):
    once = np.random.RandomState(hash(seed_string) & 0xffffffff)
    return (once.rand(bert.vector_size) - 0.5) / bert.vector_size


import chainer
from bertlib.modeling import BertConfig, BertModel
from bertlib.tokenization import FullTokenizer
vocab_file = models/uncased_L-12_H-768_A-12/vocab.txt'
bert_config_file = 'models/uncased_L-12_H-768_A-12/bert_config.json'
init_checkpoint = 'models/uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz'


class Bert:
    def __init__(self, init_checkpoint, vocab_file, bert_config_file):
        bert_config = BertConfig.from_json_file(bert_config_file)
        bert = BertEmbedding(BertModel(config=bert_config))
        with np.load(init_checkpoint) as f:
            d = chainer.serializers.NpzDeserializer(f, path='', strict=True)
            d.load(bert)

        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

        self.vocab = self.tokenizer.vocab
        self.vectors = bert.bert.word_embeddings.W.data
        self.vocab_size, self.vector_size = self.vectors.shape

    def __getitem__(self, word):
        return self.vectors[self.vocab[word]]


class BertEmbedding(chainer.Chain):
    def __init__(self, bert):
        super(BertEmbedding, self).__init__()
        with self.init_scope():
            self.bert = bert

    def __call__(self, x1, x2, x3, ts):
        output_layer = self.bert.get_embedding_output(x1, x2, x3)
        return output_layer


def load_bert_model(init_checkpoint, vocab_file, bert_config_file):

    logger.info('loading bert model: {}'.format(init_checkpoint))
    sys.stdout.flush()

    bert = Bert(init_checkpoint, vocab_file, bert_config_file)

    global UNK_VEC, PAD_VEC
    UNK_VEC = seeded_vector(bert, UNK_TOKEN)
    PAD_VEC = seeded_vector(bert, PAD_TOKEN)

    return bert


class Vectorizer():
    def __init__(self, **params):
        self.bert = load_bert_model(init_checkpoint, vocab_file, bert_config_file)
        self.feature_names = []
        self.analyzer = self.bert.tokenizer.tokenize
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
                    # v = self.alpha * self.bert[word] / (self.alpha + self.tf[word] / self.count)
                    p = (self.tf[word] / self.count) if word in self.tf else (1. / self.count)
                    v = self.alpha * self.bert[word] / (self.alpha + p)
                    vec.append(v)
                except KeyError:
                    logger.warning('unk: {}'.format(word))
                    vec.append(UNK_VEC)
            results.append(np.average(np.asarray(vec, dtype=np.float32), axis=0))
        return np.asarray(results)

    def get_feature_names(self):
        return self.feature_names
