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


vocab_file = '/Data/haradatm/DATA/Google-BERT/uncased_L-12_H-768_A-12/vocab.txt'
bert_config_file = '/Data/haradatm/DATA/Google-BERT/uncased_L-12_H-768_A-12/bert_config.json'
init_checkpoint = '/Data/haradatm/DATA/Google-BERT/uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz'

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle as skshuffle
from bertlib.modeling import BertConfig, BertModel
from bertlib.tokenization import FullTokenizer


class BertEmbedding(chainer.Chain):

    def __init__(self, bert):
        super(BertEmbedding, self).__init__()
        with self.init_scope():
            self.bert = bert

    def __call__(self, x1, x2, x3, ts):
        output_layer = self.bert.get_embedding_output(x1, x2, x3)
        return output_layer


def load_bert_model():

    logger.info('loading bert pre-trained model: {}'.format(init_checkpoint))
    sys.stdout.flush()

    bert_config = BertConfig.from_json_file(bert_config_file)
    bert = BertEmbedding(BertModel(config=bert_config))
    with np.load(init_checkpoint) as f:
        d = chainer.serializers.NpzDeserializer(f, path='', strict=True)
        d.load(bert)

    # if gpu >= 0:
    #     bert.to_gpu()

    return bert


# def batch_iter(data, batch_size):
#     batch = []
#     for line in data:
#         batch.append(line)
#         if len(batch) == batch_size:
#             yield tuple(list(x) for x in zip(*batch))
#             batch = []
#     if batch:
#         yield tuple(list(x) for x in zip(*batch))
#
#
# def to_device(device, x):
#     if device is None:
#         return x
#     elif device < 0:
#         return cuda.to_cpu(x)
#     else:
#         return cuda.to_gpu(x, device)


class Vectorizer():
    def __init__(self, **params):
        self.bert = load_bert_model().bert
        self.feature_names = []
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.vocab = self.tokenizer.vocab
        self.count = 0

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    def fit(self, documents):
        self.count = len(documents)

    def transform(self, documents, batch_size=1000):
        max_position_embeddings = self.bert.position_embeddings.W.shape[0]
        ndims = self.bert.word_embeddings.W.shape[1]
        results = np.zeros((len(documents), ndims), dtype='f')

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            start = 0

            for document in documents:
                tokens_a = self.tokenizer.tokenize(document)
                tokens = ["[CLS]"]
                segment_ids = [0]

                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                x1 = np.array([input_ids[:max_position_embeddings]], 'i')
                x2 = np.array([input_mask[:max_position_embeddings]], 'f')
                x3 = np.array([segment_ids[:max_position_embeddings]], 'i')

                embedding = self.bert.get_embedding_output(x1, x2, x3)

                end = start + embedding.shape[0]
                results[start:end] = F.sum(embedding, axis=1).data / embedding.shape[1]
                start = end

        return results

    def get_feature_names(self):
        return self.feature_names
