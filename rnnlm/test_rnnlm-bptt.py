#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Sample script of recurrent neural network language model.

    usage: python3.6 train_rnnlm.py --gpu -1 --epoch 200 --batchsize 100 --unit 300 --train datasets/soseki/neko-word-train.txt --test datasets/soseki/neko-word-test.txt --w2v datasets/soseki/neko_w2v.bin --out model-neko
    usage: python3.6  test_rnnlm.py --gpu -1 --model "model-neko/final.model" --text "吾輩 は 猫 で ある 。"
"""

__version__ = '0.0.1'

import sys, os, time, logging, json, math
import numpy as np
np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
# handler = logging.FileHandler(filename="log.txt")
handler.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'))
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

def pp(obj):
    import pprint
    pp = pprint.PrettyPrinter(indent=1, width=160)
    logger.info(pp.pformat(obj))


start_time = time.time()


import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import pickle


# UNK_ID = 0
# EOS_ID = 1
UNK_TOKEN = '<unk>'
EOS_TOKEN = '</s>'


# Definition of a recurrent net for language modeling
class RNNLM(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        super(RNNLM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.l1 = L.LSTM(n_units, n_units)
            self.l2 = L.LSTM(n_units, n_units)
            self.l3 = L.Linear(n_units, n_vocab)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, x, t):
        y = self.forward(x)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    # 1ステップ前方処理関数 (学習データ,状態を与える)
    def forward(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0))
        h2 = self.l2(F.dropout(h1))
        y = self.l3(F.dropout(h2))
        return y

    def predict(self, x):
        y = self.forward(x)
        return F.softmax(y)

    # 状態の初期化 (初期状態を現在の状態にセット)
    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def set_word_embedding(self, data):
        self.embed.W.data = data


def main():
    global xp

    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: RNNLM')
    parser.add_argument('--model', '-m', type=str, default='model/final.model', help='model data, saved by train.py')
    parser.add_argument('--text', '-t', type=str, default='吾 輩 は 猫 で あ る', help='base text data, used for text generation')
    parser.add_argument('--unit', '-u', type=int, default=200, help='Number of LSTM units in each layer')
    parser.add_argument('--sample', type=int, default=1, help='negative value indicates NOT use random choice')
    parser.add_argument('--length', type=int, default=2000, help='length of the generated text')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    # print(json.dumps(args.__dict__, indent=2))

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np
    xp.random.seed(123)

    vocab = pickle.load(open(os.path.join(os.path.dirname(args.model), 'vocab.bin'), 'rb'))
    token2id = {}
    for i, token in enumerate(vocab):
        token2id[token] = i

    logger.info('Number of units: {}'.format(args.unit))
    logger.info('Vocabulary size: {}'.format(len(vocab)))

    # Recurrent neural net languabe model
    model = RNNLM(len(vocab), args.unit)
    chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        model.reset_state()

        for token in args.text.strip().split(' '):
            if token == EOS_TOKEN:
                sys.stdout.write('\n')
                sys.stdout.flush()
            else:
                sys.stdout.write(token)
                sys.stdout.flush()
            prev_word = model.predict(xp.array([token2id[token]], dtype=np.int32))

        for i in range(args.length):
            if args.sample > 0:
                next_prob = cuda.to_cpu(prev_word.data)[0].astype(np.float64)
                next_prob /= np.sum(next_prob)
                idx = np.random.choice(range(len(next_prob)), p=next_prob)
            else:
                idx = np.argmax(cuda.to_cpu(prev_word.data))

            if vocab[idx] == EOS_TOKEN:
                sys.stdout.write('\n')
                sys.stdout.flush()
            else:
                sys.stdout.write(vocab[idx])
                sys.stdout.flush()
            prev_word = model.predict(xp.array([idx], dtype=np.int32))

        sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
