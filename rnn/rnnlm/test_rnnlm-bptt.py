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
# UNK_TOKEN = '<unk>'
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


def make_candidates(candidates, beam_width, sample=True):
    next_candidates = []

    for model, token_ids, likelihood in candidates:
        y = model.predict(xp.array([token_ids[-1]], dtype=np.int32))
        next_prob = cuda.to_cpu(y.data)[0].astype(np.float64)
        next_prob /= np.sum(next_prob)
        next_likelihood = np.log(next_prob)

        # 上位 beam_width 個の枝を残す
        if sample:
            order = np.random.choice(range(len(next_prob)), beam_width, p=next_prob)
        else:
            order = np.argsort(next_prob)[::-1][:beam_width]

        for i in order:
            ll = (likelihood * len(token_ids) + next_likelihood[i]) / (len(token_ids) + 1)
            next_candidates.append((model.copy(), token_ids + [i], ll))

        # 全ての枝の中から対数尤度の上位 beam_width 個を残す
        candidates = sorted(next_candidates, key=lambda x: -x[2])[:beam_width]

    return candidates


def main():
    global xp

    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: BPTT RNNLM')
    parser.add_argument('--model', '-m', type=str, default='model/final.model', help='model data, saved by train.py')
    parser.add_argument('--text', '-t', type=str, default='吾 輩 は 猫 で あ る', help='base text data, used for text generation')
    parser.add_argument('--unit', '-u', type=int, default=200, help='number of LSTM units in each layer')
    parser.add_argument('--sample', action='store_true', help='use random choice')
    parser.add_argument('--beam', type=int, default=5, help='number of beam width')
    parser.add_argument('--length', type=int, default=50, help='length of the generated text')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    # print(json.dumps(args.__dict__, indent=2))

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np
    xp.random.seed(123)

    vocab = pickle.load(open(os.path.join(os.path.dirname(args.model), 'vocab.bin'), 'rb'))
    token2id = {v: k for k, v in enumerate(vocab)}

    logger.info('Number of units: {}'.format(args.unit))
    logger.info('Vocabulary size: {}'.format(len(vocab)))

    # Recurrent neural net languabe model
    model = RNNLM(len(vocab), args.unit)
    chainer.serializers.load_npz(args.model, model)

    beam_width = args.beam

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        model.reset_state()

        prime_text = args.text.strip().split(' ')
        token_ids = [token2id[x] for x in prime_text]

        for token in prime_text:
            _ = model.predict(xp.array([token2id[token]], dtype=np.int32))

        candidates = [(model.copy(), token_ids, 0)]

        for i in range(args.length):
            candidates = make_candidates(candidates, beam_width, sample=args.sample)

        for x in candidates[0][1][0:]:
            if x != token2id[EOS_TOKEN]:
                print(vocab[x], end='')
                sys.stdout.flush()
            else:
                print()
                sys.stdout.flush()


if __name__ == '__main__':
    main()
