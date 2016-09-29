#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Sample script of Sequence-to-Sequence model.

    usage: python2.7 train_seq2seq.py ogura_wakachi.txt --w2vmodel "ogura_w2v.bin" --gpu -1 --epoch 250 --unit 50 --batchsize 2 --gradclip 5.0 --output model-ogura --initmodel "model-ogura/final.model"
    usage: python2.7 test_seq2seq.py  --gpu -1 --model model-ogura --file ogura-enc_wakachi.txt
"""

__version__ = '0.0.1'

import sys

reload(sys)
sys.setdefaultencoding('utf-8')
#print sys.getdefaultencoding()

import re
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

import pprint
def pp(obj):
    pp = pprint.PrettyPrinter(indent=1, width=160)
    str = pp.pformat(obj)
    print re.sub(r"\\u([0-9a-f]{4})", lambda x: unichr(int("0x"+x.group(1),16)), str)

import os, math, time
start_time = time.time()

import numpy as np
import cPickle as pickle
import midi, glob
np.set_printoptions(precision=20)

from chainer import cuda, FunctionSet, Chain, Variable, optimizers, serializers, computational_graph
import chainer.functions as F
import chainer.links as L


# RNNLM (リカレントニューラル言語モデルの設定
# 辞書データを入力ユニット数分の潜在ベクトル空間への変換
# LSTM を使用し,出力,入力制御,忘却,出力制御を行うため出力が4倍
class RNN(Chain):

    def __init__(self, n_vocab, n_units):
        super(RNN, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1_x=L.Linear(n_units, 4 * n_units),
            l1_h=L.Linear(n_units, 4 * n_units),
            l2_h=L.Linear(n_units, 4 * n_units),
            l2_x=L.Linear(n_units, 4 * n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        self.n_units = n_units

    def set_word_embedding(self, array):
        self.embed.W.data = array

    def reset_state(self):
        self.l1_x.reset_state()

    # 1ステップ前方処理関数 (学習データ,状態を与える)
    def forward(self, x_data, state):

        x = Variable(x_data.astype(np.int32), volatile='on')

        # 特徴ベクトルは Bag of words の形式なので潜在ベクトル空間に変換
        h0 = self.embed(x)

        # 過去の状態も考慮した第1の隠れ層を作成: 20x4=2600次元
        h1_in = self.l1_x(h0) + self.l1_h(state['h1'])

        # LSTM に現在の状態と先ほど定義した隠れ層を付与して学習し,隠れ層と状態を出力
        c1, h1 = F.lstm(state['c1'], h1_in)

        # 過去の状態も考慮した第2の隠れ層を作成: 650x4=2600次元
        h2_in = self.l2_x(h1) + self.l2_h(state['h2'])

        # LSTM に現在の状態と先ほど定義した隠れ層を付与して学習し,隠れ層と状態を出力
        c2, h2 = F.lstm(state['c2'], h2_in)

        # ラベル4層目の処理で出力された値を使用する
        y = self.l3(h2)
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

        # ソフトマックス関数を用いて確率を求める
        return state, F.softmax(y)

    # 状態の初期化 (初期状態を現在の状態にセット)
    def initialize_state(self, n_units, batchsize):
        state = {}
        for name in ('c1', 'h1', 'c2', 'h2'):
            state[name] = Variable(np.zeros((batchsize, n_units), dtype=np.float32), volatile='on')
        return state


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: Sequence-to-Sequence model')
    parser.add_argument('--gpu',     '-g', default=-1, type=int,     help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model',   '-m', default='', type=str,     help='model directory')
    parser.add_argument('--file',    '-f', default='', type=str,     help='evaluate input file (.txt)')
    parser.add_argument('--text',    '-t', default='', type=unicode, help='start with this text')
    parser.add_argument('--console', '-c', default='', type=unicode, help='start with input text')
    parser.add_argument('--count',   '-n', default='10', type=int,   help='number of outputs')
    args = parser.parse_args()

    model_dir = args.model

    print('# GPU: {}'.format(args.gpu))
    # print('# minibatch: {}'.format(batchsize))
    # print('# epoch: {}'.format(n_epoch))
    print('# model directory: {}'.format(model_dir))
    # print('')

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np
    # xp.random.seed(123)

    # load vocabulary
    vocab = pickle.load(open(os.path.join(model_dir, 'vocab.bin'), 'rb'))
    ivocab = {}
    for c, i in vocab.items():
        ivocab[i] = c

    # load model
    model = pickle.load(open(os.path.join(model_dir, 'final.model'), 'rb'))

    n_units = model.embed.W.data.shape[1]

    print('# unit: {}'.format(n_units))
    print('# vocab size: {}'.format(len(vocab)))
    print('')

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    encoder_model = model
    decoder_model = model


    def evaluate(tokens):

        output_text = u''

        # initialize generator
        state = model.initialize_state(n_units, 1)
        if args.gpu >= 0:
            for key, value in state.items():
                value.data = cuda.to_gpu(value.data)

        # encode
        for token in tokens:
            if token not in vocab:
                continue

            prev_token = xp.array([vocab[token]], dtype=xp.int32)
            if args.gpu >= 0:
                prev_token = cuda.to_gpu(prev_token)

            state, prob = model.forward(prev_token, state)

        prev_token = xp.array([vocab[u'<go>']], dtype=xp.int32)
        if args.gpu >= 0:
            prev_token = cuda.to_gpu(prev_token)

        # decode
        for i in xrange(2000):
            state, prob = model.forward(prev_token, state)

            if True:
                probability = cuda.to_cpu(prob.data)[0].astype(xp.float64)
                probability /= xp.sum(probability)
                index = np.random.choice(range(len(probability)), p=probability)
            else:
                index = xp.argmax(cuda.to_cpu(prob.data))

            if ivocab[index] != u'<eos>':
                output_text += ivocab[index]
            else:
                # output_text += u'\n'
                break

            prev_token = xp.array([index], dtype=xp.int32)
            if args.gpu >= 0:
                prev_token = cuda.to_gpu(prev_token)

        return u'{}'.format(output_text)


    if args.file:
        for line in open(args.file, 'rU'):
            text = unicode(line).strip().split(u'\t')[0].strip()
            if not text or text.startswith(u'#'):
                continue

            text = text.replace(u'<eos>', u'').strip()
            print(u'>> {}'.format(text))

            tokens = text.split(u' ')

            ret = []
            for i in xrange(100):
                out = evaluate(tokens)
                if out != u'' and out not in ret:
                    print out
                    ret.append(out)
                    if len(ret) > args.count:
                        break

    elif args.text:
        text = unicode(args.text).strip()
        tokens = text.split(u' ')
        evaluate(tokens)

    else:
        while True:
            text = raw_input('>> ')
            text = unicode(text).strip()
            tokens = text.split(u' ')
            evaluate(tokens)

    print('time spent:', time.time() - start_time)
