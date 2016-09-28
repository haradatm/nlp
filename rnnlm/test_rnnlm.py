#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sample script of recurrent neural network language model.

    usage: python2.7 train_rnnlm.py neko_wakachi.txt --w2vmodel "neko_w2v.bin" --gpu 0 --epoch 1 --unit 200 --batchsize 100 --gradclip 5.0 --output model-neko --initmodel "model-neko/final.model" --resume "model-neko/final.state"
    usage: python2.7 test_rnnlm.py  --gpu -1 --model model-neko --text "吾 輩 は 猫 で あ る 。" --perp
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

import chainer
from chainer import cuda, FunctionSet, Chain, Variable, optimizers, serializers, computational_graph
import chainer.functions as F
import chainer.links as L


# RNNLM (リカレントニューラル言語モデルの設定
# 辞書データを入力ユニット数分の潜在ベクトル空間への変換
# LSTM を使用し,出力,入力制御,忘却,出力制御を行うため出力が4倍
class RNNLM(Chain):

    def __init__(self, n_vocab, n_units):
        super(RNNLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1_x=L.Linear(n_units, 4 * n_units),
            l1_h=L.Linear(n_units, 4 * n_units),
            l2_h=L.Linear(n_units, 4 * n_units),
            l2_x=L.Linear(n_units, 4 * n_units),
            l3=L.Linear(n_units, n_vocab),
        )

    # 1ステップ前方処理関数 (学習データ,状態を与える)
    def forward(self, x_data, state):

        x = Variable(x_data.astype(np.int32), volatile='on')

        # 特徴ベクトルは Bag of words の形式なので潜在ベクトル空間に変換
        h0 = self.embed(x)

        # 過去の状態も考慮した第1の隠れ層を作成: 20x4=2600次元
        h1_in = self.l1_x(h0) + self.l1_h(state['h1'])

        # LSTM に現在の状態と先ほど定義した隠れ層を付与して学習し,隠れ層と状態を出力
        c1, h1 = F.lstm(state['c1'], h1_in)

        # 過去の状態も考慮した第2の隠れ層を作成
        h2_in = self.l2_x(h1) + self.l2_h(state['h2'])

        # LSTM に現在の状態と先ほど定義した隠れ層を付与して学習し,隠れ層と状態を出力
        c2, h2 = F.lstm(state['c2'], h2_in)

        # ラベルは3層目の処理で出力された値を使用する
        y = self.l3(h2)
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

        # ソフトマックス関数を用いて確率を求める
        return state, F.softmax(y)

    # 1ステップ前方処理関数 (学習データ,状態を与える)
    def evaluate(self, x_data, y_data, state):

        # 学習用データを作成(特徴量データ,ラベルデータ)
        x = Variable(x_data.astype(np.int32), volatile='on')
        t = Variable(y_data.astype(np.int32), volatile='on')

        # 特徴ベクトルは Bag of words の形式なので潜在ベクトル空間に変換
        h0 = self.embed(x)

        # 過去の状態も考慮した第1の隠れ層を作成: 20x4=2600次元
        h1_in = self.l1_x(h0) + self.l1_h(state['h1'])

        # LSTM に現在の状態と先ほど定義した隠れ層を付与して学習し,隠れ層と状態を出力
        c1, h1 = F.lstm(state['c1'], h1_in)

        # 過去の状態も考慮した第2の隠れ層を作成
        h2_in = self.l2_x(h1) + self.l2_h(state['h2'])

        # LSTM に現在の状態と先ほど定義した隠れ層を付与して学習し,隠れ層と状態を出力
        c2, h2 = F.lstm(state['c2'], h2_in)

        # ラベルは3層目の処理で出力された値を使用する
        y = self.l3(h2)
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

        # ソフトマックス関数の交差エントロピー関数を用いて誤差を求める
        return state, F.softmax_cross_entropy(y, t)

    # 状態の初期化 (初期状態を現在の状態にセット)
    def initialize_state(self, n_units, batchsize):
        state = {}
        for name in ('c1', 'h1', 'c2', 'h2'):
            state[name] = Variable(np.zeros((batchsize, n_units), dtype=np.float32), volatile='on')
        return state


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: Recurrent neural network language model')
    parser.add_argument('--gpu',   '-g', default=-1, type=int,     help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='', type=str,     help='model directory')
    parser.add_argument('--file',  '-f', default='', type=str,     help='evaluate perplexity for file (.txt)')
    parser.add_argument('--text',  '-t', default='', type=unicode, help='start with input text')
    parser.add_argument('--perp',  '-p', default=False, action='store_true', help='evaluate perplexity for text')
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
    xp.random.seed(123)

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

    # initialize generator
    state = model.initialize_state(n_units, 1)
    if args.gpu >= 0:
        for key, value in state.items():
            value.data = cuda.to_gpu(value.data)

    if args.file:
        for line in open(args.file, 'rU'):
            text = unicode(line).strip().split(u'\t')[0].strip()
            if not text or text.startswith(u'#'):
                continue

            print(u'>> {}'.format(text))

            tokens = text.split(u' ')
            sum_log_perp = xp.zeros(())
            cur_at = time.time()

            for i in xrange(len(tokens) - 1):
                prev_token = xp.array([vocab[tokens[i]]],   dtype=xp.int32)
                next_token = xp.array([vocab[tokens[i+1]]], dtype=xp.int32)
                if args.gpu >= 0:
                    prev_token = cuda.to_gpu(prev_token)
                    next_token = cuda.to_gpu(next_token)

                state, loss = model.evaluate(prev_token, next_token, state)
                sum_log_perp += loss.data

            now = time.time()
            perp = math.exp(float(sum_log_perp) / len(text))
            print 'perplexity: {:.6f}, logprob: {:.6f}, time: {:.2f}'.format(perp, -float(sum_log_perp), now - cur_at)

    elif args.text and args.perp:
        text = args.text
        text = unicode(args.text).strip()

        tokens = text.split(u' ')
        sum_log_perp = xp.zeros(())
        cur_at = time.time()

        sys.stdout.write(tokens[0])

        for i in xrange(len(tokens) - 1):
            sys.stdout.write(tokens[i + 1])

            prev_token = xp.array([vocab[tokens[i]]],   dtype=xp.int32)
            next_token = xp.array([vocab[tokens[i+1]]], dtype=xp.int32)
            if args.gpu >= 0:
                prev_token = cuda.to_gpu(prev_token)
                next_token = cuda.to_gpu(next_token)

            state, loss = model.evaluate(prev_token, next_token, state)
            sum_log_perp += loss.data
        print('')

        now = time.time()
        perp = math.exp(float(sum_log_perp) / len(text))
        print 'perplexity: {:.6f}, logprob: {:.6f}, time: {:.2f}'.format(perp, -float(sum_log_perp), now - cur_at)

    else:
        if args.text:
            text = unicode(args.text).strip()
        else:
            text = u'吾 輩 は 猫 で あ る 。'

        tokens = text.split(u' ')

        for token in tokens:
            sys.stdout.write(token)

            if token not in vocab:
                continue

            prev_token = xp.array([vocab[token]], dtype=xp.int32)
            if args.gpu >= 0:
                prev_token = cuda.to_gpu(prev_token)

            state, prob = model.forward(prev_token, state)

        for i in xrange(2000):
            state, prob = model.forward(prev_token, state)

            if True:
                probability = cuda.to_cpu(prob.data)[0].astype(xp.float64)
                probability /= xp.sum(probability)
                index = np.random.choice(range(len(probability)), p=probability)
            else:
                index = xp.argmax(cuda.to_cpu(prob.data))

            if ivocab[index] != u'</s>':
                sys.stdout.write(ivocab[index])
            else:
                sys.stdout.write(u'\n')

            prev_token = xp.array([index])
            if args.gpu >= 0:
                prev_token = cuda.to_gpu(prev_token)
        print

    print('time spent:', time.time() - start_time)
