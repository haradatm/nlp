#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Sample script of recurrent neural network language model.

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

import struct
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import copy
np.set_printoptions(precision=20)

from chainer import cuda, FunctionSet, Chain, Variable, optimizers, serializers, computational_graph
import chainer.functions as F
import chainer.links as L
import chainer.optimizer


def load_w2v_model(path):

    with open(path, 'rb') as f:
        w2i = {}
        i2w = {}

        n_vocab, n_units = map(int, f.readline().split())
        w = np.empty((n_vocab, n_units), dtype=np.float32)

        for i in xrange(n_vocab):
            word = ''
            while True:
                ch = f.read(1)
                if ch == ' ': break
                word += ch

            try:
                w2i[unicode(word)] = i
                i2w[i] = unicode(word)

            except UnicodeError:
                logging.error('Error unicode(): %s', word)
                w2i[word] = i
                i2w[i] = word

            w[i] = np.zeros(n_units)
            for j in xrange(n_units):
                w[i][j] = struct.unpack('f', f.read(struct.calcsize('f')))[0]

            # ベクトルを正規化する
            vlen = np.linalg.norm(w[i], 2)
            w[i] /= vlen

            # 改行を strip する
            assert f.read(1) == '\n'

    return w, w2i, i2w


start_text = None


def load_data(filename, vocab):
    global start_text

    dataset = []

    if not vocab:
        vocab = {}

    for i, line in enumerate(open(filename, 'rU')):
        line = unicode(line).strip()
        tokens = line.split(u' ') + [u'</s>']

        if i == 0:
            start_text = line.split(u' ')
            print('# start with: {}'.format(u' '.join(start_text)))

        for token in tokens:
            if token == u'':
                continue
            if token not in vocab:
                vocab[token] = len(vocab)
            dataset.append(vocab[token])

    return np.asarray(dataset, dtype=np.int32), vocab


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
            l3_h=L.Linear(n_units, 4 * n_units),
            l3_x=L.Linear(n_units, 4 * n_units),
            l4_h=L.Linear(n_units, 4 * n_units),
            l4_x=L.Linear(n_units, 4 * n_units),
            l5=L.Linear(n_units, n_vocab),
        )

    def set_word_embedding(self, array):
        self.embed.W.data = array

    def reset_state(self):
        self.l1_x.reset_state()

    # 1ステップ前方処理関数 (学習データ,状態を与える)
    def forward(self, x_data, y_data, state, dropout_ratio=0.5, train=True):

        # 学習用データを作成(特徴量データ,ラベルデータ)
        x = Variable(x_data.astype(np.int32), volatile=not train)
        t = Variable(y_data.astype(np.int32), volatile=not train)

        # 特徴ベクトルは Bag of words の形式なので潜在ベクトル空間に変換: 650次元(=n_units)
        h0 = self.embed(x)

        # 過学習をしないようランダムに一部のデータを捨て,過去の状態も考慮した第1の隠れ層を作成: 20x4=2600次元
        h1_in = F.dropout(self.l1_x(h0), ratio=dropout_ratio, train=train) + self.l1_h(state['h1'])

        # LSTM に現在の状態と先ほど定義した隠れ層を付与して学習し,隠れ層と状態を出力: 650個
        c1, h1 = F.lstm(state['c1'], h1_in)

        # 過学習をしないようにランダムに一部のデータを捨て,過去の状態も考慮した第2の隠れ層を作成: 650x4=2600次元
        h2_in = F.dropout(self.l2_x(h1), ratio=dropout_ratio, train=train) + self.l2_h(state['h2'])

        # LSTM に現在の状態と先ほど定義した隠れ層を付与して学習し,隠れ層と状態を出力: 650個
        c2, h2 = F.lstm(state['c2'], h2_in)

        # 過学習をしないようにランダムに一部のデータを捨て,過去の状態も考慮した第2の隠れ層を作成: 650x4=2600次元
        h3_in = F.dropout(self.l3_x(h2), ratio=dropout_ratio, train=train) + self.l3_h(state['h3'])

        # LSTM に現在の状態と先ほど定義した隠れ層を付与して学習し,隠れ層と状態を出力: 650個
        c3, h3 = F.lstm(state['c3'], h3_in)

        # 過学習をしないようにランダムに一部のデータを捨て,過去の状態も考慮した第2の隠れ層を作成: 650x4=2600次元
        h4_in = F.dropout(self.l4_x(h3), ratio=dropout_ratio, train=train) + self.l4_h(state['h4'])

        # LSTM に現在の状態と先ほど定義した隠れ層を付与して学習し,隠れ層と状態を出力: 650個
        c4, h4 = F.lstm(state['c4'], h4_in)

        # ラベル4層目の処理で出力された値を使用する
        y = self.l5(h4)
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2, 'c3': c3, 'h3': h3, 'c4': c4, 'h4': h4}

        loss = F.softmax_cross_entropy(y, t)

        if train:
            # ソフトマックス関数の交差エントロピー関数を用いて誤差を求める
            return state, F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        else:
            # ソフトマックス関数を用いて確率を求める
            return state, F.softmax(y)

    # 状態の初期化 (初期状態を現在の状態にセット)
    def initialize_state(self, n_units, batchsize, train=True):
        state = {}
        for name in ('c1', 'h1', 'c2', 'h2', 'c3', 'h3', 'c4', 'h4'):
            state[name] = Variable(np.zeros((batchsize, n_units), dtype=np.float32), volatile=not train)
        return state


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: Recurrent neural network language model')
    parser.add_argument('file',              default='',       type=str,     help='training file (.txt)')
    parser.add_argument('--w2vmodel',  '-w', default='',       type=str,     help='initialize word embedding layer with word2vec (.bin)')
    parser.add_argument('--initmodel', '-i', default='',       type=str,     help='initialize the model with file')
    parser.add_argument('--resume',    '-r', default='',       type=str,     help='resume the optimization from snapshot')
    parser.add_argument('--gpu',       '-g', default=-1,       type=int,     help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch',     '-e', default=100,      type=int,     help='number of epochs')
    parser.add_argument('--unit',      '-u', default=50,       type=int,     help='number of units')
    parser.add_argument('--batchsize', '-b', default=20,       type=int,     help='learning minibatch size')
    parser.add_argument('--gradclip',  '-c', default=5.,       type=float,   help='gradient clip threshold')
    parser.add_argument('--bproplen',  '-l', default=35,       type=int,     help='BPTT truncate length')
    parser.add_argument('--output',    '-o', default='model',  type=str,     help='output directory')
    args = parser.parse_args()

    # 学習の繰り返し回数
    n_epoch = args.epoch

    # 中間層の数
    n_units = args.unit

    # 確率的勾配降下法で学習させる際の1回分のバッチサイズ
    batch_size = args.batchsize

    # 学習に使用する文字列の長さ
    bprop_len = args.bproplen

    # 勾配法で使用するしきい値
    grad_clip = args.gradclip

    # その他の学習パラメータ
    dropout = 0.5

    data_file = args.file
    model_dir = args.output

    print('# GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(n_units))
    print('# minibatch: {}'.format(batch_size))
    print('# epoch: {}'.format(n_epoch))
    # print('')

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np
    xp.random.seed(123)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if args.w2vmodel and os.path.exists(args.w2vmodel):
        (w2v, vocab, inv_vocab) = load_w2v_model(args.w2vmodel)
    else:
        vocab = None

    train_data, vocab = load_data(data_file, vocab)
    N = len(train_data)
    n_vocab = len(vocab)

    with open(os.path.join(model_dir, 'vocab.bin'), 'wb') as f:
        pickle.dump(vocab, f)

    print('# training data from: {}'.format(data_file))
    print('# data size: {}'.format(N))
    print('# vocab size: {}'.format(n_vocab))

    # Recurrent neural net languabe model
    model = RNNLM(n_vocab, n_units)
    optimizer = optimizers.Adam()

    # 初期のパラメータを -0.1..0.1 の間で与える
    # for param in model.params():
    #     data = param.data
    #     data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    # Load initialize model
    if args.initmodel and os.path.exists(args.initmodel):
        print('# Load model from: {}'.format(args.initmodel))
        model = pickle.load(open(args.initmodel, 'rb'))

        if args.resume and os.path.exists(args.resume):
            print('# Load optimizer state from: {}'.format(args.resume))
            optimizer = pickle.load(open(args.resume, 'rb'))

    # initialize word embedding layer with word2vec
    else:
        if args.w2vmodel:
            print('initializing word embedding with word2vec')
            model.set_word_embedding(w2v)
        else:
            print('initializing word embedding with random number')
            w = xp.array(
                [xp.array(
                    [xp.random.random() for unit in xrange(n_units)],
                    dtype=np.float32) for token in xrange(n_vocab)],
                dtype=np.float32)
            model.set_word_embedding(w)

    print('')

    if args.gpu >= 0:
        model.to_gpu()

    # Setup optimizer
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

    # ジャンプの幅を設定（順次学習しない）
    jump = N / batch_size

    # 最初の時間情報を取得
    start_at = time.time()
    cur_at = start_at

    # 状態の初期化 (初期状態を現在の状態にセット)
    state = model.initialize_state(n_units, batch_size, train=True)
    if args.gpu >= 0:
        for key, value in state.items():
            value.data = cuda.to_gpu(value.data)

    # 損失を 0 で初期化
    sum_loss = Variable(xp.zeros((), dtype=np.float32))
    sum_accuracy = Variable(xp.zeros((), dtype=np.float32))
    cur_log_perp = xp.zeros(())
    epoch = 0
    batch_idxs = list(range(batch_size))

    # プロット用に実行結果を保存する
    train_loss = []
    train_acc  = []
    train_prep = []

    # Learning loop
    print 'going to train {} iterations'.format(jump * n_epoch)

    # 確率的勾配法を用いた学習
    for i in xrange(jump * n_epoch):

        # 一定のデータを選択し損失計算をしながらパラメータ更新
        x_batch = xp.array([train_data[(jump * x + i    ) % N] for x in batch_idxs])
        y_batch = xp.array([train_data[(jump * x + i + 1) % N] for x in batch_idxs])
        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        # 逐次尤度を計算
        state, loss, acc = model.forward(x_batch, y_batch, state, dropout_ratio=dropout, train=True)
        sum_loss += loss
        sum_accuracy += acc
        cur_log_perp += loss.data

        # バックプロパゲーションでパラメータを更新 (truncate はどれだけ過去の履歴を見るか)
        if (i + 1) % bprop_len == 0:  # Run truncated BPTT
            now = time.time()
            perp = math.exp(float(cur_log_perp) / (i % 10000 + 1))
            print '{:.2f}%, train loss: {:.6f}, perplexity: {:.6f}, time: {:.2f}'.format(100. * (i+1) / (jump * n_epoch), float(sum_loss.data) / bprop_len, perp, now - cur_at)
            train_loss.append(sum_loss.data / bprop_len)
            train_acc.append(sum_accuracy.data / bprop_len)
            train_prep.append(perp)
            cur_at = now

            optimizer.zero_grads()
            sum_loss.backward()
            sum_loss.unchain_backward()  # truncate
            sum_loss = Variable(xp.zeros((), dtype=np.float32))
            sum_accuracy = Variable(xp.zeros((), dtype=np.float32))

            # L2 正則化
            optimizer.update()
            optimizer.clip_grads(grad_clip)

        if (i + 1) % 10000 == 0:
            now = time.time()
            throuput = 10000. / (now - cur_at)
            perp = math.exp(float(cur_log_perp) / 10000)
            print('epoch {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(i+1, perp, throuput))
            cur_at = now
            cur_log_perp.fill(0)

            if args.gpu >= 0: model.to_cpu()
            with open(os.path.join(model_dir, 'epoch_{:.2f}.model'.format(float(i)/jump)), 'wb') as f:
                pickle.dump(model, f)
            if args.gpu >= 0: model.to_gpu()

            with open(os.path.join(model_dir, 'epoch_{:.2f}.state'.format(float(i)/jump)), 'wb') as f:
                pickle.dump(optimizer, f)

        sys.stdout.flush()

    if args.gpu >= 0: model.to_cpu()
    with open(os.path.join(model_dir, 'final.model'), 'wb') as f:
        pickle.dump(model, f)
    if args.gpu >= 0: model.to_gpu()

    with open(os.path.join(model_dir, 'final.state'), 'wb') as f:
        pickle.dump(optimizer, f)

    # 精度と誤差をグラフ描画
    if True:

        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        # plt.ylim(0., 1.)
        plt.plot(range(len(train_loss)), train_loss)
        plt.plot(range(len(train_acc)), train_acc)
        plt.legend(['train_loss', 'train_accu'], loc=4)
        plt.title('Loss of rnn training.')
        plt.subplot(1, 2, 2)
        # plt.ylim(0., 1.)
        plt.plot(range(len(train_prep)), train_prep)
        plt.legend(['train_prep'], loc=4)
        plt.title('Perplexity for training data.')
        plt.savefig('train-rnnlm-{}_acc-loss.png'.format(args.output))
        # plt.show()

    # テスト
    # load vocabulary
    vocab = pickle.load(open(os.path.join(model_dir, 'vocab.bin'), 'rb'))
    ivocab = {}
    for c, i in vocab.items():
        ivocab[i] = c

    # load model
    model = pickle.load(open(os.path.join(model_dir, 'final.model'), 'rb'))

    n_units = model.embed.W.data.shape[1]

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # initialize generator
    state = model.initialize_state(n_units, 1, train=False)
    if args.gpu >= 0:
        for key, value in state.items():
            value.data = cuda.to_gpu(value.data)

    for token in start_text:
        sys.stdout.write(token)

        prev_token = xp.array([vocab[token]], dtype=xp.int32)
        if args.gpu >= 0:
            prev_token = cuda.to_gpu(prev_token)

        state, prob = model.forward(prev_token, prev_token, state, train=False)

    for i in xrange(2000):
        state, prob = model.forward(prev_token, prev_token, state, train=False)

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
