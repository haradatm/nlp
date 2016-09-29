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
logger.setLevel(logging.INFO)
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

    v = False if vocab is None else True

    if not v:
        vocab = {u'<go>': 0, u'<eos>': 1}

    for i, line in enumerate(open(filename, 'rU')):
        cols = unicode(line).strip().split(u'\t')

        if i == 0:
            start_text = cols[0].split(u' ')[:-1]
            print('# start with: {}'.format(u' '.join(start_text)))

        encode = []
        tokens = cols[0].split(u' ')
        for token in tokens:
            if token == u'':
                continue
            if token not in vocab:
                if v:
                    continue
                else:
                    vocab[token] = len(vocab)
            encode.append(vocab[token])

        decode = []
        tokens = cols[1].split(u' ')
        for token in tokens:
            if token == u'':
                continue
            if token not in vocab:
                if v:
                    continue
                else:
                    vocab[token] = len(vocab)
            decode.append(vocab[token])

        dataset.append((encode, decode))

    return dataset, vocab


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
    def forward(self, x, state, dropout_ratio=0.5, train=True):

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

        # ラベル4層目の処理で出力された値を使用する
        y = self.l3(h2)
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

        return y, state

    # 状態の初期化 (初期状態を現在の状態にセット)
    def initialize_state(self, n_units, batch_size, train=True):
        state = {}
        for name in ('c1', 'h1', 'c2', 'h2'):
            state[name] = Variable(np.zeros((batch_size, n_units), dtype=np.float32), volatile=not train)
        return state

if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: Sequence-to-Sequence model')
    parser.add_argument('file',              default='',       type=str,     help='training file (.txt)')
    parser.add_argument('--w2vmodel',  '-w', default='',       type=str,     help='initialize word embedding layer with word2vec (.bin)')
    parser.add_argument('--initmodel', '-m', default='',       type=str,     help='initialize the model with file')
    parser.add_argument('--gpu',       '-g', default=-1,       type=int,     help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch',     '-e', default=100,      type=int,     help='number of epochs')
    parser.add_argument('--unit',      '-u', default=50,       type=int,     help='number of units')
    parser.add_argument('--batchsize', '-b', default=20,       type=int,     help='learning minibatch size')
    parser.add_argument('--gradclip',  '-c', default=5.,       type=float,   help='gradient clip threshold')
    parser.add_argument('--output',    '-o', default='model',  type=str,     help='output directory')
    args = parser.parse_args()

    # 学習の繰り返し回数
    n_epoch = args.epoch

    # 中間層の数
    n_units = args.unit

    # 確率的勾配降下法で学習させる際の1回分のバッチサイズ
    batch_size = args.batchsize

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

    # Prepare encoder RNN model
    model = RNN(n_vocab, n_units)
    optimizer = optimizers.Adam()

    # 初期のパラメータを -0.1..0.1 の間で与える
    # for param in model.params():
    #     data = param.data
    #     data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    # Load initialize model
    if args.initmodel and os.path.exists(args.initmodel):
        print('# Load model from: {}'.format(args.initmodel))
        model = pickle.load(open(args.initmodel, 'rb'))

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

    # 確率的勾配法を用いた学習
    start_at = time.time()
    cur_at = start_at

    # プロット用に実行結果を保存する
    train_loss = []
    train_acc  = []
    train_prep = []
    train_grad = []

    # Learning loop
    print 'going to train {} iterations'.format(jump * n_epoch)

    # Learning loop
    for epoch in xrange(0, n_epoch):

        # training
        # N 個の順番をランダムに並び替える
        perm = np.random.permutation(N)
        sum_accuracy = Variable(np.zeros((), dtype=np.float32))
        sum_loss = Variable(np.zeros((), dtype=np.float32))

        # 0..N までのデータをバッチサイズごとに使って学習
        for i in xrange(0, N, batch_size):

            batch_loss = 0
            for j in perm[i:i + batch_size]:
                encode, decode = train_data[j]

                # 状態を初期化
                state = model.initialize_state(n_units, 1, train=True)
                if args.gpu >= 0:
                    for key, value in state.items():
                        value.data = cuda.to_gpu(value.data)

                # encoder
                if args.gpu >= 0:
                    x_batch = [Variable(cuda.to_gpu(xp.array([x], dtype=xp.int32)), volatile='off') for x in encode]
                else:
                    x_batch = [Variable(xp.array([x], dtype=xp.int32), volatile='off') for x in encode]

                for x in x_batch[:-1]:
                    y, state = model.forward(x, state=state, dropout_ratio=dropout, train=True)

                # decoder
                if args.gpu >= 0:
                    x_batch = [Variable(cuda.to_gpu(xp.array([x], dtype=xp.int32)), volatile='off') for x in decode]
                else:
                    x_batch = [Variable(xp.array([x], dtype=xp.int32), volatile='off') for x in decode]

                ys = []
                for x in x_batch:
                    y, state = model.forward(x, state=state, dropout_ratio=dropout, train=True)
                    ys.append(y)

                loss = 0
                for y, t in zip(ys, x_batch[1:]):
                    new_loss = F.softmax_cross_entropy(y, t)
                    loss += new_loss
                batch_loss += (loss / (len(x_batch) - 1))

            sum_loss = (batch_loss / batch_size)

            # バックプロパゲーションでパラメータを更新
            optimizer.zero_grads()
            sum_loss.backward()
            # sum_loss.unchain_backward()
            optimizer.update()

            now = time.time()
            norm = optimizer.compute_grads_norm()
            print '{:.2f}%, train loss: {:.6f}, grad L2 norm: {:.6f}, time: {:.2f}'.format((100. * epoch * N + (i+1) * batch_size) / (n_epoch * N), float(sum_loss.data), norm, now - cur_at)
            train_loss.append(sum_loss.data)
            train_acc.append(sum_accuracy.data)
            train_grad.append(norm)
            cur_at = now

            sum_loss = Variable(xp.zeros((), dtype=np.float32))
            sum_accuracy = Variable(xp.zeros((), dtype=np.float32))

            sys.stdout.flush()

        if (epoch + 1) % 100 == 0:
            now = time.time()
            throuput = 100. / (now - cur_at)
            print('epoch {} training throuput: {:.2f} iters/sec'.format(epoch + 1, throuput))
            cur_at = now

            if args.gpu >= 0: model.to_cpu()
            with open(os.path.join(model_dir, 'epoch_{}.model'.format(epoch + 1)), 'wb') as f:
                pickle.dump(model, f)
            if args.gpu >= 0: model.to_gpu()

        sys.stdout.flush()

    if args.gpu >= 0: model.to_cpu()
    with open(os.path.join(model_dir, 'final.model'), 'wb') as f:
        pickle.dump(model, f)
    if args.gpu >= 0: model.to_gpu()

    # 精度と誤差をグラフ描画
    if True:
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        # plt.ylim(0., 1.)
        plt.plot(range(len(train_loss)), train_loss)
        plt.plot(range(len(train_acc)), train_acc)
        plt.legend(['train_loss', 'train_acc'], loc=4)
        plt.title('Loss and accuracy of training.')
        plt.subplot(1, 2, 2)
        # plt.ylim(0., 1.)
        plt.plot(range(len(train_grad)), train_grad)
        plt.legend(['train_grad'], loc=4)
        plt.title('grad L2 norm of training.')
        plt.savefig('train-seq2seq-{}_acc-loss.png'.format(args.output))
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

    encoder_model = model
    decoder_model = model

    for order in range(1, 5):

        output_text = u''

        # initialize generator
        state = model.initialize_state(n_units, 1, train=False)
        if args.gpu >= 0:
            for key, value in state.items():
                value.data = cuda.to_gpu(value.data)

        # encode
        for token in start_text:
            if token not in vocab:
                continue

            prev_token = xp.array([vocab[token]], dtype=xp.int32)
            if args.gpu >= 0:
                prev_token = cuda.to_gpu(prev_token)

            y, state = model.forward(Variable(prev_token.astype(np.int32), volatile='on'), state, train=False)

        prev_token = xp.array([vocab[u'<go>']], dtype=xp.int32)
        if args.gpu >= 0:
            prev_token = cuda.to_gpu(prev_token)

        # decode
        for i in xrange(2000):
            y, state = model.forward(Variable(prev_token.astype(np.int32), volatile='on'), state, train=False)

            if output_text == u'':
                index = cuda.to_cpu(y.data[0]).argsort()[-order]
            else:
                index = cuda.to_cpu(y.data[0]).argmax()

            if ivocab[index] != u'<eos>':
                output_text += ivocab[index]
            else:
                # output_text += u'\n'
                break

            prev_token = xp.array([index], dtype=xp.int32)
            if args.gpu >= 0:
                prev_token = cuda.to_gpu(prev_token)

        print(u'{}'.format(output_text))

    print('time spent:', time.time() - start_time)
