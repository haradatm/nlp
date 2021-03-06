#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: Convolutional Neural Networks for Sentence Classification

http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf

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

import os, time, six
start_time = time.time()

import struct
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import copy

from chainer import cuda, Chain, Variable, optimizers, serializers, computational_graph
import chainer.functions as F
import chainer.links as L
import chainer.optimizer

xp = np
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'


def load_data(path, labels={}, vocab={}):
    X, Y = [], []
    max_len = 0

    if len(vocab) > 0:
        train = False
    else:
        train = True
        vocab[UNK_TOKEN] = len(vocab)
        vocab[PAD_TOKEN] = len(vocab)

    f = open(path, 'rU')
    for i, line in enumerate(f):
        # if i >= 10:
        #     break

        line = unicode(line).strip()
        if line == u'':
            continue

        line = line.replace(u'. . .', u'…')

        cols = line.split(u'\t')
        if len(cols) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        label = cols[0]
        text  = cols[1]

        tokens = text.split(' ')

        vec = []
        for token in tokens:
            try:
                vec.append(vocab[token])
            except KeyError:
                if train:
                    vocab[token] = len(vocab)
                    vec.append(vocab[token])
                else:
                    sys.stderr.write('unk: {}\n'.format(token))
                    vec.append(vocab[UNK_TOKEN])

        if len(vec) > max_len:
            max_len = len(vec)

        X.append(vec)

        if label not in labels:
            labels[label] = len(labels)
        Y.append(labels[label])

    f.close()

    for vec in X:
        pad = [vocab[PAD_TOKEN] for _ in range(max_len - len(vec))]
        vec.extend(pad)

    return X, Y, labels, vocab


class MyCNN(Chain):
    def __init__(self, input_channel, output_channel, height, width, n_label, n_vocab):
        super(MyCNN, self).__init__(
            embed=L.EmbedID(n_vocab, width),
            conv1=L.Convolution2D(input_channel, output_channel, (3, width), pad=0),
            conv2=L.Convolution2D(input_channel, output_channel, (4, width), pad=0),
            conv3=L.Convolution2D(input_channel, output_channel, (5, width), pad=0),
            fc4=L.Linear(output_channel * 3, output_channel * 3),
            fc5=L.Linear(output_channel * 3, n_label)
        )
        self.input_channel = input_channel
        self.height = height
        self.width = width

    def __call__(self, x, t, train=True):
        # (nsample, channel, height, width) の4次元テンソルに変換
        x = self.embed(x).reshape((x.shape[0], self.input_channel, x.shape[1], self.width))
        y = self.forward(x, train=train)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def forward(self, x, train=True):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), (self.height + 1 - 3))
        h2 = F.max_pooling_2d(F.relu(self.conv2(x)), (self.height + 1 - 4))
        h3 = F.max_pooling_2d(F.relu(self.conv3(x)), (self.height + 1 - 5))

        # Convolution + Pooling を行った結果を結合する
        concat = F.concat((h1, h2, h3), axis=2)

        # 結合した結果に Dropout をかける
        h4 = F.dropout(F.tanh(self.fc4(concat)), ratio=0.5, train=train)

        # Dropout の結果を結合する
        y = self.fc5(h4)

        return y


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: MyCNN')
    parser.add_argument('--train',           default='',  type=unicode, help='training file (.txt)')
    parser.add_argument('--test',            default='',  type=unicode, help='evaluating file (.txt)')
    # parser.add_argument('--w2v',       '-w', default='',  type=unicode, help='word2vec model file (.bin)')
    parser.add_argument('--gpu',       '-g', default=-1,  type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch',     '-e', default=25,  type=int, help='number of epochs to learn')
    parser.add_argument('--unit',      '-u', default=100, type=int, help='number of output channels')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning batchsize size')
    parser.add_argument('--output',    '-o', default='model-cnn-embed',  type=str, help='output directory')
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np
    # xp.random.seed(123)

    # 学習の繰り返し回数
    n_epoch = args.epoch

    # 中間層の数
    n_units = args.unit

    # 確率的勾配降下法で学習させる際の1回分のバッチサイズ
    batchsize = args.batchsize

    model_dir = args.output
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # print('# loading word2vec model: {}'.format(args.w2v))
    # sys.stdout.flush()
    # model = load_w2v_model(args.w2v)
    # n_vocab = len(model.vocab)

    input_channel = 1
    output_channel = args.unit

    # データの読み込み
    if not args.test:
        # トレーニング+テストデータ
        X, y, labels, vocab = load_data(args.train)
        X = xp.asarray(X, dtype=np.int32)
        y = xp.asarray(y, dtype=np.int32)

        # (nsample, channel, height, width) の4次元テンソルに変換
        # X = X.reshape((X.shape[0], input_channel, X.shape[1], X.shape[2]))

        # トレーニングデータとテストデータに分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    else:
        # トレーニングデータ
        X, y, labels, vocab = load_data(args.train)
        X_train = xp.asarray(X, dtype=np.int32)
        y_train = xp.asarray(y, dtype=np.int32)

        # (nsample, channel, height, width) の4次元テンソルに変換
        # X_train = X_train.reshape((X_train.shape[0], input_channel, X_train.shape[1], X_train.shape[2]))

        # テストデータ
        X, y, labels, vocab = load_data(args.test, labels=labels, vocab=vocab)
        X_test = xp.asarray(X, dtype=np.int32)
        y_test = xp.asarray(y, dtype=np.int32)

        # (nsample, channel, height, width) の4次元テンソルに変換
        # X_test = X_test.reshape((X_test.shape[0], input_channel, X_test.shape[1], X_test.shape[2]))

    n_dim   = 300
    n_vocab = len(vocab)
    n_label = len(labels)
    height  = X_train.shape[1]
    width   = n_dim

    N = len(X_train)
    N_test = len(X_test)

    print('# gpu: {}'.format(args.gpu))
    print('# embedding dim: {}, vocab {}'.format(n_dim, n_vocab))
    print('# epoch: {}'.format(n_epoch))
    print('# batchsize: {}'.format(batchsize))
    print('# input channel: {}'.format(1))
    print('# output channel: {}'.format(n_units))
    print('# train: {}, test: {}'.format(N, N_test))
    print('# data height: {}, width: {}, labels: {}'.format(height, width, n_label))
    sys.stdout.flush()

    # Prepare CNN model
    model = MyCNN(input_channel, output_channel, height, width, n_label, n_vocab)

    if args.gpu >= 0:
        model.to_gpu()

    # 重み減衰
    decay = 0.0001

    # 勾配上限
    grad_clip = 3

    # Setup optimizer (Optimizer の設定)
    # optimizer = optimizers.Adam()
    optimizer = optimizers.AdaDelta()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

    # プロット用に実行結果を保存する
    train_loss = []
    train_norm = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    start_at = time.time()
    cur_at = start_at

    # Learning loop
    for epoch in six.moves.range(1, n_epoch + 1):

        print('epoch {:} / {:}'.format(epoch, n_epoch))
        sys.stdout.flush()

        # sorted_gen = batch(sorted_parallel(X_train, y_train, N * batchsize), batchsize)
        sum_train_loss = 0.
        sum_train_accuracy = 0.
        K = 0

        # training
        # N 個の順番をランダムに並び替える
        perm = np.random.permutation(N)
        for i in six.moves.range(0, N, batchsize):

            x = Variable(X_train[perm[i:i + batchsize]], volatile='off')
            t = Variable(y_train[perm[i:i + batchsize]], volatile='off')

            # 勾配を初期化
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            loss, accuracy = model(x, t, train=True)

            sum_train_loss += float(loss.data) * len(t)
            sum_train_accuracy += float(accuracy.data) * len(t)
            K += len(t)

            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

        train_loss.append(sum_train_loss / K)
        train_accuracy.append(sum_train_accuracy / K)

        # 訓練データの誤差と,正解精度を表示
        now = time.time()
        throuput = now - cur_at
        norm = optimizer.compute_grads_norm()
        print('train mean loss={:.6f}, accuracy={:.6f} ({:.6f} sec)'.format(sum_train_loss / K, sum_train_accuracy / K, throuput))
        sys.stdout.flush()
        cur_at = now

        # evaluation
        sum_test_loss = 0.
        sum_test_accuracy = 0.
        K = 0
        for i in six.moves.range(0, N_test, batchsize):

            x = Variable(X_test[i:i + batchsize], volatile='on')
            t = Variable(y_test[i:i + batchsize], volatile='on')

            # 順伝播させて誤差と精度を算出
            loss, accuracy = model(x, t, train=False)

            sum_test_loss += float(loss.data) * len(t)
            sum_test_accuracy += float(accuracy.data) * len(t)
            K += len(t)

        test_loss.append(sum_test_loss / K)
        test_accuracy.append(sum_test_accuracy / K)

        # テストデータでの誤差と正解精度を表示
        now = time.time()
        throuput = now - cur_at
        print(' test mean loss={:.6f}, accuracy={:.6f} ({:.6f} sec)'.format(sum_test_loss / K, sum_test_accuracy / K, throuput))
        sys.stdout.flush()
        cur_at = now

        # model と optimizer を保存する
        if args.gpu >= 0: model.to_cpu()
        with open(os.path.join(model_dir, 'epoch_{:03d}.model'.format(epoch)), 'wb') as f:
            pickle.dump(model, f)
        if args.gpu >= 0: model.to_gpu()
        with open(os.path.join(model_dir, 'epoch_{:03d}.state'.format(epoch)), 'wb') as f:
            pickle.dump(optimizer, f)

        # 精度と誤差をグラフ描画
        if True:
            ylim1 = [min(train_loss + test_loss), max(train_loss + test_loss)]
            ylim2 = [0.5, 1.0]

            # グラフ左
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.ylim(ylim1)
            plt.plot(range(1, len(train_loss) + 1), train_loss, 'b')
            plt.grid()
            plt.ylabel('loss')
            plt.legend(['train loss', 'train l2-norm'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'm')
            plt.grid()
            # plt.ylabel('accuracy')
            plt.legend(['train accuracy'], loc="upper left")
            plt.title('Loss and accuracy of training.')

            # グラフ右
            plt.subplot(1, 2, 2)
            plt.ylim(ylim1)
            plt.plot(range(1, len(test_loss) + 1), test_loss, 'b')
            plt.grid()
            # plt.ylabel('loss')
            plt.legend(['test loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(test_accuracy) + 1), test_accuracy, 'm')
            plt.grid()
            plt.ylabel('accuracy')
            plt.legend(['test accuracy'], loc="upper left")
            plt.title('Loss and accuracy of test.')

            plt.savefig('{}.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()

        cur_at = now

    # model と optimizer を保存する
    if args.gpu >= 0: model.to_cpu()
    with open(os.path.join(model_dir, 'final.model'), 'wb') as f:
        pickle.dump(model, f)
    if args.gpu >= 0: model.to_gpu()
    with open(os.path.join(model_dir, 'final.state'), 'wb') as f:
        pickle.dump(optimizer, f)

print('time spent:', time.time() - start_time)
