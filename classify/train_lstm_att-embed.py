#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: LSTM Neural Networks with Attention Mechanizm for Sentence Classification

http://

"""

__version__ = '0.0.1'

import sys

reload(sys)
sys.setdefaultencoding('utf-8')
#print sys.getdefaultencoding()

import re, math
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

    # for vec in X:
    #     pad = [vocab[PAD_TOKEN] for _ in range(max_len - len(vec))]
    #     vec.extend(pad)

    return X, Y, labels, vocab


def batch(generator, batch_size):
    batch = []
    for line in generator:
        batch.append(line)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def batch_tuple(generator, batch_size):
    batch = []
    for line in generator:
        batch.append(line)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch))
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch))


def sorted_parallel(generator1, generator2, pooling, order=0):
    gen1 = batch(generator1, pooling)
    gen2 = batch(generator2, pooling)
    for batch1, batch2 in zip(gen1, gen2):
        for x in sorted(zip(batch1, batch2), key=lambda x: len(x[order])):
            yield x


def fill_batch(batch, padding, min_height=1):
    max_len = max([len(x) for x in batch] + [min_height])
    return [x + [padding] * (max_len - len(x) + 1) for x in batch]


class MyATT(Chain):
    def __init__(self, n_vocab, n_dim, n_label):
        super(MyATT, self).__init__(
            embed=L.EmbedID(n_vocab, n_dim),
            fwd=L.GRU(50, n_inputs=n_dim),
            bwd=L.GRU(50, n_inputs=n_dim),
            fc0=L.Linear(100, 100),
            fc1=L.Linear(100, 100),
            fc2=L.Linear(100, 100),
        )
        self.uw = (xp.random.rand(1, 100).astype(np.float32) - 0.5) / 100

    def __call__(self, x, t, train=True):
        x = self.embed(x)
        y = self.forward(x, train=train)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def forward(self, x, train=True):
        # x:  50x300
        # uw: 1x100
        uw = F.tanh(self.fc0(Variable(self.uw, volatile=not train)))

        h1_list = []    # h1: 50x50
        # c1 = Variable(xp.zeros((x.shape[0], 50), dtype=np.float32), volatile=not train)
        h1 = Variable(xp.zeros((x.shape[0], 50), dtype=np.float32), volatile=not train)
        for i in six.moves.range(x.shape[1]):
            # c1, h1 = self.fwd(c1, h1, x[:, i, :])
            h1 = self.fwd(h1, x[:, i, :])
            h1_list.append(h1)

        h2_list = []    # h2: 50x50
        # c2 = Variable(xp.zeros((x.shape[0], 50), dtype=np.float32), volatile=not train)
        h2 = Variable(xp.zeros((x.shape[0], 50), dtype=np.float32), volatile=not train)
        for i in reversed(six.moves.range(x.shape[1])):
            # c2, h2 = self.bwd(c2, h2, x[:, i, :])
            h2 = self.bwd(h2, x[:, i, :])
            h2_list.append(h2)

        hi_list = []    # h: 50x100
        for i in six.moves.range(x.shape[1]):
            hi = F.concat((h1_list[i], h2_list[i]), axis=1)
            hi_list.append(hi)

        ui_list = []    # u: 50x100
        for i in six.moves.range(x.shape[1]):
            ui = F.tanh(self.fc1(hi_list[i]))
            ui_list.append(ui)

        sim_list = []   # sim: 50x1
        for i in six.moves.range(x.shape[1]):
            sim = F.batch_matmul(F.broadcast_to(uw, (x.shape[0], 100)), ui_list[i], transa=True)
            sim_list.append(F.reshape(sim, (x.shape[0], 1)))

        alpha_list = [] # alpah: 50x1
        alpha_mat = F.softmax(F.concat(sim_list, axis=1))
        for i in six.moves.range(x.shape[1]):
            alpha = F.reshape(alpha_mat[:, i], (x.shape[0], 1))
            alpha_list.append(alpha)

        # s: 50x100
        si = Variable(xp.zeros((x.shape[0], 100), dtype=np.float32), volatile=not train)
        for i in six.moves.range(x.shape[1]):
            si += F.reshape(F.batch_matmul(hi_list[i], alpha_list[i]), (x.shape[0], 100))

        y = self.fc2(si)

        return y


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: MyATT')
    parser.add_argument('--train',           default='',  type=unicode, help='training file (.txt)')
    parser.add_argument('--test',            default='',  type=unicode, help='evaluating file (.txt)')
    # parser.add_argument('--w2v',       '-w', default='',  type=unicode, help='word2vec model file (.bin)')
    parser.add_argument('--gpu',       '-g', default=-1,  type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch',     '-e', default=25,  type=int, help='number of epochs to learn')
    parser.add_argument('--unit',      '-u', default=100, type=int, help='number of output channels')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning batchsize size')
    parser.add_argument('--output',    '-o', default='model-gru_att-embed-sort',  type=str, help='output directory')
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

    # データの読み込み
    if not args.test:
        # トレーニング+テストデータ
        X, y, labels, vocab = load_data(args.train)

        # トレーニングデータとテストデータに分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    else:
        # トレーニングデータ
        X_train, y_train, labels, vocab = load_data(args.train)

        # テストデータ
        X_test, y_test, labels, vocab = load_data(args.test, labels=labels, vocab=vocab)

    n_dim   = 300
    n_vocab = len(vocab)
    n_label = len(labels)
    height  = len(X_train[0])
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

    # Prepare LSTM-ATT model
    model = MyATT(n_vocab, n_dim, n_label)

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

        # sorted_gen = batch_tuple(sorted_parallel(X_train, y_train, 100 * batchsize), batchsize)
        sorted_gen = batch_tuple(sorted_parallel(X_train, y_train, N), batchsize)
        sum_train_loss = 0.
        sum_train_accuracy = 0.
        K = 0

        # training
        for x_batch, t_batch in sorted_gen:
            x_batch = fill_batch(x_batch, vocab[PAD_TOKEN], min_height=5)

            # N 個の順番をランダムに並び替える
            perm = np.random.permutation(len(x_batch))
            x = Variable(xp.asarray(x_batch, dtype=np.int32)[perm], volatile='off')
            t = Variable(xp.asarray(t_batch, dtype=np.int32)[perm], volatile='off')

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

        # sorted_gen = batch_tuple(sorted_parallel(X_test, y_test, 100 * batchsize), batchsize)
        sorted_gen = batch_tuple(sorted_parallel(X_test, y_test, N_test), batchsize)
        sum_test_loss = 0.
        sum_test_accuracy = 0.
        K = 0

        # evaluation
        for x_batch, t_batch in sorted_gen:
            x_batch = fill_batch(x_batch, vocab[PAD_TOKEN], min_height=5)

            x = Variable(xp.asarray(x_batch, dtype=np.int32), volatile='on')
            t = Variable(xp.asarray(t_batch, dtype=np.int32), volatile='on')

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
