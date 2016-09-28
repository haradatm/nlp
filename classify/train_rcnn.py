#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: Recurrent convolutional neural networks for text classification

http://dl.acm.org/citation.cfm?id=2886636

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

from chainer import cuda, FunctionSet, Chain, Variable, optimizers, serializers, computational_graph
import chainer.functions as F
import chainer.links as L
from sklearn import metrics


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

            except RuntimeError:
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


def make_one_vector(w2v, vocab, words):

    vec = [0] * len(w2v[0])
    for word in words:
        if word not in vocab:
            logger.error('Word: "{0}" is not found'.format(word))
        else:
            seed_idx = vocab[word]
            seed_vector = w2v[seed_idx]
            vec += seed_vector

    vlen = np.linalg.norm(vec, 2)
    if vlen > 0:
        vec /= vlen
    return vec


def load_data(filename, w2v, vocab):

    X = None
    Y = []

    labels = {}
    inv_labels = {}
    max_tokens = 0

    for i, line in enumerate(open(filename, 'rU')):
        # if i >= 100:
        #     break

        line = unicode(line).strip()
        if line == u'':
            continue

        cols = line.split(u'\t')

        text = cols[1]
        tokens = text.split(u' ') + [u'</s>']

        # vec = make_one_vector(w2v, vocab, tokens)
        vec = []
        for token in tokens:
            if token not in vocab:
                logger.error('Word: "{0}" is not found'.format(token))
            else:
                vec.append(vocab[token])

        # X.append(vec)
        if X is None:
            X = np.array([vec])

        else:
            # 長いベクトルが来たら,行列を拡張する (不足は eos でパディングする)
            if len(vec) > max_tokens:
                max_tokens = len(vec)
                nrow, ncol = X.shape
                pad_size = max_tokens - ncol
                pad_array = np.array([vocab[u'</s>'] for _ in xrange(nrow * pad_size)]).reshape((nrow, pad_size))
                X = np.hstack([X, pad_array])

            # 短いベクトルが来たら,ベクトルを拡張する (不足は eos でパディングする)
            elif len(vec) < max_tokens:
                for _ in xrange(max_tokens - len(vec)):
                    vec.append(vocab[u'</s>'])

            X = np.vstack([X, np.asarray(vec)])

        label = cols[0]
        if label not in inv_labels:
            inv_labels[label] = len(inv_labels)
            labels[inv_labels[label]] = label
        Y.append(inv_labels[label])

    x = np.asarray(X, dtype=np.int32)
    y = np.asarray(Y, dtype=np.int32)

    print('Loading training dataset ... done.')

    return x, y, labels


class RCNN(Chain):

    def __init__(self, n_vocab, n_dim, n_unit, n_labels):
        super(RCNN, self).__init__(
            embed=L.EmbedID(n_vocab, n_dim),
            fwd=L.StatefulGRU(n_dim, n_dim),
            bwd=L.StatefulGRU(n_dim, n_dim),
            fc1=L.Linear(n_dim * 3, n_unit),
            fc2=L.Linear(n_unit, n_labels)
        )

    def __call__(self, x, t, train=True, gpu=-1, eos_id=0):

        x_list = []
        for i in xrange(x.shape[1]):
            v = np.asarray(x[:, i], dtype=np.int32)
            if gpu >= 0:
                v = cuda.to_gpu(v)
            x_list.append(Variable(v, volatile=not train))

        y = self.forward(x_list, eos_id=eos_id)

        t = np.asarray(t, dtype=np.int32)
        if gpu >= 0:
            t = cuda.to_gpu(t)
        t = Variable(t, volatile=not train)

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def forward(self, x_list, eos_id=0):

        self.fwd.reset_state()
        self.bwd.reset_state()

        w_list = []
        for x in x_list:
            w_list.append(self.embed(x))

        cl_list = []
        cr_list = []
        for i in xrange(len(w_list)):
            cl_list.append(self.fwd(w_list[i]))
            cr_list.append(self.bwd(w_list[-(i+1)]))

        xi_list = []
        for i in xrange(len(w_list)):
            xi_list.append(F.concat((cl_list[i], w_list[i], cr_list[-(i+1)])))

        yi_list = []
        for xi in xi_list:
            yi_list.append(F.tanh(self.fc1(xi)))

        y3 = yi_list[0]
        for yi in yi_list[1:]:
            y3 = F.maximum(yi, y3)

        y4 = self.fc2(y3)
        return y4

    def set_word_embedding(self, array):
        self.embed.W.data = array


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: MLP on MNIST')
    parser.add_argument('file',              default='',   type=str, help='training file (.txt)')
    parser.add_argument('--w2vmodel',  '-w', default='',   type=str, help='initialize word embedding layer with word2vec (.bin)')
    parser.add_argument('--initmodel', '-i', default='',   type=str, help='initialize the model with file')
    parser.add_argument('--resume',    '-r', default='',   type=str, help='resume the optimization from snapshot')
    parser.add_argument('--gpu',       '-g', default=-1,   type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch',     '-e', default=25,   type=int, help='number of epochs to learn')
    parser.add_argument('--unit',      '-u', default=1000, type=int, help='number of units')
    parser.add_argument('--batchsize', '-b', default=100,  type=int, help='learning minibatch size')
    parser.add_argument('--output',    '-o', default='model',  type=str, help='output directory')
    args = parser.parse_args()

    # 学習の繰り返し回数
    n_epoch = args.epoch

    # 中間層の数
    n_units = args.unit

    # 確率的勾配降下法で学習させる際の1回分のバッチサイズ
    batchsize = args.batchsize

    data_file = args.file
    model_dir = args.output

    print('# GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(n_units))
    print('# minibatch: {}'.format(batchsize))
    print('# epoch: {}'.format(n_epoch))
    # print('')
    sys.stdout.flush()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np
    xp.random.seed(123)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    (w2v, vocab, inv_vocab) = load_w2v_model(args.w2vmodel)
    n_vocab = len(vocab)
    n_dim = len(w2v[0])

    X, Y, labels = load_data(data_file, w2v, vocab)
    n_labels = len(labels)

    # DATASET = 'dataset.pkl'
    # if os.path.exists(DATASET):
    #     print('Loading Dataset: {}'.format(DATASET))
    #     with open(DATASET, 'rb') as f:
    #         dataset = pickle.load(f)
    # else:
    #     X, Y, labels = load_data(data_file, w2v, vocab)
    #     dataset = (X, Y, labels)
    #     print('Saving Dataset file: {}'.format(DATASET))
    #     pickle.dump(dataset, open(DATASET, 'wb'))
    #
    # X, Y, labels = dataset
    # n_labels = len(labels)

    # 学習用データを N個, 残りの個数を検証用データに設定
    N = (len(X) * 9 / 10)
    perm = np.random.permutation(len(X))
    train_idx = perm[0:N].copy()
    test_idx  = perm[N:].copy()

    x_train = X[perm[train_idx]]
    x_test  = X[perm[test_idx]]
    y_train = Y[perm[train_idx]]
    y_test  = Y[perm[test_idx]]

    N = len(x_train)
    N_test = len(x_test)
    print('train: {}, test: {}'.format(N, N_test))
    print('# data size: {}'.format(N))
    print('# vocab size: {}'.format(n_vocab))
    sys.stdout.flush()

    # Prepare RNN + Maxout model
    model = RCNN(n_vocab, n_dim, n_units, n_labels)

    # Load initialize model
    if args.initmodel and os.path.exists(args.initmodel):
        print('# Load model from: {}'.format(args.initmodel))
        sys.stdout.flush()
        model = pickle.load(open(args.initmodel, 'rb'))

        if args.resume and os.path.exists(args.resume):
            print('# Load optimizer state from: {}'.format(args.resume))
            sys.stdout.flush()
            optimizer = pickle.load(open(args.resume, 'rb'))

    # initialize word embedding layer with word2vec
    else:
        if args.w2vmodel:
            print('initializing word embedding with word2vec')
            sys.stdout.flush()
            model.set_word_embedding(w2v)
        else:
            print('initializing word embedding with random number')
            sys.stdout.flush()
            w = xp.array(
                [xp.array(
                    [xp.random.random() for unit in xrange(n_units)],
                    dtype=np.float32) for token in xrange(n_vocab)],
                dtype=np.float32)
            model.set_word_embedding(w)

    print('')
    sys.stdout.flush()

    if args.gpu >= 0:
        model.to_gpu()

    # Setup optimizer (Optimizer の設定)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # プロット用に実行結果を保存する
    train_loss = []
    train_acc  = []
    test_loss = []
    test_acc  = []

    start_at = time.time()
    cur_at = start_at

    # Learning loop
    print 'going to train {} epochs'.format(n_epoch)

    # Learning loop
    for epoch in xrange(1, n_epoch+1):

        sum_loss = 0.
        sum_accuracy = 0.

        # training
        # N 個の順番をランダムに並び替える
        perm = np.random.permutation(N)

        # 0..N までのデータをバッチサイズごとに使って学習
        for i in xrange(0, N, batchsize):
            x_batch = x_train[perm[i:i + batchsize]]
            y_batch = y_train[perm[i:i + batchsize]]
            # if args.gpu >= 0:
            #     x_batch = cuda.to_gpu(x_batch)
            #     y_batch = cuda.to_gpu(y_batch)

            # 勾配を初期化
            optimizer.zero_grads()

            # 順伝播させて誤差と精度を算出
            loss, acc = model(x_batch, y_batch, train=True, gpu=args.gpu, eos_id=vocab[u'</s>'])

            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

            sum_loss     += float(cuda.to_cpu(loss.data)) * len(x_batch)
            sum_accuracy += float(cuda.to_cpu(acc.data))  * len(x_batch)

            # if i % 500 == 0:
            # print '{} / {} loss={:.6f} accuracy={:.6f}'.format((i+len(x_batch)), N, sum_loss / (i+len(x_batch)), sum_accuracy / (i+len(x_batch)))
            # sys.stdout.flush()

        # 訓練データの誤差と,正解精度を表示
        now = time.time()
        throuput = now - cur_at
        print 'epoch: {} done'.format(epoch)
        print 'train mean loss={:.6f}, accuracy={:.6f} ({:.2f} sec)'.format(sum_loss / N, sum_accuracy / N, throuput)
        train_loss.append(sum_loss / N)
        train_acc.append(sum_accuracy / N)
        sys.stdout.flush()
        cur_at = now

        # evaluation
        # テストデータで誤差と正解精度を算出し汎化性能を確認
        sum_loss = 0.
        sum_accuracy = 0.
        for i in xrange(0, N_test, batchsize):
            x_batch = x_test[i:i + batchsize]
            y_batch = y_test[i:i + batchsize]
            # if args.gpu >= 0:
            #     x_batch = cuda.to_gpu(x_batch)
            #     y_batch = cuda.to_gpu(y_batch)

            # 順伝播させて誤差と精度を算出
            loss, acc = model(x_batch, y_batch, train=False, gpu=args.gpu, eos_id=vocab[u'</s>'])

            sum_loss     += float(cuda.to_cpu(loss.data)) * len(x_batch)
            sum_accuracy += float(cuda.to_cpu(acc.data))  * len(x_batch)

        # テストデータでの誤差と正解精度を表示
        now = time.time()
        throuput = now - cur_at
        print 'test  mean loss={:.6f}, accuracy={:.6f} ({:.2f} sec)'.format(sum_loss / N_test, sum_accuracy / N_test, throuput)
        test_loss.append(sum_loss / N_test)
        test_acc.append(sum_accuracy / N_test)
        sys.stdout.flush()
        cur_at = now

        # model と optimizer を保存する
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
        plt.plot(range(len(train_acc)), train_acc)
        plt.plot(range(len(test_acc)), test_acc)
        plt.legend(['train_acc', 'test_acc'], loc=4)
        plt.title('Accuracy of rcnn recognition.')
        plt.subplot(1, 2, 2)
        # plt.ylim(0., 1.)
        plt.plot(range(len(train_loss)), train_loss)
        plt.plot(range(len(test_loss)), test_loss)
        plt.legend(['train_loss', 'test_loss'], loc=4)
        plt.title('Loss of rcnn recognition.')
        plt.savefig('train_rcnn_acc-loss_batch.png')
        # plt.show()

print('time spent:', time.time() - start_time)
