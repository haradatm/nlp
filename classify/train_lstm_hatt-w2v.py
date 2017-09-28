#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: Hierarchical Attention Networks for Document Classification

https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf

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
UNK_VEC = None
PAD_VEC = None


def load_w2v_model(path):

    # with open(path, 'rb') as f:
    #     w2i = {}
    #     i2w = {}
    #
    #     n_vocab, n_units = map(int, f.readline().split())
    #     w = np.empty((n_vocab, n_units), dtype=np.float32)
    #
    #     for i in xrange(n_vocab):
    #         word = ''
    #         while True:
    #             ch = f.read(1)
    #             if ch == ' ': break
    #             word += ch
    #
    #         try:
    #             w2i[unicode(word)] = i
    #             i2w[i] = unicode(word)
    #
    #         except RuntimeError:
    #             logging.error('Error unicode(): %s', word)
    #             w2i[word] = i
    #             i2w[i] = word
    #
    #         w[i] = np.zeros(n_units)
    #         for j in xrange(n_units):
    #             w[i][j] = struct.unpack('f', f.read(struct.calcsize('f')))[0]
    #
    #         # ベクトルを正規化する
    #         vlen = np.linalg.norm(w[i], 2)
    #         w[i] /= vlen
    #
    #         # 改行を strip する
    #         assert f.read(1) == '\n'
    # return w, w2i, i2w

    from gensim.models import KeyedVectors
    w2v = KeyedVectors.load_word2vec_format(path, binary=True)

    global UNK_VEC, PAD_VEC
    UNK_VEC = seeded_vector(w2v, UNK_TOKEN)
    PAD_VEC = seeded_vector(w2v, PAD_TOKEN)

    return w2v


def seeded_vector(w2v, seed_string):
    once = xp.random.RandomState(hash(seed_string) & 0xffffffff)
    return (once.rand(w2v.vector_size) - 0.5) / w2v.vector_size


def load_data(path, w2v, labels={}):
    X, Y = [], []

    max_len_words = 0
    max_len_sents = 0

    f = open(path, 'rU')
    for i, line in enumerate(f):
        # if i >= 100:
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

        sentences = sent_splitter(text)
        # sentences = [text]

        sent_vec = []
        for sent in sentences:
            tokens = sent.split(' ')

            word_vec = []
            for token in tokens:
                try:
                    word_vec.append(w2v[token])
                except KeyError:
                    sys.stderr.write('unk: {}\n'.format(token))
                    word_vec.append(UNK_VEC)

            if len(word_vec) > max_len_words:
                max_len_words = len(word_vec)

            sent_vec.append(word_vec)

        if len(sent_vec) > max_len_sents:
            max_len_sents = len(sent_vec)

        X.append(sent_vec)

        if label not in labels:
            labels[label] = len(labels)
        Y.append(labels[label])

    f.close()

    # for sent_vec in X:
    #     for word_vec in sent_vec:
    #         pad = [PAD_VEC for _ in range(max_len_words - len(word_vec))]
    #         word_vec.extend(pad)
    #
    #     pad = [[PAD_VEC for _ in range(max_len_words)] for _ in range(max_len_sents - len(sent_vec))]
    #     sent_vec.extend(pad)

    return X, Y, labels


def sent_splitter(text):
    parenthesis = u'（）「」『』【】［］〈〉《》〔〕｛｝””'
    close2open = dict(zip(parenthesis[1::2], parenthesis[0::2]))
    paren_chars = set(parenthesis)
    delimiters = set(u'。．？！.?!\n\r')
    pstack = []
    buff = []

    ret = []

    for i, c in enumerate(text):
        c_next = None
        if i + 1 < len(text):
            c_next = text[i + 1]

        # check correspondence of parenthesis
        if c in paren_chars:
            # close
            if c in close2open:
                if len(pstack) > 0 and pstack[-1] == close2open[c]:
                    pstack.pop()
            # open
            else:
                pstack.append(c)

        buff.append(c)
        if c in delimiters:
            if len(pstack) == 0 and c_next not in delimiters:
                ret.append(u''.join(buff).strip())
                buff = []

    if len(buff) > 0:
        ret.append(u''.join(buff).strip())

    return ret


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
    # for batch1, batch2 in zip(gen1, gen2):
    #     for x in sorted(zip(batch1, batch2), key=lambda x: len(x[order])):
    #         yield x
    for batch1, batch2 in zip(gen1, gen2):
        for x1 in sorted(zip(batch1, batch2), key=lambda x1: (len(x1[order]), max([len(x2) for x2 in x1[order]]))):
            yield x1


def fill_batch(batch, padding, min_height=1):
    # max_len = max([len(x) for x in batch] + [min_height])
    # return [x + [padding] * (max_len - len(x) + 1) for x in batch]
    max_len_sents = max([len(x) for x in batch] + [min_height])
    max_len_words = max([max(len(x1) for x1 in x2) for x2 in batch] + [min_height])
    for sents in batch:
        for words in sents:
            vec = [padding] * (max_len_words - len(words))
            words.extend(vec)
        vec = [[padding] * max_len_words] * (max_len_sents - len(sents))
        sents.extend(vec)
    return batch


class MyHATT(Chain):
    def __init__(self, n_dim, n_label):
        super(MyHATT, self).__init__(
            w_fc0=L.Linear(100, 100),
            w_fwd=L.GRU(50, n_inputs=n_dim),
            w_bwd=L.GRU(50, n_inputs=n_dim),
            w_fc1=L.Linear(100, 100),
            s_fc0=L.Linear(100, 100),
            s_fwd=L.GRU(50, n_inputs=100),
            s_bwd=L.GRU(50, n_inputs=100),
            s_fc1=L.Linear(100, 100),
            s_fc2=L.Linear(100, n_label),
        )
        self.uw = (xp.random.rand(1, 100).astype(np.float32) - 0.5) / 100
        self.us = (xp.random.rand(1, 100).astype(np.float32) - 0.5) / 100

    def __call__(self, x, t, train=True):

        y = self.forward(x, train=train)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def forward(self, x, train=True):

        N      = x.shape[0]     # 50
        height = x.shape[1]     # 4
        width  = x.shape[2]     # 49
        dim    = x.shape[3]     # 300

        # x:   50x4x49x300

        # uw: 1x100
        # us: 1x100
        uw = F.tanh(self.w_fc0(Variable(self.uw, volatile=not train)))
        us = F.tanh(self.s_fc0(Variable(self.us, volatile=not train)))

        # x:  (50x4)x49x300
        x = F.reshape(x, (height * N, width, dim))

        hit_fwd_list = []   # h: (50x4)x50  x49
        hit_fwd = Variable(xp.zeros((x.shape[0], 50), dtype=np.float32), volatile=not train)
        for t in six.moves.range(x.shape[1]):
            hit_fwd = self.w_fwd(hit_fwd, x[:, t, :])
            hit_fwd_list.append(hit_fwd)

        hit_bwd_list = []   # h: (50x4)x50  x49
        hit_bwd = Variable(xp.zeros((x.shape[0], 50), dtype=np.float32), volatile=not train)
        for t in reversed(six.moves.range(x.shape[1])):
            hit_bwd = self.w_bwd(hit_bwd, x[:, t, :])
            hit_bwd_list.append(hit_bwd)

        hit_list = []       # h: (50x4)x100  x49
        for t in six.moves.range(x.shape[1]):
            hi = F.concat((hit_fwd_list[t], hit_bwd_list[t]), axis=1)
            hit_list.append(hi)

        uit_list = []       # u: (50x4)x100  x49
        for t in six.moves.range(x.shape[1]):
            ui = F.tanh(self.w_fc1(hit_list[t]))
            uit_list.append(ui)

        simit_list = []     # sim: (50x4)  x49
        for t in six.moves.range(x.shape[1]):
            simit = F.batch_matmul(F.broadcast_to(uw, (height * N, 100)), uit_list[t], transa=True)
            simit_list.append(F.reshape(simit, (x.shape[0], 1)))

        # alpah: (50x4)x49
        ait_mat = F.softmax(F.concat(simit_list, axis=1))

        ait_list = []       # a: (50x4)  x49
        for t in six.moves.range(x.shape[1]):
            ait = F.reshape(ait_mat[:, t], (x.shape[0], 1))
            ait_list.append(ait)

        # s: (50x4)x100
        si = Variable(xp.zeros((x.shape[0], 100), dtype=np.float32), volatile=not train)
        for t in six.moves.range(x.shape[1]):
            si += F.reshape(F.batch_matmul(hit_list[t], ait_list[t]), (x.shape[0], 100))

        # s:  50x4x100
        si = F.reshape(si, (N, height, 100))

        hi_fwd_list = []    # h: 50x50  x4
        hi_fwd = Variable(xp.zeros((N, 50), dtype=np.float32), volatile=not train)
        for i in six.moves.range(si.shape[1]):
            hi_fwd = self.s_fwd(hi_fwd, si[:, i, :])
            hi_fwd_list.append(hi_fwd)

        hi_bwd_list = []    # h: 50x50  x4
        hi_bwd = Variable(xp.zeros((N, 50), dtype=np.float32), volatile=not train)
        for i in reversed(six.moves.range(si.shape[1])):
            hi_bwd = self.s_bwd(hi_bwd, si[:, i, :])
            hi_bwd_list.append(hi_bwd)

        hi_list = []        # h: 50x100  x4
        for i in six.moves.range(si.shape[1]):
            hi = F.concat((hi_fwd_list[i], hi_bwd_list[i]), axis=1)
            hi_list.append(hi)

        ui_list = []        # u: 50x100  x4
        for i in six.moves.range(si.shape[1]):
            ui = F.tanh(self.s_fc1(hi_list[i]))
            ui_list.append(ui)

        simi_list = []      # sim: 50x1  x4
        for i in six.moves.range(si.shape[1]):
            simi = F.batch_matmul(F.broadcast_to(us, (N, 100)), ui_list[i], transa=True)
            simi_list.append(F.reshape(simi, (si.shape[0], 1)))

        # alpah: 50x1
        ai_mat = F.softmax(F.concat(simi_list, axis=1))

        ai_list = []        # a: 50x1  x4
        for i in six.moves.range(si.shape[1]):
            ai = F.reshape(ai_mat[:, i], (si.shape[0], 1))
            ai_list.append(ai)

        # s: 50x100
        s = Variable(xp.zeros((si.shape[0], 100), dtype=np.float32), volatile=not train)
        for i in six.moves.range(si.shape[1]):
            s += F.reshape(F.batch_matmul(hi_list[i], ai_list[i]), (si.shape[0], 100))

        y = self.s_fc2(s)

        return y


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: MyHATT')
    parser.add_argument('--train',           default='',  type=unicode, help='training file (.txt)')
    parser.add_argument('--test',            default='',  type=unicode, help='evaluating file (.txt)')
    parser.add_argument('--w2v',       '-w', default='',  type=unicode, help='word2vec model file (.bin)')
    parser.add_argument('--gpu',       '-g', default=-1,  type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch',     '-e', default=25,  type=int, help='number of epochs to learn')
    parser.add_argument('--unit',      '-u', default=100, type=int, help='number of output channels')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning batchsize size')
    parser.add_argument('--output',    '-o', default='model-gru_hatt-w2v-sort',  type=str, help='output directory')
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

    print('# loading word2vec model: {}'.format(args.w2v))
    sys.stdout.flush()
    w2v = load_w2v_model(args.w2v)

    # データの読み込み
    if not args.test:
        # トレーニング+テストデータ
        X, y, labels = load_data(args.train, w2v=w2v)

        # トレーニングデータとテストデータに分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    else:
        # トレーニングデータ
        X_train, y_train, labels = load_data(args.train, w2v=w2v)

        # テストデータ
        X_test, y_test, labels = load_data(args.test, w2v=w2v, labels=labels)

    n_dim   = w2v.vector_size
    n_vocab = len(w2v.vocab)
    n_label = len(labels)
    height  = len(X_train[0])
    width   = len(X_train[0][0])

    N = len(X_train)
    N_test = len(X_test)

    print('# gpu: {}'.format(args.gpu))
    print('# embedding dim: {}, vocab {}'.format(n_dim, n_vocab))
    print('# epoch: {}'.format(n_epoch))
    print('# batchsize: {}'.format(batchsize))
    print('# input channel: {}'.format(1))
    print('# output channel: {}'.format(n_units))
    print('# train: {}, test: {}'.format(N, N_test))
    print('# data min height: {}, width: {}, labels: {}'.format(height, width, n_label))
    sys.stdout.flush()

    # Prepare LSTM model
    model = MyHATT(n_dim, n_label)

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
            x_batch = fill_batch(x_batch, PAD_VEC, min_height=5)

            # N 個の順番をランダムに並び替える
            perm = np.random.permutation(len(x_batch))
            x = Variable(xp.asarray(x_batch, dtype=np.float32)[perm], volatile='off')
            t = Variable(xp.asarray(t_batch, dtype=np.int32)[perm],   volatile='off')

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
            x_batch = fill_batch(x_batch, PAD_VEC, min_height=5)

            x = Variable(xp.asarray(x_batch, dtype=np.float32), volatile='on')
            t = Variable(xp.asarray(t_batch, dtype=np.int32),   volatile='on')

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
