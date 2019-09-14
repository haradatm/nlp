#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, os, time, logging
import numpy as np
np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def pp(obj):
    import pprint
    pp = pprint.PrettyPrinter(indent=1, width=160)
    print(pp.pformat(obj))
    sys.stdout.flush()


start_time = time.time()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pickle as pickle
import matplotlib.pyplot as plt

from chainer import cuda, Chain, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
import chainer.optimizer

import collections
from six.moves import xrange

xp = np
# BOS_TOKEN = '<s>'
# EOS_TOKEN = '</s>'
# UNK_TOKEN = '<unk>'
# PAD_TOKEN = '<pad>'


def load_data(path):
    X, y = [], []

    data = collections.defaultdict(lambda: [[], []])

    f = open(path, 'r')
    for i, line in enumerate(f):
        # if i >= 10:
        #     break

        line = line.strip()
        if line == '':
            continue

        row = line.split(' ')[:48]
        if len(row) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        r = int(row[0])
        q = int(row[1][4:])
        v = [float(x.split(':')[1]) for x in row[2:]]

        data[q][0].append(v)
        data[q][1].append(r)

    f.close()

    for d in data.values():
        X.append(d[0])
        y.append(d[1])

    return X, y


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
    max_len = max(map(len, batch) + [min_height])
    return [x + [padding] * (max_len - len(x)) for x in batch]


def permutation_probability_loss(x, t):

    # permutation probability distribution
    log_ppd_x = x - F.broadcast_to(F.expand_dims(F.logsumexp(x, axis=1), 1), x.shape)
    log_ppd_t = t - F.broadcast_to(F.expand_dims(F.logsumexp(t, axis=1), 1), t.shape)
    ppd_t = F.softmax(t, axis=1)

    # loss normalized over all instances
    loss = (ppd_t * log_ppd_t) - (ppd_t * log_ppd_x)

    return F.sum(loss) / float(x.shape[0])


def n_discount_cumulative_count(probs, labels, k):

    num_queries = len(probs)
    out = 0.

    for i in xrange(num_queries):
        argsort_indices = probs[i].argsort()[::-1]
        candidates = labels[i, argsort_indices]

        dcg = 0
        # for i in xrange(min(k, len(candidates))):
        #     dcg += (2 ** candidates[j] - 1.) / np.log2(i + 2)

        for j in xrange(1, min(k, len(candidates))):
            dcg += candidates[j] / np.log2(j + 1)

        argsort_indices = labels[i].argsort()[::-1]
        candidates = labels[i, argsort_indices]

        ideal_dcg = 0
        # for i in xrange(min(k, len(candidates))):
        #     ideal_dcg += (2 ** candidates[j] - 1.) / np.log2(i + 2)

        for j in xrange(1, min(k, len(candidates))):
            ideal_dcg += candidates[j] / np.log2(j + 1)

        ndcg = 0.
        if ideal_dcg != 0. and dcg != 0.:
            ndcg = dcg / ideal_dcg
        out += ndcg

    return out / float(num_queries)


def mean_average_precision(probs, labels, k):

    num_queries = len(probs)
    out = 0.

    for i in xrange(num_queries):
        argsort_indices = probs[i].argsort()[::-1]
        candidates = labels[i, argsort_indices]
        num_correct = 0.
        precisions = []
        for j in xrange(min(k, len(candidates))):
            if int(candidates[j]) >= 1:
                num_correct += 1
                precisions.append(num_correct / (j + 1))

        avg_prec = 0.
        if len(precisions) > 0:
            avg_prec = sum(precisions) / len(precisions)
        out += avg_prec

    return out / float(num_queries)


class RankNet(Chain):
    def __init__(self, n_dim, n_units):
        super(RankNet, self).__init__(
            l1=L.Linear(n_dim, n_units, initialW=I.GlorotUniform()),
            l2=L.Linear(n_units, n_units, initialW=I.GlorotUniform()),
            l3=L.Linear(n_units, 1, initialW=I.GlorotUniform(), nobias=True)
        )

    def __call__(self, x, t, train=True):
        """
        RankNet
          See: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
        """

        i, j = np.random.randint(x.shape[1], size=2)

        x_i = x[:, i:i+1, :]
        x_j = x[:, j:j+1, :]
        s_i = F.reshape(self.forward(x_i, train=train), (x.shape[0],))
        s_j = F.reshape(self.forward(x_j, train=train), (x.shape[0],))
        # s_d = F.sigmoid(s_i - s_j)
        s_d = s_i - s_j

        t_i = t[:, i]
        t_j = t[:, j]
        t_d = t_i - t_j
        s_ij = np.zeros(t_d.shape, dtype=np.int32)
        s_ij += np.where(t_d.data > 0,  1, 0)
        s_ij += np.where(t_d.data < 0, -1, 0)

        loss = (1 - s_ij) * s_d / 2. + F.log(1 + F.exp(-s_d))

        return F.sum(loss) / float(x.shape[0])

    def forward(self, x, train=True):
        N      = x.shape[0]  # 100
        height = x.shape[1]  # 40
        width  = x.shape[2]  # 46

        x = F.reshape(x, (N * height, width))

        h1 = F.relu(self.l1(x))
        # h1 = F.dropout(h1, ratio=0.5, train=train)

        h2 = F.relu(self.l2(h1))
        # h2 = F.dropout(h2, ratio=0.5, train=train)

        h3 = self.l3(h2)

        return F.reshape(h3, (N, height))

    def eval(self, x, t):
        y = self.forward(x, train='on')

        acc_map = mean_average_precision(y.data, t.data, k=10)
        acc_ndcg = n_discount_cumulative_count(y.data, t.data, k=10)

        return acc_map, acc_ndcg


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: RankNet')
    parser.add_argument('--train',           default='',   type=unicode, help='training data file (.txt)')
    parser.add_argument('--valid',           default='',   type=unicode, help='validating data file (.txt)')
    parser.add_argument('--test',            default='',   type=unicode, help='testing data file (.txt)')
    # parser.add_argument('--w2v',       '-w', default='',   type=unicode, help='word2vec model file (.bin)')
    parser.add_argument('--gpu',       '-g', default=-1,   type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch',     '-e', default=1000, type=int, help='number of epochs to learn')
    parser.add_argument('--unit',      '-u', default=200,  type=int, help='number of output channels')
    parser.add_argument('--batchsize', '-b', default=1,    type=int, help='learning batchsize size')
    parser.add_argument('--output',    '-o', default='model', type=str, help='output directory')
    # args = parser.parse_args([
    #     '--train', 'MQ2007/Fold1/train.txt',
    #     '--valid', 'MQ2007/Fold1/vali.txt',
    #     '--test',  'MQ2007/Fold1/test.txt'
    # ])
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device_from_id(args.gpu).use()

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
    output_channel = 50

    # データの読み込み
    if not args.valid:
        # トレーニング+テストデータ
        X, y = load_data(args.train)

        # トレーニングデータとテストデータに分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    else:
        # トレーニングデータ
        X_train, y_train = load_data(args.train)

        # 検証データ
        X_test, y_test = load_data(args.valid)

    n_dim   = len(X_train[0][0])
    # n_vocab = len(vocab)
    n_label = 3
    height  = len(X_train[0])
    width   = n_dim

    N = len(X_train)
    N_test = len(X_test)

    print('# gpu: {}'.format(args.gpu))
    # print('# embedding dim: {}, vocab {}'.format(n_dim, n_vocab))
    print('# epoch: {}'.format(n_epoch))
    print('# batchsize: {}'.format(batchsize))
    # print('# input channel: {}'.format(1))
    # print('# output channel: {}'.format(n_units))
    print('# train: {}, test: {}'.format(N, N_test))
    print('# data height: {}, width: {}, labels: {}'.format(height, width, n_label))
    sys.stdout.flush()

    # Prepare RankNet
    model = RankNet(n_dim, n_units)

    if args.gpu >= 0:
        model.to_gpu()

    # 重み減衰
    # decay = 0.0001
    decay = 0.0005

    # 勾配上限
    grad_clip = 3

    # 学習率の減衰
    lr_decay = 0.995

    # Setup optimizer (Optimizer の設定)
    optimizer = optimizers.Adam(alpha=0.0007)
    # optimizer = optimizers.AdaDelta()
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

    # プロット用に実行結果を保存する
    train_loss = []
    train_norm = []
    train_accuracy1 = []
    train_accuracy2 = []
    test_loss = []
    test_accuracy1 = []
    test_accuracy2 = []

    min_loss = float('inf')
    min_epoch = 0

    start_at = time.time()
    cur_at = start_at

    # Learning loop
    for epoch in xrange(1, n_epoch + 1):

        # print('epoch {:} / {:}'.format(epoch, n_epoch))
        # sys.stdout.flush()

        sorted_gen = batch_tuple(sorted_parallel(X_train, y_train, 100 * batchsize), batchsize)
        # sorted_gen = batch_tuple(sorted_parallel(X_train, y_train, 1), 1)
        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        # training
        for x_batch, t_batch in sorted_gen:
            x_batch = fill_batch(x_batch, [0.] * width)
            t_batch = fill_batch(t_batch, 0)

            # N 個の順番をランダムに並び替える
            perm = np.random.permutation(len(x_batch))
            x = Variable(xp.asarray(x_batch, dtype=np.float32)[perm], volatile='off')
            t = Variable(xp.asarray(t_batch, dtype=np.float32)[perm], volatile='off')

            # 勾配を初期化
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            loss = model(x, t, train=True)
            acc1, acc2 = model.eval(x, t)

            sum_train_loss += float(loss.data) * len(t_batch)
            sum_train_accuracy1 += float(acc1) * len(t_batch)
            sum_train_accuracy2 += float(acc2) * len(t_batch)
            K += len(t_batch)

            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

        # 訓練データの誤差と,正解精度を表示
        mean_train_loss = sum_train_loss / K
        mean_train_accuracy1 = sum_train_accuracy1 / K
        mean_train_accuracy2 = sum_train_accuracy2 / K
        train_loss.append(mean_train_loss)
        train_accuracy1.append(mean_train_accuracy1)
        train_accuracy2.append(mean_train_accuracy2)
        # norm = optimizer.compute_grads_norm()
        now = time.time()
        train_throughput = now - cur_at
        cur_at = now

        # sorted_gen = batch_tuple(sorted_parallel(X_test, y_test, 100 * batchsize), batchsize)
        sorted_gen = batch_tuple(sorted_parallel(X_test, y_test, 1), 1)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        # evaluation
        for x_batch, t_batch in sorted_gen:
            x_batch = fill_batch(x_batch, [0.] * width)
            t_batch = fill_batch(t_batch, 0)

            x = Variable(xp.asarray(x_batch, dtype=np.float32), volatile='on')
            t = Variable(xp.asarray(t_batch, dtype=np.float32), volatile='on')

            # 順伝播させて誤差と精度を算出
            loss = model(x, t, train=False)
            acc1, acc2 = model.eval(x, t)

            sum_test_loss += float(loss.data) * len(t_batch)
            sum_test_accuracy1 += float(acc1) * len(t_batch)
            sum_test_accuracy2 += float(acc2) * len(t_batch)
            K += len(t_batch)

        # テストデータでの誤差と正解精度を表示
        mean_test_loss = sum_test_loss / K
        mean_test_accuracy1 = sum_test_accuracy1 / K
        mean_test_accuracy2 = sum_test_accuracy2 / K
        test_loss.append(mean_test_loss)
        test_accuracy1.append(mean_test_accuracy1)
        test_accuracy2.append(mean_test_accuracy2)
        now = time.time()
        test_throughput = now - cur_at
        cur_at = now

        print('[{:>3d}] T/loss={:.6f} T/map={:.6f} T/ndcg@10={:.6f} T/sec={:.6f} D/loss={:.6f} D/map={:.6f} D/ndcg@10={:.6f} D/sec={:.6f} lr={:.6f}'.format(epoch, mean_train_loss, mean_train_accuracy1, mean_train_accuracy2, train_throughput, mean_test_loss, mean_test_accuracy1, mean_test_accuracy2, test_throughput, optimizer.alpha))
        sys.stdout.flush()

        # model と optimizer を保存する
        if float(loss.data) < min_loss:
            min_loss = float(loss.data)
            min_epoch = epoch
            if args.gpu >= 0: model.to_cpu()
            serializers.save_npz(os.path.join(model_dir, 'early_stopped.model'), model)
            serializers.save_npz(os.path.join(model_dir, 'early_stopped.state'), optimizer)
            if args.gpu >= 0: model.to_gpu()

        optimizer.alpha *= lr_decay

        # 精度と誤差をグラフ描画
        if True:
            ylim1 = [min(train_loss + test_loss), max(train_loss + test_loss)]
            ylim2 = [min(train_accuracy1 + test_accuracy1 + train_accuracy2 + test_accuracy2), max(train_accuracy1 + test_accuracy1 + train_accuracy2 + test_accuracy2)]
            ylim1 = [0.0, 1.0]
            ylim2 = [0.6, 1.0]
            ylim2 = [0.4, 0.6]
            ylim2 = [0.6, 1.0]

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
            plt.plot(range(1, len(train_accuracy1) + 1), train_accuracy1, 'm')
            plt.plot(range(1, len(train_accuracy2) + 1), train_accuracy2, 'r')
            plt.grid()
            # plt.ylabel('accuracy')
            plt.legend(['train map', 'train ndcg'], loc="upper left")
            plt.title('Loss and accuracy of train.')

            # グラフ右
            plt.subplot(1, 2, 2)
            plt.ylim(ylim1)
            plt.plot(range(1, len(test_loss) + 1), test_loss, 'b')
            plt.grid()
            # plt.ylabel('loss')
            plt.legend(['dev loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(test_accuracy1) + 1), test_accuracy1, 'm')
            plt.plot(range(1, len(test_accuracy2) + 1), test_accuracy2, 'r')
            plt.grid()
            plt.ylabel('accuracy')
            plt.legend(['dev map', 'dev ndcg'], loc="upper left")
            plt.title('Loss and accuracy of dev.')

            plt.savefig('{}.png'.format(args.output))
            # plt.savefig('{}.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()

        cur_at = now

    # model と optimizer を保存する
    if args.gpu >= 0: model.to_cpu()
    serializers.save_npz(os.path.join(model_dir, 'final.model'), model)
    serializers.save_npz(os.path.join(model_dir, 'final.state'), optimizer)
    if args.gpu >= 0: model.to_gpu()

    # テスト
    if args.test:

        # テストデータ
        X_test, y_test = load_data(args.test)

        # sorted_gen = batch_tuple(sorted_parallel(X_test, y_test, 100 * batchsize), batchsize)
        sorted_gen = batch_tuple(sorted_parallel(X_test, y_test, 1), 1)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        for x_batch, t_batch in sorted_gen:
            x_batch = fill_batch(x_batch, [0.] * width)
            t_batch = fill_batch(t_batch, 0)

            x = Variable(xp.asarray(x_batch, dtype=np.float32), volatile='on')
            t = Variable(xp.asarray(t_batch, dtype=np.float32), volatile='on')

            # 順伝播させて誤差と精度を算出
            loss = model(x, t, train=False)
            acc1, acc2 = model.eval(x, t)

            sum_test_loss += float(loss.data) * len(t_batch)
            sum_test_accuracy1 += float(acc1) * len(t_batch)
            sum_test_accuracy2 += float(acc2) * len(t_batch)
            K += len(t_batch)

        test_loss.append(sum_test_loss / K)
        test_accuracy1.append(sum_test_accuracy1 / K)
        test_accuracy2.append(sum_test_accuracy2 / K)

        # テストデータでの誤差と正解精度を表示
        now = time.time()
        throuput = now - cur_at
        print('test/loss={:.6f} test/map={:.6f} test/ndcg@10={:.6f} test/sec={:.6f}'.format(sum_test_loss / K, sum_test_accuracy1 / K, sum_test_accuracy2 / K, throuput))
        sys.stdout.flush()
        cur_at = now

        print('loading early stopped-model at epoch {}'.format(min_epoch))
        serializers.load_npz(os.path.join(model_dir, 'early_stopped.model'), model)
        sys.stdout.flush()

        # sorted_gen = batch_tuple(sorted_parallel(X_test, y_test, 100 * batchsize), batchsize)
        sorted_gen = batch_tuple(sorted_parallel(X_test, y_test, 1), 1)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        for x_batch, t_batch in sorted_gen:
            x_batch = fill_batch(x_batch, [0.] * width)
            t_batch = fill_batch(t_batch, 0)

            x = Variable(xp.asarray(x_batch, dtype=np.float32), volatile='on')
            t = Variable(xp.asarray(t_batch, dtype=np.float32), volatile='on')

            # 順伝播させて誤差と精度を算出
            loss = model(x, t, train=False)
            acc1, acc2 = model.eval(x, t)

            sum_test_loss += float(loss.data) * len(t_batch)
            sum_test_accuracy1 += float(acc1) * len(t_batch)
            sum_test_accuracy2 += float(acc2) * len(t_batch)
            K += len(t_batch)

        test_loss.append(sum_test_loss / K)
        test_accuracy1.append(sum_test_accuracy1 / K)
        test_accuracy2.append(sum_test_accuracy2 / K)

        # テストデータでの誤差と正解精度を表示
        now = time.time()
        throuput = now - cur_at
        print('test/loss={:.6f} test/map={:.6f} test/ndcg@10={:.6f} test/sec={:.6f}'.format(sum_test_loss / K, sum_test_accuracy1 / K, sum_test_accuracy2 / K, throuput))
        sys.stdout.flush()
        cur_at = now

print('time spent:', time.time() - start_time)
