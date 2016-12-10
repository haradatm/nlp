#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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

import os, time
start_time = time.time()

import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
np.set_printoptions(precision=20)

import chainer
from chainer import cuda, Function, gradient_check, Variable
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainer.optimizer
import collections
from nltk.translate import bleu_score

xp = np
BOS_TOKEN = u'<s>'
BOS_ID = 0
EOS_TOKEN = u'</s>'
EOS_ID = 1


def load_comparable_data(path, vocab=None, inv_vocab=None):
    global BOS_ID, EOS_ID

    x_data, t_data = [], []

    if not vocab:
        vocab = {}
        inv_vocab = {}
        vocab[BOS_TOKEN] = BOS_ID
        vocab[EOS_TOKEN] = EOS_ID
        inv_vocab[BOS_ID] = BOS_TOKEN
        inv_vocab[EOS_ID] = EOS_TOKEN
    else:
        if not inv_vocab:
            for token, id in vocab.items():
                inv_vocab[id] = token

        if BOS_TOKEN not in vocab:
            BOS_ID = len(vocab)
            vocab[BOS_TOKEN] = BOS_ID
            inv_vocab[id] = BOS_TOKEN
        else:
            BOS_ID = vocab[BOS_TOKEN]

        if EOS_TOKEN not in vocab:
            EOS_ID = len(vocab)
            vocab[EOS_TOKEN] = EOS_ID
            inv_vocab[id] = EOS_TOKEN
        else:
            EOS_ID = vocab[EOS_TOKEN]

    f = open(path, 'rU')
    for i, line in enumerate(f):
        # if i > 1000:
        #     break

        x_line, t_line = unicode(line).strip().split(u'\t')

        tokens = x_line.split(u' ')
        tokens.insert(0, BOS_TOKEN)
        tokens.append(EOS_TOKEN)
        vec = []
        for token in tokens:
            if token not in vocab:
                id = len(vocab)
                vocab[token] = id
                inv_vocab[id] = token
            vec.append(vocab[token])
        x_data.append(vec)

        tokens = t_line.split(u' ')
        vec = []
        for token in tokens:
            if token not in vocab:
                id = len(vocab)
                vocab[token] = id
                inv_vocab[id] = token
            vec.append(vocab[token])
        t_data.append(vec)

    f.close()

    return x_data, t_data, vocab, inv_vocab


def batch(generator, batch_size):
    batch = []
    is_tuple = False
    for line in generator:
        is_tuple = isinstance(line, tuple)
        batch.append(line)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch


def sorted_parallel(generator1, generator2, pooling, order=1):
    gen1 = batch(generator1, pooling)
    gen2 = batch(generator2, pooling)
    for batch1, batch2 in zip(gen1, gen2):
        for x in sorted(zip(batch1, batch2), key=lambda x: len(x[order])):
            yield x


def fill_batch(batch):
    max_len = max(len(x) for x in batch)
    return [x + [EOS_ID] * (max_len - len(x) + 1) for x in batch]


class MyATT(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(MyATT, self).__init__(
            embedx=L.EmbedID(vocab_size, embed_size),
            embedy=L.EmbedID(vocab_size, embed_size),
            w1=L.Linear(embed_size, hidden_size),
            h1=L.StatelessLSTM(hidden_size, hidden_size),
            w2=L.Linear(embed_size, hidden_size),
            h2=L.StatelessLSTM(hidden_size, hidden_size),
            w3=L.Linear(hidden_size, hidden_size),
            w4=L.Linear(hidden_size, hidden_size),
            w5=L.Linear(embed_size, hidden_size),

            aw=L.Linear(hidden_size, hidden_size),
            bw=L.Linear(hidden_size, hidden_size),
            pw=L.Linear(hidden_size, hidden_size),
            ew=L.Linear(hidden_size, 1),

            h3=L.StatelessLSTM(hidden_size, hidden_size),
            w6=L.Linear(hidden_size, embed_size),
            w7=L.Linear(hidden_size, embed_size),
            wf=L.Linear(embed_size, vocab_size)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

    def __call__(self, x_batch, t_batch, train=True):

        batch_size = len(x_batch)
        hidden_size = self.hidden_size
        x_len = len(x_batch[0])
        t_len = len(t_batch[0])

        x = xp.asarray(x_batch, dtype=np.int32).reshape((batch_size, x_len))
        t = xp.asarray(t_batch, dtype=np.int32).reshape((batch_size, t_len))

        # encoder
        c = Variable(xp.zeros((batch_size, hidden_size), dtype=np.float32), volatile=not train)
        h = Variable(xp.zeros((batch_size, hidden_size), dtype=np.float32), volatile=not train)
        fw_list = []
        for i in xrange(x_len):
            x_k = self.embedx(Variable(x[:,  i], volatile=not train))
            c, h = self.h1(c, h, self.w1(x_k))
            fw_list.append(h)

        c = Variable(xp.zeros((batch_size, hidden_size), dtype=np.float32), volatile=not train)
        h = Variable(xp.zeros((batch_size, hidden_size), dtype=np.float32), volatile=not train)
        bw_list = []
        for i in xrange(x_len):
            x_k = self.embedx(Variable(x[:, -i], volatile=not train))
            c, h = self.h2(c, h, self.w2(x_k))
            bw_list.insert(0, h)

        gh_list = []
        for i in xrange(x_len):
            h = F.tanh(self.w3(fw_list[i]) + self.w4(bw_list[i]))
            gh_list.append(h)

        # decoder
        output = np.zeros(t.shape)
        accum_loss = 0.

        c = Variable(xp.zeros((batch_size, hidden_size), dtype=np.float32), volatile=not train)
        h = Variable(xp.zeros((batch_size, hidden_size), dtype=np.float32), volatile=not train)
        for i in xrange(t_len):

            if i == 0:
                y = Variable(xp.asarray([BOS_ID] * batch_size, dtype=np.int32), volatile=not train)
            else:
                if train:
                    y = Variable(t[:, i - 1], volatile=not train)
                else:
                    y = (y.data.argmax(1)).astype(np.int32)
            e = F.tanh(self.embedy(y))
            c, h = self.h3(c, h, self.w5(e))

            # ct = mk_ct(gh_list, h, train=train)

            exp_list = []
            sum_e = Variable(xp.zeros((batch_size, 1), dtype=np.float32), volatile=not train)
            for j in xrange(len(fw_list)):

                # # score: dot
                # # w = F.sum(F.batch_matmul(h, gh_list[j], transa=True), axis=1)
                # w = F.reshape(F.batch_matmul(h, gh_list[j], transa=True), (batch_size, 1))
                # e = F.exp(w)

                # score: concat
                w = F.tanh(self.aw(fw_list[i]) + self.bw(bw_list[i]) + self.pw(h))
                e = F.exp(self.ew(w))

                exp_list.append(e)
                sum_e += e

            alpha_list = []
            for j in xrange(len(exp_list)):
                alpha_list.append(exp_list[j] / sum_e)

            ct = Variable(xp.zeros((batch_size, hidden_size), dtype=np.float32), volatile=not train)
            for j in xrange(len(alpha_list)):

                # ct += F.broadcast_to(alpha_list[j], (batch_size, hidden_size)) * gh_list[j]
                ct += F.reshape(F.batch_matmul(gh_list[j], alpha_list[j]), (batch_size, hidden_size))

            y = self.wf(F.tanh(self.w6(ct) + self.w7(h)))
            output[:, i] = cuda.to_cpu(y.data.argmax(1))
            yt = Variable(t[:, i], volatile=not train)
            accum_loss += F.softmax_cross_entropy(y, yt)

        return output, accum_loss


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: Attention model')
    parser.add_argument('--train',           default='train.txt', type=unicode, help='training file (.txt)')
    parser.add_argument('--test',            default='test.txt',  type=unicode, help='evaluating file (.txt)')
    parser.add_argument('--gpu',       '-g', default=-1,  type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch',     '-e', default=100, type=int, help='number of epochs to learn')
    parser.add_argument('--dim',       '-d', default=100, type=int, help='number of dimensions')
    parser.add_argument('--unit',      '-u', default=200, type=int, help='number of units')
    parser.add_argument('--batchsize', '-b', default=10,  type=int, help='minibatch size')
    parser.add_argument('--output',    '-o', default='model', type=str, help='output directory')
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np
    xp.random.seed(123)

    # 学習の繰り返し回数
    n_epoch = args.epoch

    # 次元数
    n_dim = args.dim

    # 中間層の数
    n_units = args.unit

    # ミニバッチサイズ
    minibatch = args.batchsize

    model_dir = '.'

    x_train, t_train, vocab, inv_vocab = load_comparable_data(args.train)
    x_test,  t_test,  vocab, inv_vocab = load_comparable_data(args.test,  vocab=vocab, inv_vocab=inv_vocab)

    N = len(x_train)
    N_test = len(x_test)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    with open(os.path.join(model_dir, 'vocab.bin'), 'wb') as f:
        pickle.dump(vocab, f)

    model = MyATT(len(vocab), n_dim, n_units)
    if args.gpu >= 0:
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    # optimizer = optimizers.AdaGrad(lr=0.01)
    # optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.GradientClipping(10.0))

    start_at = time.time()
    cur_at = start_at

    # プロット用に実行結果を保存する
    train_loss = []
    train_norm = []
    train_bleu = []
    train_correct = []
    test_loss = []
    test_bleu = []
    test_correct = []

    for epoch in xrange(n_epoch):
        sorted_gen = batch(sorted_parallel(x_train, t_train, N * minibatch), minibatch)
        accum_loss = 0.
        accum_bleu = 0.
        accum_correct = 0.
        trained = 0

        for x_batch, t_batch in sorted_gen:
            x_batch = fill_batch(x_batch)
            t_batch = fill_batch(t_batch)
            K = len(x_batch)

            # 勾配を初期化
            # model.h1.reset_state()
            # model.h2.reset_state()
            # model.h3.reset_state()
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            h_batch, loss = model(x_batch, t_batch, train=True)

            # 誤差逆伝播で勾配を計算
            loss.backward()
            # loss.unchain_backward()
            optimizer.update()

            accum_loss += loss
            for k in range(K):
                reference = [inv_vocab[x] if x != EOS_ID else '*' for x in t_batch[k]]
                hypothesis = [inv_vocab[x] if x != EOS_ID else '*' for x in h_batch[k]]
                accum_bleu += bleu_score.bleu([reference], hypothesis, [0.25, 0.25, 0.25, 0.25])
                if reference == hypothesis:
                    accum_correct += 1

            for k in range(K):
                print(u'epoch {:3d}/{:3d}, train sample {:8d}'.format(epoch + 1, 100, trained + k + 1))
                print(u'  src = ' + ' '.join([inv_vocab[x] if x != EOS_ID else '*' for x in x_batch[k]]))
                print(u'  trg = ' + ' '.join([inv_vocab[x] if x != EOS_ID else '*' for x in t_batch[k]]))
                print(u'  hyp = ' + ' '.join([inv_vocab[x] if x != EOS_ID else '*' for x in h_batch[k]]))
                sys.stdout.flush()

            trained += K

            train_loss.append(float(accum_loss.data) / trained)
            train_norm.append(optimizer.compute_grads_norm())
            train_bleu.append(accum_bleu / trained)
            train_correct.append(accum_correct / trained)

        # 訓練データの誤差と,正解精度を表示
        now = time.time()
        throuput = now - cur_at
        print(u'epoch {:3d}, train loss: {:.2f}, BLUE: {:.2f}, correctness: {:.2f} ({:.2f} sec)'.format(epoch + 1, float(accum_loss.data) / trained, accum_bleu / trained, accum_correct / trained, throuput))
        sys.stdout.flush()
        cur_at = now

        # evaluation
        # テストデータで誤差と正解精度を算出し汎化性能を確認
        sorted_gen = batch(sorted_parallel(x_test, t_test, N_test * minibatch), minibatch)
        accum_loss = 0.
        accum_bleu = 0.
        accum_correct = 0.
        trained = 0

        for x_batch, t_batch in sorted_gen:
            x_batch = fill_batch(x_batch)
            t_batch = fill_batch(t_batch)
            K = len(x_batch)

            # 勾配を初期化
            # model.h1.reset_state()
            # model.h2.reset_state()
            # model.h3.reset_state()
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            h_batch, loss = model(x_batch, t_batch, train=False)

            accum_loss += loss
            for k in range(K):
                reference = [inv_vocab[x] if x != EOS_ID else '*' for x in t_batch[k]]
                hypothesis = [inv_vocab[x] if x != EOS_ID else '*' for x in h_batch[k]]
                accum_bleu += bleu_score.bleu([reference], hypothesis, [0.25, 0.25, 0.25, 0.25])
                if reference == hypothesis:
                    accum_correct += 1

            for k in range(K):
                print(u'epoch {:3d}/{:3d}, test  sample {:8d}'.format(epoch + 1, 100, trained + k + 1))
                print(u'  src = ' + ' '.join([inv_vocab[x] if x != EOS_ID else '*' for x in x_batch[k]]))
                print(u'  trg = ' + ' '.join([inv_vocab[x] if x != EOS_ID else '*' for x in t_batch[k]]))
                print(u'  hyp = ' + ' '.join([inv_vocab[x] if x != EOS_ID else '*' for x in h_batch[k]]))
                sys.stdout.flush()

            trained += K

            test_loss.append(float(accum_loss.data) / trained)
            test_bleu.append(accum_bleu / trained)
            test_correct.append(accum_correct / trained)

        # テストデータの誤差と,正解精度を表示
        now = time.time()
        throuput = now - cur_at
        print(u'epoch {:3d}, test  loss: {:.2f}, BLUE: {:.2f}, correctness: {:.2f} ({:.2f} sec)'.format(epoch + 1, float(accum_loss.data) / trained, accum_bleu / trained, accum_correct / trained, throuput))
        sys.stdout.flush()
        cur_at = now

        # 精度と誤差をグラフ描画
        if True:
            ylim1 = [min(train_loss + train_norm + test_loss), max(train_loss + train_norm + test_loss)]
            ylim2 = [min(train_bleu + test_bleu + train_correct + test_correct), max(train_bleu + test_bleu + train_correct + test_correct)]

            # グラフ左
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.ylim(ylim1)
            plt.plot(range(1, len(train_loss) + 1), train_loss, 'b')
            plt.plot(range(1, len(train_norm) + 1), train_norm, 'g')
            plt.grid()
            plt.ylabel('Loss and L2-norm')
            plt.legend(['train loss', 'train l2-norm'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(train_bleu) + 1),    train_bleu,    'r')
            plt.plot(range(1, len(train_correct) + 1), train_correct, 'm')
            plt.grid()
            # plt.ylabel('BLEU')
            plt.legend(['train bleu', 'train correctness'], loc="upper left")
            plt.title('Loss and accuracy of training.')

            # グラフ右
            plt.subplot(1, 2, 2)
            plt.ylim(ylim1)
            plt.plot(range(1, len(test_loss) + 1), test_loss, 'b')
            plt.grid()
            # plt.ylabel('Loss and L2-norm')
            plt.legend(['test loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(test_bleu) + 1),    test_bleu,    'r')
            plt.plot(range(1, len(test_correct) + 1), test_correct, 'm')
            plt.grid()
            plt.ylabel('BLEU')
            plt.legend(['test bleu', 'test correctness'], loc="upper left")
            plt.title('Loss and accuracy of test.')

            plt.savefig('train_attention-concat.png'.format(epoch + 1))
            # plt.show()

    if args.gpu >= 0: model.to_cpu()
    with open(os.path.join(model_dir, 'final.model'), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(model_dir, 'final.state'), 'wb') as f:
        pickle.dump(optimizer, f)

    print('time spent:', time.time() - start_time)
