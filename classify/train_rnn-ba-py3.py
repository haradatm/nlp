#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: Bi-LSTM Neural Networks for Sentence Classification

"""

__version__ = '0.0.1'

import sys, time, logging, os, json, re, random
import numpy as np
np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'))
console.setLevel(logging.DEBUG)
logger.addHandler(console)
# logfile = logging.FileHandler(filename="log.txt")
# logfile.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'))
# logfile.setLevel(logging.DEBUG)
# logger.addHandler(logfile)


def pp(obj):
    import pprint
    pp = pprint.PrettyPrinter(indent=1, width=160)
    logger.info(pp.pformat(obj))


start_time = time.time()

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle as skshuffle


def sequence_embed(embed, xs, dropout=0.):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


class RNNClassifier(chainer.Chain):

    """A classifier using a LSTM-RNN Encoder with Word Embedding.

    This chain encodes a sentence and classifies it into classes.
    This model encodes a sentence sequentially using LSTM.

    Args:
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.
        n_class (int): The number of classes to be predicted.

    """

    def __init__(self, n_layers, n_vocab, n_units, n_class, dropout=0.1):
        super(RNNClassifier, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=chainer.initializers.Uniform(.25))
            self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, dropout)
            self.l1 = L.Linear(n_units * 2, 24)
            self.l2 = L.Linear(24, 1)
            self.output = L.Linear(n_units * 2, n_class)

        self.dropout = dropout

    def __call__(self, xs, ys):
        concat_outputs = self.predict(xs)
        concat_truths = F.concat(ys, axis=0)

        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        return loss, accuracy

    def predict(self, xs, softmax=False, argmax=False):
        # Input is a list of variables whose shapes are (sentence_length, ).
        # Output is a variable whose shape is "(batchsize, n_units).
        exs = sequence_embed(self.embed, xs, self.dropout)
        last_h, last_c, ys = self.encoder(None, None, exs)
        # h = last_h[-1]

        y_len = [len(y) for y in ys]
        y_section = np.cumsum(y_len[:-1])
        ay = self.l2(F.relu(self.l1(F.dropout(F.concat(ys, axis=0), ratio=self.dropout))))
        ays = F.split_axis(ay, y_section, 0)

        h_list = []
        for y, ay in zip(ys, ays):
            h_list.append(F.sum(y * F.broadcast_to(F.softmax(ay, axis=0), y.shape), axis=0)[None, :])
        h = F.concat(h_list, axis=0)

        concat_encodings = F.dropout(h, ratio=self.dropout)
        concat_outputs = self.output(concat_encodings)
        if softmax:
            return F.softmax(concat_outputs).data
        elif argmax:
            return self.xp.argmax(concat_outputs.data, axis=1)
        else:
            return concat_outputs


def load_data(path, vocab, labels):
    data = []

    print('loading...: %s' % path)
    for i, line in enumerate(open(path)):
        # if i > 100:
        #     break

        line = line.strip()
        line = line.replace(u'. . .', u'…')
        if line == '':
            continue

        label, words = line.split('\t')

        if label not in labels:
            labels[label] = len(labels)

        for word in words.split(' '):
            if word == '':
                continue
            if word not in vocab:
                vocab[word] = len(vocab)

        data.append((
            np.array([vocab.get(w) for w in words.split(' ')], 'i'),
            np.array([labels.get(label)], 'i')
        ))

    return data, vocab, labels


def batch_iter(data, batch_size, shuffle=True):
    batch = []
    shuffled_data = np.copy(data)
    if shuffle:
        shuffled_data = skshuffle(shuffled_data)

    for line in shuffled_data:
        batch.append(line)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch))
            # yield batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch))
        # yield batch


def to_device(device, x):
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: BiLSTM+Attention Classifier')
    parser.add_argument('--train',           default='',  type=str, help='training file (.txt)')
    parser.add_argument('--test',            default='',  type=str, help='evaluating file (.txt)')
    parser.add_argument('--gpu',       '-g', default=-1,  type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch',     '-e', default=30,  type=int, help='number of epochs to learn')
    parser.add_argument('--unit',      '-u', default=300, type=int, help='number of output channels')
    parser.add_argument('--batchsize', '-b', default=64, type=int, help='learning batchsize size')
    parser.add_argument('--out',       '-o', default='model-rnn_ba-embed-py3',  type=str, help='output directory')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.gpu >= 0 and cuda.check_cuda_available():
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.cupy.random.seed(seed)

    vocab, labels = {'<eos>': 0, '<unk>': 1, '<pad>': -1}, {}
    train, vocab, labels = load_data(args.train, vocab, labels)
    test,  vocab, labels = load_data(args.test,  vocab, labels)

    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    print('# class: {}'.format(len(labels)))
    sys.stdout.flush()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    # 学習の繰り返し回数
    n_epoch = args.epoch

    # 中間層の数
    n_units = args.unit

    # 確率的勾配降下法で学習させる際の1回分のバッチサイズ
    batchsize = args.batchsize

    input_channel = 1
    output_channel = 50
    width = 300
    n_units = args.unit
    n_vocab = len(vocab)
    n_class = len(labels)

    # Setup model
    model = RNNClassifier(n_layers=1, n_vocab=n_vocab, n_units=n_units, dropout=0.4, n_class=n_class)
    if args.gpu >= 0:
        model.to_gpu()

    # 重み減衰
    decay = 0.0001

    # 勾配上限
    grad_clip = 3

    # Setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

    # プロット用に実行結果を保存する
    train_loss = []
    train_accuracy1 = []
    train_accuracy2 = []
    test_loss = []
    test_accuracy1 = []
    test_accuracy2 = []
    best_accuracy = .0
    min_epoch = 0

    start_at = time.time()
    cur_at = start_at

    # Learning loop
    for epoch in range(1, args.epoch + 1):

        # logger.info('epoch {:} / {:}'.format(epoch, n_epoch))
        # handler1.flush()

        # training
        train_iter = batch_iter(train, args.batchsize, shuffle=True)
        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        for x, t in train_iter:
            x = to_device(args.gpu, x)
            t = to_device(args.gpu, t)

            # 勾配を初期化
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            loss, accuracy = model(x, t)
            sum_train_loss += float(loss.data) * len(t)
            sum_train_accuracy1 += float(accuracy.data) * len(t)
            sum_train_accuracy2 += .0
            K += len(t)

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
        now = time.time()
        train_throughput = now - cur_at
        cur_at = now

        # evaluation
        test_iter = batch_iter(test, args.batchsize, shuffle=False)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        # evaluation
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for x, t in test_iter:
                x = to_device(args.gpu, x)
                t = to_device(args.gpu, t)

                # 順伝播させて誤差と精度を算出
                loss, accuracy = model(x, t)
                sum_test_loss += float(loss.data) * len(t)
                sum_test_accuracy1 += float(accuracy.data) * len(t)
                sum_test_accuracy2 += .0
                K += len(t)

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

        logger.info(''
                    '[{:>3d}] '
                    'T/loss={:.6f} '
                    'T/acc1={:.6f} '
                    'T/acc2={:.6f} '
                    'T/sec= {:.6f} '
                    'D/loss={:.6f} '
                    'D/acc1={:.6f} '
                    'D/acc2={:.6f} '
                    'D/sec= {:.6f} '
                    'lr={:.6f}'
                    ''.format(
            epoch,
            mean_train_loss,
            mean_train_accuracy1,
            mean_train_accuracy2,
            train_throughput,
            mean_test_loss,
            mean_test_accuracy1,
            mean_test_accuracy2,
            test_throughput,
            optimizer.alpha)
        )
        sys.stdout.flush()

        # 精度と誤差をグラフ描画
        if True:
            ylim1 = [min(train_loss + test_loss), max(train_loss + test_loss)]
            # ylim2 = [min(train_accuracy1 + test_accuracy2), max(train_accuracy1 + test_accuracy2)]
            ylim2 = [0.5, 1.0]

            # グラフ左
            plt.figure(figsize=(10, 10))

            plt.subplot(1, 2, 1)
            plt.ylim(ylim1)
            plt.plot(range(1, len(train_loss) + 1), train_loss, color='C1', marker='x')
            # plt.grid()
            plt.ylabel('loss')
            plt.legend(['train loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(train_accuracy1) + 1), train_accuracy1, color='C0', marker='x')
            # plt.plot(range(1, len(train_accuracy2) + 1), train_accuracy2, color='C2', marker='x')
            plt.yticks(np.arange(ylim2[0], ylim2[1], .1))
            plt.grid(True)
            # plt.ylabel('accuracy')
            plt.legend(['train turn', 'train call'], loc="upper right")
            plt.title('Loss and accuracy of train.')

            # グラフ右
            plt.subplot(1, 2, 2)
            plt.ylim(ylim1)
            plt.plot(range(1, len(test_loss) + 1), test_loss, color='C1', marker='x')
            # plt.grid()
            # plt.ylabel('loss')
            plt.legend(['dev loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(test_accuracy1) + 1), test_accuracy1, color='C0', marker='x')
            # plt.plot(range(1, len(test_accuracy2) + 1), test_accuracy2, color='C2', marker='x')
            plt.yticks(np.arange(ylim2[0], ylim2[1], .1))
            plt.grid(True)
            plt.ylabel('accuracy')
            plt.legend(['dev turn', 'dev call'], loc="upper right")
            plt.title('Loss and accuracy of dev.')

            # plt.savefig('{}.png'.format(args.out))
            plt.savefig('{}.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        cur_at = now

print('time spent:', time.time() - start_time)
