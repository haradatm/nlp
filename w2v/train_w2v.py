#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" KGE: Knowledge Graph Embedding example
"""

__version__ = '0.0.1'

import sys, time, logging, os, json
import numpy as np

np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
# handler = logging.FileHandler(filename="log.txt")
handler.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'))
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def pp(obj):
    import pprint
    pp = pprint.PrettyPrinter(indent=1, width=160)
    logger.info(pp.pformat(obj))


start_time = time.time()

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import matplotlib.pyplot as plt
import collections, pickle


class ContinuousBoW(chainer.Chain):
    """Definition of Continuous Bag of Words Model"""

    def __init__(self, n_vocab, n_units):
        super(ContinuousBoW, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.out = L.Linear(n_units, n_vocab, initialW=0)

    def __call__(self, x, contexts):
        e = self.embed(contexts)
        h = F.sum(e, axis=1) * (1. / contexts.shape[1])
        loss = F.softmax_cross_entropy(self.out(h), x)
        return loss


class SkipGram(chainer.Chain):
    """Definition of Skip-gram Model"""

    def __init__(self, n_vocab, n_units):
        super(SkipGram, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.out = L.Linear(n_units, n_vocab, initialW=0)

    def __call__(self, x, contexts):
        e = self.embed(contexts)
        batch_size, n_context, n_units = e.shape
        x = F.broadcast_to(x[:, None], (batch_size, n_context))
        e = F.reshape(e, (batch_size * n_context, n_units))
        x = F.reshape(x, (batch_size * n_context,))
        loss = F.softmax_cross_entropy(self.out(e), x)
        return loss


class ContinuousBoW_NS(chainer.Chain):
    """Definition of Continuous Bag of Words Model"""

    def __init__(self, n_vocab, n_units, cs, n_ngs):
        super(ContinuousBoW_NS, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.out = L.NegativeSampling(n_units, cs, n_ngs)

        self.out.W.data[...] = 0

    def __call__(self, x, contexts):
        e = self.embed(contexts)
        h = F.sum(e, axis=1) * (1. / contexts.shape[1])
        loss = self.out(h, x)
        return loss


class SkipGram_NS(chainer.Chain):
    """Definition of Skip-gram Model"""

    def __init__(self, n_vocab, n_units, cs, n_ngs):
        super(SkipGram_NS, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.out = L.Linear(n_units, n_vocab, initialW=0)
            self.out = L.NegativeSampling(n_units, cs, n_ngs)

        self.out.W.data[...] = 0

    def __call__(self, x, contexts):
        e = self.embed(contexts)
        batch_size, n_context, n_units = e.shape
        x = F.broadcast_to(x[:, None], (batch_size, n_context))
        e = F.reshape(e, (batch_size * n_context, n_units))
        x = F.reshape(x, (batch_size * n_context,))
        loss = self.out(e, x)
        return loss


EOS_TOKEN = '</s>'


def load_data(path, vocab={}):
    dataset = []

    for line in open(path, 'r'):
        line = line.strip()
        if line == '':
            continue
        words = line.split(' ') + [EOS_TOKEN]
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
            dataset.append(vocab[word])

    return np.array(dataset, 'i'), vocab


class WindowIterator(object):
    """Dataset iterator to create a batch of sequences at different positions.

    This iterator returns a pair of the current words and the context words.
    """

    def __init__(self, dataset, window, batch_size, repeat=True):
        self.dataset = np.array(dataset, np.int32)
        self.window = window
        self.batch_size = batch_size
        self._repeat = repeat
        self.order = np.random.permutation(len(dataset) - window * 2).astype(np.int32)
        self.order += window
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __iter__(self):
        return self

    def __next__(self):
        """This iterator returns a list representing a mini-batch.

        Each item indicates a different position in the original sequence.
        """
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i:i_end]
        w = np.random.randint(self.window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
        pos = position[:, None] + offset[None, :]
        contexts = self.dataset.take(pos)
        center = self.dataset.take(position)

        if i_end >= len(self.order):
            np.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return center, contexts


def convert(batch, device):
    center, contexts = batch
    if device >= 0:
        center = cuda.to_gpu(center)
        contexts = cuda.to_gpu(contexts)
    return center, contexts


def main():
    global xp

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=100, type=int, help='number of units')
    parser.add_argument('--window', '-w', default=5, type=int, help='window size')
    parser.add_argument('--batchsize', '-b', type=int, default=500, help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int, help='number of epochs to learn')
    parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'], default='skipgram', help='model type ("skipgram", "cbow")')
    parser.add_argument('--negative-size', default=5, type=int, help='number of negative samples')
    parser.add_argument('--out-type', '-o', choices=['hsm', 'ns', 'original'], default='original', help='output model type ("hsm": hierarchical softmax, "ns": negative sampling, "original": no approximation)')
    parser.add_argument('--out', default='result', help='Directory to output the result')
    parser.add_argument('--test', dest='test', action='store_true')
    # parser.set_defaults(test=True)
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('Window: {}'.format(args.window))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Training model: {}'.format(args.model))
    print('Output type: {}'.format(args.out_type))
    print('')
    sys.stdout.flush()

    xp = cuda.cupy if args.gpu >= 0 else np

    # Set random seed
    xp.random.seed(123)

    # Load the dataset
    train, vocab = load_data("datasets/ptb/ptb.train.txt")
    val,   vocab = load_data("datasets/ptb/ptb.valid.txt", vocab)

    counts = collections.Counter(train)
    counts.update(collections.Counter(val))
    n_vocab = max(train) + 1

    if args.test:
        train = train[:100]
        val = val[:100]

    index2word = {wid: word for word, wid in vocab.items()}

    logger.info('vocabulary size: %d' % n_vocab)
    logger.info('train data length: %d' % len(train))
    logger.info('vaid  data length: %d' % len(val))
    sys.stdout.flush()

    cs = [counts[w] for w in range(len(counts))]

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    with open(os.path.join(args.out, 'vocab.bin'), 'wb') as f:
        pickle.dump(vocab, f)

    # Model Parameters
    model = SkipGram(n_vocab, args.unit)

    if args.gpu >= 0:
        model.to_gpu()

    # 学習率
    lr = 0.0007

    # 勾配上限
    gradclip = 0.0005

    # L2正則化
    decay = 0.0005

    # 学習率の減衰
    lr_decay = 0.995

    # Setup optimizer (Optimizer の設定)
    optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

    # Set up an iterator
    train_iter = WindowIterator(train, args.window, args.batchsize)
    val_iter = WindowIterator(val, args.window, args.batchsize, repeat=False)

    # プロット用に実行結果を保存する
    train_loss = []
    train_accuracy1 = []
    train_accuracy2 = []
    test_loss = []
    test_accuracy1 = []
    test_accuracy2 = []
    min_loss = float('inf')
    min_epoch = 0

    # 最初の時間情報を取得する
    start_at = time.time()
    cur_at = start_at

    # Learning loop
    for epoch in range(1, args.epoch + 1):

        # logger.info('epoch {:} / {:}'.format(epoch, n_epoch))
        # handler1.flush()

        # training
        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        iteration = 1.

        for batch in train_iter:
            center, contexts = convert(batch, args.gpu)

            # 順伝播させて誤差と精度を算出
            loss = model(center, contexts)
            sum_train_loss += float(loss.data) * len(center)
            sum_train_accuracy1 += .0
            sum_train_accuracy2 += .0
            K += len(center)

            # 誤差逆伝播で勾配を計算 (minibatch ごと)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            if iteration != 0 and iteration % 100 == 0:
                logger.debug("epoch: {} ({:2.2f}%)  loss: {:.6f}".format(epoch, (iteration * 100 / len(train) / args.batchsize), float(loss.data)))
            iteration += 1

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
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for batch in val_iter:
                center, contexts = convert(batch, args.gpu)

                # 順伝播させて誤差と精度を算出
                loss = model(center, contexts )
                sum_test_loss += float(loss.data) * len(center)
                sum_test_accuracy1 += .0
                sum_test_accuracy2 += .0
                K += len(center)

        # テストデータでの誤差と正解精度を表示
        mean_test_loss = sum_test_loss / K
        mean_test_accuracy1 = sum_test_accuracy1 / K
        mean_test_accuracy2 = sum_test_accuracy2 / K
        test_loss.append(mean_test_loss)
        test_accuracy1.append(mean_test_accuracy1)
        test_accuracy2.append(mean_test_accuracy2)
        now = time.time()
        test_throughput = now - cur_at

        logger.info(''
                    '[{:>3d}] '
                    'T/loss={:.6f} '
                    'T/acc={:.6f} '
                    'T/perp={:.6f} '
                    'T/sec= {:.6f} '
                    'D/loss={:.6f} '
                    'D/acc={:.6f} '
                    'D/perp={:.6f} '
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

        # model と optimizer を保存する
        if mean_train_loss < min_loss:
            min_loss = mean_train_loss
            min_epoch = epoch
            if args.gpu >= 0: model.to_cpu()
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped.model'), model)
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped.state'), optimizer)
            if args.gpu >= 0: model.to_gpu()

        # 精度と誤差をグラフ描画
        if True:
            ylim1 = [min(train_loss + train_accuracy2 + test_loss + test_accuracy2), max(train_loss + train_accuracy2 + test_loss + test_accuracy2)]
            ylim2 = [min(train_accuracy1 + test_accuracy1), max(train_accuracy1 + test_accuracy1)]

            # グラフ左
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.ylim(ylim1)
            plt.plot(range(1, len(train_loss) + 1), train_loss, 'b')
            plt.plot(range(1, len(train_accuracy2) + 1), train_accuracy2, 'm')
            plt.grid(False)
            plt.ylabel('loss and perplexity')
            plt.legend(['train loss', 'train perplexity'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(train_accuracy1) + 1), train_accuracy1, 'r')
            plt.grid(False)
            # plt.ylabel('accuracy')
            plt.legend(['train accuracy'], loc="upper right")
            plt.title('Loss and accuracy of train.')

            # グラフ右
            plt.subplot(1, 2, 2)
            plt.ylim(ylim1)
            plt.plot(range(1, len(test_loss) + 1), test_loss, 'b')
            plt.plot(range(1, len(test_accuracy2) + 1), test_accuracy2, 'm')
            plt.grid(False)
            # plt.ylabel('loss and perplexity')
            plt.legend(['valid loss', 'valid perplexity'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(test_accuracy1) + 1), test_accuracy1, 'r')
            plt.grid(False)
            plt.ylabel('accuracy')
            plt.legend(['valid accuracy'], loc="upper right")
            plt.title('Loss and accuracy of valid.')

            plt.savefig('{}.png'.format(args.out))
            # plt.savefig('{}.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()

        optimizer.alpha *= lr_decay
        cur_at = now

    # model と optimizer を保存する
    if args.gpu >= 0: model.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.out, 'final.model'), model)
    chainer.serializers.save_npz(os.path.join(args.out, 'final.state'), optimizer)
    if args.gpu >= 0: model.to_gpu()

    # word2vec model を出力する
    print('save the word2vec model at epoch {}'.format(min_epoch))
    with open(os.path.join(args.out, "word2vec.model"), 'w') as f:
        f.write('%d %d\n' % (len(index2word), args.unit))
        w = cuda.to_cpu(model.embed.W.data)
        for i, wi in enumerate(w):
            v = ' '.join(map(str, wi))
            f.write('%s %s\n' % (index2word[i], v))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
