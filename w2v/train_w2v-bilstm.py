#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
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


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class ContinuousBoW(chainer.Chain):
    """Definition of Continuous Bag of Words Model"""

    def __init__(self, n_vocab, n_units):
        super(ContinuousBoW, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.l1 = L.NStepBiLSTM(2, n_units, n_units, 0.25)
            self.out = L.Linear(n_units * 2, n_vocab, initialW=I.HeNormal())

    def __call__(self, x, contexts):
        y = self.forward(contexts)
        return F.softmax_cross_entropy(y, x), F.accuracy(F.softmax(y), x)

    def predict(self, contexts):
        y = self.forward(contexts)
        return F.softmax(y)

    def forward(self, contexts):
        exs = sequence_embed(self.embed, contexts)
        hx = None
        cx = None

        hx, cx, os = self.l1(hx, cx, exs)
        ys = F.concat([y[-1:, :] for y in os], axis=0)
        y = self.out(ys)
        return y


EOS_TOKEN = '</s>'


def load_data(path, vocab={}):
    dataset = []

    for i, line in enumerate(open(path, 'r')):
        line = line.strip()
        if line == '':
            continue
        # if i > 10000:
        #     break

        # words = ['吾輩', 'は', '猫', 'で', 'ある', '。'] + [EOS_TOKEN]
        words = line.split(' ') + [EOS_TOKEN]

        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
            dataset.append(vocab[word])

    return np.array(dataset, 'i'), vocab


def get_minibatches(dataset, window, batch_size):

    # 学習時に文書の最初から最後まで順番に学習するのではなく,文書からランダムに単語を選択し学習するため,
    # ウィンドウサイズ分だけ最初と最後を切り取った単語の位置をシャッフルする
    order = np.random.permutation(len(dataset) - window * 2).astype(np.int32)
    order += window

    for i in range(0, len(order), batch_size):

        # 単語の位置をシャッフルしたリストから batch_size 分の Target Word のインデックスを生成する
        position = order[i:i + batch_size]

        # ウインドウを表現するオフセットを作成する
        w = np.random.randint(window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])

        # 各 Target Word に対する Context Word のインデックスを生成する
        pos = position[:, None] + offset[None, :]

        contexts = dataset.take(pos)
        center = dataset.take(position)

        yield center, contexts


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
    parser.add_argument('--batchsize', '-b', type=int, default=1000, help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int, help='number of epochs to learn')
    parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'], default='cbow', help='model type ("skipgram", "cbow")')
    parser.add_argument('--negative-size', default=5, type=int, help='number of negative samples')
    parser.add_argument('--out-type', '-o', choices=['hsm', 'ns', 'original'], default='original', help='output model type ("hsm": hierarchical softmax, "ns": negative sampling, "original": no approximation)')
    parser.add_argument('--out', default='result-3', help='Directory to output the result')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
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
    train, vocab = load_data("datasets/soseki/neko-word-train.txt")
    val,   vocab = load_data("datasets/soseki/neko-word-test.txt", vocab)

    counts = collections.Counter(train)
    counts.update(collections.Counter(val))
    n_vocab = len(vocab)

    if args.test:
        train = train[:1000]
        val = val[:1000]

    index2word = {wid: word for word, wid in vocab.items()}

    logger.info('vocabulary size: %d' % n_vocab)
    logger.info('train data length: %d' % len(train))
    logger.info('vaid  data length: %d' % len(val))
    sys.stdout.flush()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    with open(os.path.join(args.out, 'vocab.bin'), 'wb') as f:
        pickle.dump(vocab, f)

    # Model Parameters
    # if args.model == 'skipgram':
    #     model = SkipGram(n_vocab, args.unit)
    # else:
    #     model = ContinuousBoW(n_vocab, args.unit)
    model = ContinuousBoW(n_vocab, args.unit)

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
    # optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

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

        # Set up an iterator
        train_iter = get_minibatches(train, args.window, args.batchsize)
        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        # training
        for batch in train_iter:
            center, contexts = convert(batch, args.gpu)

            # 誤差逆伝播で勾配を計算 (minibatch ごと)
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            loss, accuracy = model(center, contexts)
            sum_train_loss += float(loss.data) * len(center)
            sum_train_accuracy1 += float(accuracy.data) * len(center)
            sum_train_accuracy2 += .0
            K += len(center)

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

        # Set up an iterator
        val_iter = get_minibatches(val, args.window, args.batchsize)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        # evaluation
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for batch in val_iter:
                center, contexts = convert(batch, args.gpu)

                # 順伝播させて誤差と精度を算出
                loss, accuracy = model(center, contexts)
                sum_test_loss += float(loss.data) * len(center)
                sum_test_accuracy1 += float(accuracy.data) * len(center)
                sum_test_accuracy2 += .0
                K += len(center)

            # test
            contexts = [vocab[x] for x in ['吾輩', 'は', 'で', 'ある']]
            y = model.predict(xp.array([contexts], 'i'))
            yy = np.argsort(cuda.to_cpu(y.data)[0])
            print("SAMPLE# '吾輩 は【 】で ある' => ", end='')
            print(" ".join(["{}. {}".format(x1 + 1, index2word[yy[x2]]) for x1, x2 in enumerate(range(-5, -0, 1))]))
            sys.stdout.flush()

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
                    'T/accuracy={:.6f} '
                    # 'T/acc2={:.6f} '
                    'T/sec= {:.6f} '
                    'D/loss={:.6f} '
                    'D/accuracy={:.6f} '
                    # 'D/acc2={:.6f} '
                    'D/sec= {:.6f} '
                    'lr={:.6f}'
                    ''.format(
            epoch,
            mean_train_loss,
            mean_train_accuracy1,
            # mean_train_accuracy2,
            train_throughput,
            mean_test_loss,
            mean_test_accuracy1,
            # mean_test_accuracy2,
            test_throughput,
            optimizer.alpha)
        )
        sys.stdout.flush()

        # model と optimizer を保存する
        if mean_test_loss < min_loss:
            min_loss = mean_test_loss
            min_epoch = epoch
            if args.gpu >= 0: model.to_cpu()
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped.model'), model)
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped.state'), optimizer)
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

        # 精度と誤差をグラフ描画
        if True:
            ylim1 = [min(train_loss + test_loss), max(train_loss + test_loss)]
            # ylim2 = [min(train_accuracy1 + test_accuracy1), max(train_accuracy1 + test_accuracy1)]
            ylim2 = [0, 1]

            # グラフ左
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.ylim(ylim1)
            plt.plot(range(1, len(train_loss) + 1), train_loss, 'b')
            # plt.plot(range(1, len(train_accuracy2) + 1), train_accuracy2, 'm')
            plt.grid(False)
            plt.ylabel('loss and accuracy')
            plt.legend(['train loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(train_accuracy1) + 1), train_accuracy1, 'r')
            plt.grid(False)
            # plt.ylabel('accuracy')
            plt.legend(['train accuracy'], loc="upper right")
            plt.title('Loss and accuracy for train.')

            # グラフ右
            plt.subplot(1, 2, 2)
            plt.ylim(ylim1)
            plt.plot(range(1, len(test_loss) + 1), test_loss, 'b')
            # plt.plot(range(1, len(test_accuracy2) + 1), test_accuracy2, 'm')
            plt.grid(False)
            # plt.ylabel('loss and accuracy')
            plt.legend(['test loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(test_accuracy1) + 1), test_accuracy1, 'r')
            plt.grid(False)
            plt.ylabel('accuracy')
            plt.legend(['test accuracy'], loc="upper right")
            plt.title('Loss and accuracy for test.')

            plt.savefig('{}.png'.format(args.out))
            # plt.savefig('{}.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()

        # optimizer.alpha *= lr_decay
        cur_at = now

    # model と optimizer を保存する
    if args.gpu >= 0: model.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.out, 'final.model'), model)
    chainer.serializers.save_npz(os.path.join(args.out, 'final.state'), optimizer)
    if args.gpu >= 0: model.to_gpu()

    # word2vec model を出力する
    with open(os.path.join(args.out, "final-word2vec.model"), 'w') as f:
        f.write('%d %d\n' % (len(index2word), args.unit))
        w = cuda.to_cpu(model.embed.W.data)
        for i, wi in enumerate(w):
            v = ' '.join(map(str, wi))
            f.write('%s %s\n' % (index2word[i], v))


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
