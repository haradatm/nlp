#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sample script of Bi-LSTM CRF model.

Bidirectional LSTM-CRF for Sequence Labeling like Named-Entity Recognition
[Lample,2016] Neural Architectures for Named Entity Recognition by Lample, Guillaume, et al., NAACL 2016.
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
from seqeval.metrics import f1_score, accuracy_score, classification_report
from struct import unpack, calcsize


UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
START_TAG = '<START>'
STOP_TAG  = '<STOP>'


def load_w2v_model(path, vocab):
    n_reserved = len(vocab)

    with open(path, 'rb') as f:
        n_rows, n_cols = map(int, f.readline().split())
        reserved = np.random.uniform(-0.1, 0.1, (n_reserved, n_cols)).astype(np.float32)
        reserved /= np.linalg.norm(reserved, 2)
        M = np.vstack((reserved, np.empty((n_rows, n_cols), dtype=np.float32)))
        n_vocab = n_reserved + n_rows

        for i in range(n_reserved, n_vocab):
            b_str = b''

            while True:
                b_ch = f.read(1)
                if b_ch == b' ':
                    break
                b_str += b_ch

            token = b_str.decode(encoding='utf-8')

            if token not in vocab:
                vocab[token] = len(vocab)
            else:
                logging.error("Duplicate token: {}", token)

            M[i] = np.zeros(n_cols)
            for j in range(n_cols):
                M[i][j] = unpack('f', f.read(calcsize('f')))[0]

            # ベクトルを正規化する
            vlen = np.linalg.norm(M[i], 2)
            M[i] /= vlen

            # 改行を strip する
            assert f.read(1) != '\n'

    return M, vocab


def load_data(path, vocab_word, vocab_tag, w2v):
    X, y = [], []
    words, tags = [], []

    for i, line in enumerate(open(path, 'r')):
        # if i > 10000:
        #     break

        line = line.strip()
        if line != '':
            word, tag = line.split('\t')
            if word not in vocab_word:
                if w2v is not None:
                    v = np.random.uniform(-0.1, 0.1, (1, w2v.shape[1])).astype(np.float32)
                    v /= np.linalg.norm(v, 2)
                    w2v = np.vstack((w2v, v))
                vocab_word[word] = len(vocab_word)
            if tag not in vocab_tag:
                vocab_tag[tag] = len(vocab_tag)
            words.append(vocab_word[word])
            tags.append(vocab_tag[tag])
        else:
            X.append(xp.array(words, 'i'))
            y.append(xp.array(tags, 'i'))
            words, tags = [], []

    return X, y, vocab_word, vocab_tag


def convert(batch, device):
    center, contexts = batch
    if device >= 0:
        center = cuda.to_gpu(center)
        contexts = cuda.to_gpu(contexts)
    return center, contexts


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class BiLSTM_CRF(chainer.Chain):

    def __init__(self, word_vocab_size, word_emb_size, word_lstm_units, num_tags):
        super(BiLSTM_CRF, self).__init__()

        with self.init_scope():
            self.embed_word = L.EmbedID(word_vocab_size, word_emb_size, initialW=I.HeNormal())
            self.l2 = L.NStepBiLSTM(1, word_emb_size, word_lstm_units, 0.25)
            self.l3 = L.Linear(word_lstm_units * 2, num_tags, initialW=I.HeNormal())
            self.crf = L.CRF1d(num_tags)

    def __call__(self, x_list, t_list):
        y_list = self.forward(x_list)
        loss = self.crf(y_list, t_list)
        return loss

    def forward(self, x_list):
        exs = sequence_embed(self.embed_word, x_list)
        hx, cx, ys = self.l2(None, None, exs)
        y_list = [self.l3(y) for y in ys]
        return y_list

    def predict(self, x_list):
        ys = self.forward(x_list)
        _, y_list = self.crf.argmax(ys)
        return y_list

    def set_word_embedding(self, data):
        self.embed_word.W.data = data


def sorted_parallel(generator1, generator2, pooling, order=0):
    gen1 = batch(generator1, pooling)
    gen2 = batch(generator2, pooling)
    for batch1, batch2 in zip(gen1, gen2):
        for x in sorted(zip(batch1, batch2), key=lambda x: len(x[order]), reverse=True):
            yield x


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


def main():
    global xp

    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: Bi-LSTM CRF')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train', default='datasets/train.txt', type=str, help='dataset to train (.txt)')
    parser.add_argument('--valid', default='datasets/valid.txt', type=str, help='use tiny datasets to evaluate (.txt)')
    parser.add_argument('--w2v', '-w', default='', type=str, help='initialize word embedding layer with word2vec (.bin)')
    parser.add_argument('--test', default='datasets/test.txt', type=str, help='use tiny datasets to test (.txt)')
    parser.add_argument('--unit', '-u', default=100, type=int, help='number of units')
    parser.add_argument('--batchsize', '-b', type=int, default=1000, help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=100, type=int, help='number of epochs to learn')
    parser.add_argument('--out', default='result-2', help='Directory to output the result')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()

    xp = cuda.cupy if args.gpu >= 0 else np

    # Set random seed
    xp.random.seed(123)

    vocab_word = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    vocab_tag = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    word_emb_size = 100
    w2v = None

    if args.w2v:
        w2v, vocab = load_w2v_model(args.w2v, vocab_word)
        word_emb_size = w2v.shape[1]

    # Load the dataset
    x_train_words, y_train, vocab_word, vocab_tag = load_data(args.train, vocab_word, vocab_tag, w2v)
    x_valid_words, y_valid, vocab_word, vocab_tag = load_data(args.valid, vocab_word, vocab_tag, w2v)
    x_test_words,  y_test,  vocab_word, vocab_tag = load_data(args.test,  vocab_word, vocab_tag, w2v)

    index2word = {v: k  for k, v in vocab_word.items()}
    index2tag  = {v: k  for k, v in vocab_tag.items()}

    word_vocab_size = len(vocab_word)
    word_lstm_units = args.unit
    num_tags = len(vocab_tag)

    logger.info('vocabulary size: %d' % word_vocab_size)
    logger.info('number of word embedding dims: %d' % word_emb_size)
    logger.info('number of lstm units: %d' % word_lstm_units)
    logger.info('number of tags: %d' % num_tags)
    logger.info('train data length: %d' % len(y_train))
    logger.info('valid data length: %d' % len(y_valid))
    logger.info('test  data length: %d' % len(y_test))
    sys.stdout.flush()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    model = BiLSTM_CRF(word_vocab_size, word_emb_size, word_lstm_units, num_tags)

    # Initialize word embedding layer with word2vec
    if w2v is not None:
        print('Initialize word embedding from word2vec model: {}'.format(args.w2v))
        model.set_word_embedding(w2v)
        sys.stdout.flush()

    if args.gpu >= 0:
        model.to_gpu()

    # 学習率
    lr = 0.015

    # 勾配上限
    gradclip = 5.

    # L2正則化
    decay = 0.0005

    # 学習率の減衰
    lr_decay = 0.95

    # Setup optimizer (Optimizer の設定)
    optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

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
        train_iter = batch_tuple(sorted_parallel(x_train_words, y_train, args.batchsize), args.batchsize)

        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        # training
        for x_batch, t_batch in train_iter:

            # 順伝播させて誤差と精度を算出
            loss = model(x_batch, t_batch)
            sum_train_loss += float(loss.data) * len(t_batch)

            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                y_pred = model.predict(x_batch)
                y_tags = [[index2tag[label_id] for label_id in labels] for labels in cuda.to_cpu(y_pred)]
                t_tags = [[index2tag[label_id] for label_id in labels] for labels in cuda.to_cpu(t_batch)]
                sum_train_accuracy1 += f1_score(t_tags, y_tags) * len(t_batch)
                sum_train_accuracy2 += accuracy_score(t_tags, y_tags) * len(t_batch)

            K += len(t_batch)

            # 誤差逆伝播で勾配を計算
            model.cleargrads()
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
        val_iter = batch_tuple(sorted_parallel(x_valid_words, y_valid, args.batchsize), args.batchsize)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        # evaluation
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for x_batch, t_batch in val_iter:

                # 順伝播させて誤差と精度を算出
                loss = model(x_batch, t_batch)
                sum_test_loss += float(loss.data) * len(t_batch)

                y_pred = model.predict(x_batch)
                y_tags = [[index2tag[label_id] for label_id in labels] for labels in cuda.to_cpu(y_pred)]
                t_tags = [[index2tag[label_id] for label_id in labels] for labels in cuda.to_cpu(t_batch)]
                sum_test_accuracy1 += f1_score(t_tags, y_tags) * len(t_batch)
                sum_test_accuracy2 += accuracy_score(t_tags, y_tags) * len(t_batch)

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

        logger.info(''
                    '[{:>3d}] '
                    'T/loss={:.6f} '
                    'T/f1={:.6f} '
                    'T/acc={:.6f} '
                    'T/sec= {:.6f} '
                    'V/loss={:.6f} '
                    'V/f1={:.6f} '
                    'V/acc={:.6f} '
                    'V/sec= {:.6f} '
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
            ylim1 = [min(train_loss + test_loss), max(train_loss + test_loss)]
            # ylim2 = [min(train_accuracy1 + test_accuracy1), max(train_accuracy1 + test_accuracy1)]
            ylim2 = [0, 1]

            # グラフ左
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.ylim(ylim1)
            plt.plot(range(1, len(train_loss) + 1), train_loss, 'b')
            plt.grid(False)
            plt.ylabel('loss and score')
            plt.legend(['train loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(train_accuracy1) + 1), train_accuracy1, 'r')
            plt.plot(range(1, len(train_accuracy2) + 1), train_accuracy2, 'm')
            plt.grid(False)
            # plt.ylabel('score')
            plt.legend(['train f1-score', 'train accuracy'], loc="upper right")
            plt.title('Loss and score for train.')

            # グラフ右
            plt.subplot(1, 2, 2)
            plt.ylim(ylim1)
            plt.plot(range(1, len(test_loss) + 1), test_loss, 'b')
            plt.grid(False)
            # plt.ylabel('loss and score')
            plt.legend(['valid loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(test_accuracy1) + 1), test_accuracy1, 'r')
            plt.plot(range(1, len(test_accuracy2) + 1), test_accuracy2, 'm')
            plt.grid(False)
            plt.ylabel('score')
            plt.legend(['valid f1-score', 'valid accuracy'], loc="upper right")
            plt.title('Loss and score for valid.')

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

    # test
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        test_iter = batch_tuple(sorted_parallel(x_test_words, y_test, len(y_test)), len(y_test))
        for x_batch, t_batch in test_iter:
            y_pred = model.predict(x_batch)
            y_tags = [[index2tag[label_id] for label_id in labels] for labels in cuda.to_cpu(y_pred)]
            t_tags = [[index2tag[label_id] for label_id in labels] for labels in cuda.to_cpu(t_batch)]
            print(classification_report(t_tags, y_tags))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
