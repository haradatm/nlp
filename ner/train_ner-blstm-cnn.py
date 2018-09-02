#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sample script of Bi-LSTM CRF model.

Bidirectional LSTM-CRF for Sequence Labeling like Named-Entity Recognition
[Ma and Hovy,2016] End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF.
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
import chainer.links as L
import matplotlib.pyplot as plt
from seqeval.metrics import f1_score, accuracy_score, classification_report
from struct import unpack, calcsize


UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
START_TAG = '<START>'
STOP_TAG  = '<STOP>'


def load_glove_model(path):
    vocab = {}
    w = []

    for i, line in enumerate(open(path, 'r')):
        line = line.strip()
        if line == '':
            continue
        cols = line.split(' ')

        token = cols[0]
        if token not in vocab:
            vocab[token] = len(vocab)
        else:
            logging.error("Duplicate token: {}", token)

        v = np.array([float(x) for x in cols[1:]], dtype=np.float32)
        v /= np.linalg.norm(v, 2)
        w.append(v)

    M = np.array(w, dtype=np.float32)

    return M, vocab


def load_w2v_model(path):
    vocab = {}

    with open(path, 'rb') as f:
        n_vocab, n_units = map(int, f.readline().split())
        M = np.empty((n_vocab, n_units), dtype=np.float32)

        for i in range(n_vocab):
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

            M[i] = np.zeros(n_units)
            for j in range(n_units):
                M[i][j] = unpack('f', f.read(calcsize('f')))[0]

            # ベクトルを正規化する
            vlen = np.linalg.norm(M[i], 2)
            M[i] /= vlen

            # 改行を strip する
            assert f.read(1) != '\n'

    return M, vocab


def load_data(path, vocab_word, vocab_char, vocab_tag):
    X_word, X_char, y = [], [], []
    words, chars, tags = [], [], []

    for i, line in enumerate(open(path, 'r')):
        # if i > 100:
        #     break

        line = line.strip()
        if line != '':
            word, tag = line.split('\t')
            word = word.lower()

            if word not in vocab_word:
                vocab_word += [word]

            if tag not in vocab_tag:
                vocab_tag[tag] = len(vocab_tag)

            chs = []
            for ch in word:
                if ch not in vocab_char:
                    vocab_char += [ch]
                chs.append(vocab_char.index(ch))

            words.append(vocab_word.index(word))
            chars.append(xp.array(chs, 'i'))
            tags.append(vocab_tag[tag])
        else:
            X_word.append(xp.array(words, 'i'))
            X_char.append(chars)
            y.append(xp.array(tags, 'i'))
            words, chars, tags = [], [], []

    return X_word, X_char, y, vocab_word, vocab_char, vocab_tag


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class BLSTM_CRF_CNN(chainer.Chain):

    def __init__(self, word_vocab_size, word_emb_size, word_lstm_units, char_vocab_size, char_emb_size, char_lstm_units, num_tags, windows=3, filters=30):
        super(BLSTM_CRF_CNN, self).__init__()

        with self.init_scope():
            self.embed_char = L.EmbedID(char_vocab_size, char_emb_size, ignore_label=-1)
            self.conv1 = L.Convolution2D(1, filters, (windows, char_emb_size), pad=0)
            self.fc1 = L.Linear(filters * windows, char_emb_size)
            self.embed_word = L.EmbedID(word_vocab_size, word_emb_size, ignore_label=-1)
            self.lstm2 = L.NStepBiLSTM(1, word_emb_size + char_emb_size, word_lstm_units // 2, 0.5)
            self.fc2 = L.Linear(word_lstm_units, num_tags, initialW=.0)
            self.crf = L.CRF1d(num_tags)

        for param in self.params():
            param.data[...] = np.random.uniform(-np.sqrt(6. / sum(param.data.shape)), np.sqrt(6. / sum(param.data.shape)), param.data.shape).astype(np.float32)
        self.embed_char.W.data = np.random.uniform(-np.sqrt(3. / char_emb_size), np.sqrt(3. / char_emb_size), self.embed_char.W.shape).astype(np.float32)
        self.embed_word.W.data = np.random.uniform(-np.sqrt(3. / word_emb_size), np.sqrt(3. / word_emb_size), self.embed_word.W.shape).astype(np.float32)

    def __call__(self, x_words_list, x_chars_list, t_list):
        y_list = self.forward(x_words_list, x_chars_list)
        ys = F.transpose_sequence(y_list)
        ts = F.transpose_sequence(t_list)
        loss = self.crf(ys, ts)
        return loss

    def forward(self, x_words_list, x_chars_list):

        # CNN (char)
        exs_chars = []
        for x_chars in x_chars_list:
            x_chars_filled = []
            max_len = max([len(x) for x in x_chars] + [4])
            for x_char in x_chars:
                left = (max_len - len(x_char)) // 2
                right = max_len - len(x_char) - left
                x_chars_filled.append(xp.pad(x_char, [left, right], 'constant', constant_values=-1))
            exs = sequence_embed(self.embed_char, x_chars_filled)

            # char CNN
            # (rows, channel, height, width) の4次元テンソルに変換
            x = xp.zeros((len(exs), 1, exs[0].shape[0], exs[0].shape[1]), dtype=np.float32)
            for i, exs in enumerate(exs):
                x[i, 0] = (F.dropout(exs, ratio=0.5)).data
            h1 = F.spatial_pyramid_pooling_2d(F.relu(self.conv1(x)), 2, F.MaxPooling2D)
            h2 = F.relu(self.fc1(h1))

            exs_chars.append(h2)

        # BiLSTM (char + word)
        exs_words = sequence_embed(self.embed_word, x_words_list)
        exs_concat = []
        for w, c in zip(exs_words, exs_chars):
            exs_concat.append(F.dropout(F.concat((w, c), axis=1), ratio=0.5))

        hx, cx, ys = self.lstm2(None, None, exs_concat)
        y_list = [F.dropout(self.fc2(y), ratio=0.5) for y in ys]
        return y_list

    def predict(self, x_words_list, x_chars_list):
        y_list = self.forward(x_words_list, x_chars_list)
        ys = F.transpose_sequence(y_list)
        _, predict = self.crf.argmax(ys)
        return [y.data for y in F.transpose_sequence(predict)]

    def set_word_embedding(self, data):
        self.embed_word.W.data = data


def sorted_parallel(generator1, generator2, generator3, pooling, order=0):
    gen1 = batch(generator1, pooling)
    gen2 = batch(generator2, pooling)
    gen3 = batch(generator3, pooling)
    for batch1, batch2, batch3 in zip(gen1, gen2, gen3):
        for x in sorted(zip(batch1, batch2, batch3), key=lambda x: len(x[order]), reverse=True):
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
    parser = argparse.ArgumentParser(description='Chainer example: BLSTM-CRF_CNN')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train', default='datasets/train.txt', type=str, help='dataset to train (.txt)')
    parser.add_argument('--valid', default='datasets/test.txt', type=str, help='use tiny datasets to evaluate (.txt)')
    parser.add_argument('--test', default='datasets/test.txt', type=str, help='use tiny datasets to test (.txt)')
    parser.add_argument('--glove', default='', type=str, help='initialize word embedding layer with glove (.txt)')
    parser.add_argument('--w2v', default='', type=str, help='initialize word embedding layer with word2vec (.bin)')
    parser.add_argument('--unit', '-u', default=200, type=int, help='number of units')
    parser.add_argument('--batchsize', '-b', type=int, default=10, help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=100, type=int, help='number of epochs to learn')
    parser.add_argument('--out', default='result-blstm-cnn', help='Directory to output the result')
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

    vocab_word = [UNK_TOKEN]
    vocab_char = [PAD_TOKEN]
    vocab_tag = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    word_emb_size = 200
    char_emb_size = 30

    # Load the pre-trained word embeddings
    pre_embed, pre_vocab = None, None
    if args.w2v:
        pre_embed, pre_vocab = load_w2v_model(args.w2v)
        word_emb_size = pre_embed.shape[1]
    if args.glove:
        pre_embed, pre_vocab = load_glove_model(args.glove)
        word_emb_size = pre_embed.shape[1]

    # Load the dataset
    X_train_words, X_train_chars, y_train, vocab_word, vocab_char, vocab_tag = load_data(args.train, vocab_word, vocab_char, vocab_tag)
    X_valid_words, X_valid_chars, y_valid, vocab_word, vocab_char, vocab_tag = load_data(args.valid, vocab_word, vocab_char, vocab_tag)
    X_test_words,  X_test_chars,  y_test,  vocab_word, vocab_char, vocab_tag = load_data(args.test,  vocab_word, vocab_char, vocab_tag)

    index2tag = {v: k  for k, v in vocab_tag.items()}

    char_vocab_size = len(vocab_char)
    word_vocab_size = len(vocab_word)

    char_lstm_units = 30
    word_lstm_units = 200
    cnn_windows = 3
    cnn_filters = 30

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

    model = BLSTM_CRF_CNN(word_vocab_size, word_emb_size, word_lstm_units, char_vocab_size, char_emb_size, char_lstm_units, num_tags, windows=cnn_windows, filters=cnn_filters)

    # Initialize word embedding layer with word2vec
    if args.w2v or args.glove:
        print('Initialize word embedding by pre-trained model: {}'.format(args.w2v if args.w2v else args.glove))
        w = np.zeros((word_vocab_size, word_emb_size), dtype=np.float32)
        for i in range(word_vocab_size):
            if vocab_word[i] in pre_vocab:
                v = pre_embed[pre_vocab[vocab_word[i]]]
            else:
                v = np.random.uniform(-np.sqrt(3. / word_emb_size), np.sqrt(3. / word_emb_size), (1, word_emb_size)).astype(np.float32)
                v /= np.linalg.norm(v, 2)
            w[i] = v
        model.set_word_embedding(w)
        sys.stdout.flush()

    if args.gpu >= 0:
        model.to_gpu()

    # 学習率
    lr = 0.015

    # 勾配上限
    gradclip = 5

    # L2正則化
    decay = 0.0005

    # 学習率の減衰
    lr_decay = 0.995

    # Setup optimizer (Optimizer の設定)
    # optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=0.9)
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
        train_iter = batch_tuple(sorted_parallel(X_train_words, X_train_chars, y_train, args.batchsize), args.batchsize)

        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        # training
        for x_words_batch, x_chars_batch, t_batch in train_iter:

            # 順伝播させて誤差と精度を算出
            loss = model(x_words_batch, x_chars_batch, t_batch)
            sum_train_loss += float(loss.data) * len(t_batch)

            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                y_pred = model.predict(x_words_batch, x_chars_batch)
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
        val_iter = batch_tuple(sorted_parallel(X_valid_words, X_valid_chars, y_valid, args.batchsize), args.batchsize)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        # evaluation
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for x_words_batch, x_chars_batch, t_batch in val_iter:

                # 順伝播させて誤差と精度を算出
                loss = model(x_words_batch, x_chars_batch, t_batch)
                sum_test_loss += float(loss.data) * len(t_batch)

                y_pred = model.predict(x_words_batch, x_chars_batch)
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
            optimizer.lr)
        )
        sys.stdout.flush()

        # model と optimizer を保存する
        if mean_train_loss < min_loss:
            min_loss = mean_test_loss
            min_epoch = epoch
            print('saving early stopped-model at epoch {}'.format(min_epoch))
            if args.gpu >= 0: model.to_cpu()
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped.model'), model)
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped.state'), optimizer)
            if args.gpu >= 0: model.to_gpu()
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

        optimizer.lr *= lr_decay
        cur_at = now

    # model と optimizer を保存する
    if args.gpu >= 0: model.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.out, 'final.model'), model)
    chainer.serializers.save_npz(os.path.join(args.out, 'final.state'), optimizer)
    if args.gpu >= 0: model.to_gpu()

    # test
    print('loading early stopped-model at epoch {}'.format(min_epoch))
    chainer.serializers.load_npz(os.path.join(args.out, 'early_stopped.model'), model)
    sys.stdout.flush()
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        test_iter = batch_tuple(sorted_parallel(X_test_words, X_test_chars, y_test, len(y_test)), len(y_test))
        for x_words_batch, x_chars_batch, t_batch in test_iter:
            y_pred = model.predict(x_words_batch, x_chars_batch)
            y_tags = [[index2tag[label_id] for label_id in labels] for labels in cuda.to_cpu(y_pred)]
            t_tags = [[index2tag[label_id] for label_id in labels] for labels in cuda.to_cpu(t_batch)]
            print(classification_report(t_tags, y_tags))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
