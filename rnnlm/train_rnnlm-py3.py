#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Sample script of recurrent neural network language model.

    usage: python3.6 train_rnnlm.py --gpu -1 --epoch 200 --batchsize 100 --unit 300 --train datasets/soseki/neko-wakachi.txt --w2v datasets/soseki/neko_w2v.bin --out model-neko
    usage: python3.6  test_rnnlm.py --gpu -1 --model "model-neko/final.model" --text "吾 輩 は 猫 で ある 。"
"""

__version__ = '0.0.1'

import sys, os, time, logging, json, math
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
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import pickle
from struct import unpack, calcsize

# UNK_ID = 0
# EOS_ID = 1
UNK_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'

prime_text = ""


def load_w2v_model(path, vocab=[]):

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
                vocab += [token]
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


def load_data(filename, vocab, w2v=None):
    global prime_text

    dataset = []

    for i, line in enumerate(open(filename, 'r')):
        line = line.strip()
        tokens = line.split(' ') + [EOS_TOKEN]

        if i == 0:
            prime_text = line.split(' ')

        for token in tokens:
            if token == '':
                continue

            if token not in vocab:
                vocab += [token]
                if w2v is not None:
                    v = np.random.uniform(-0.1, 0.1, (1, w2v.shape[1])).astype(np.float32)
                    v /= np.linalg.norm(v, 2)
                    w2v = np.vstack((w2v, v))

            dataset.append(vocab.index(token))

    return dataset, vocab, w2v


# Definition of a recurrent net for language modeling
class RNNLM(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        super(RNNLM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.l1 = L.LSTM(n_units, n_units)
            self.l2 = L.LSTM(n_units, n_units)
            self.l3 = L.Linear(n_units, n_vocab)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, x, t):
        y = self.forward(x)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    # 1ステップ前方処理関数 (学習データ,状態を与える)
    def forward(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0))
        h2 = self.l2(F.dropout(h1))
        y = self.l3(F.dropout(h2))
        return y

    def predict(self, x):
        y = self.forward(x)
        return F.softmax(y)

    # 状態の初期化 (初期状態を現在の状態にセット)
    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def set_word_embedding(self, data):
        self.embed.W.data = data


def test(model, vocab, token2id, text, length=20):
    model.reset_state()

    for token in list(text):
        sys.stdout.write(token)
        prev_word = model.predict(xp.array([token2id[token]], dtype=np.int32))

    for i in range(length):
        # idx = np.argmax(cuda.to_cpu(prev_word.data))
        next_prob = cuda.to_cpu(prev_word.data)[0].astype(np.float64)
        next_prob /= np.sum(next_prob)
        idx = np.random.choice(range(len(next_prob)), p=next_prob)

        if vocab[idx] == EOS_TOKEN:
            sys.stdout.write(EOS_TOKEN)
        else:
            sys.stdout.write(vocab[idx])
        prev_word = model.predict(xp.array([idx], dtype=np.int32))

    sys.stdout.write('\n')
    sys.stdout.flush()


def main():
    global xp

    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: RNNLM')
    parser.add_argument('--train', default='datasets/soseki/neko-char.txt', type=str, help='training file (.txt)')
    parser.add_argument('--w2v', '-w', default='', type=str, help='initialize word embedding layer with word2vec (.bin)')
    parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35, help='Number of words in each mini-batch (= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=200, help='Number of sweeps over the dataset to train')
    parser.add_argument('--unit', '-u', type=int, default=300, help='Number of LSTM units in each layer')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5, help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='results', help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true', help='Use tiny datasets for quick tests')
    # parser.set_defaults(test=True)
    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np
    xp.random.seed(123)

    if args.w2v:
        w2v, vocab = load_w2v_model(args.w2v)
        n_dims = w2v.shape[1]
        train_data, vocab, w2v = load_data(args.train, vocab, w2v)
    else:
        vocab = []
        n_dims = args.unit
        train_data, vocab, _ = load_data(args.train, vocab)

    train_length = len(train_data)

    token2id = {}
    for i, token in enumerate(vocab):
        token2id[token] = i

    logger.info('Train vocabulary size: %d' % len(vocab))
    logger.info('Train data size: %d' % train_length)
    logger.info('Train data starts with: {} ...'.format(' '.join(prime_text)))
    sys.stdout.flush()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    with open(os.path.join(args.out, 'vocab.bin'), 'wb') as f:
        pickle.dump(vocab, f)

    # Recurrent neural net languabe model
    model = RNNLM(len(vocab), n_dims)

    # 学習率
    lr = 0.0007

    # 重み減衰
    decay = 0.0005

    # 学習率の減衰
    lr_decay = 0.995

    # Setup optimizer (Optimizer の設定)
    optimizer = chainer.optimizers.Adam(alpha=lr)
    # optimizer = optimizers.AdaDelta()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

    # Resume the training from snapshot
    if args.resume:
        print('Resume the training from snapshot: {0}.{{model,state}}'.format(args.resume))
        chainer.serializers.load_npz('{}.model'.format(args.resume), model)
        chainer.serializers.load_npz('{}.state'.format(args.resume), optimizer, strict=False)
        sys.stdout.flush()

    # Initialize word embedding layer with word2vec
    if not args.resume and args.w2v:
        print('Initialize the embedding from word2vec model: {}'.format(args.w2v))
        model.set_word_embedding(w2v)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # プロット用に実行結果を保存する
    train_loss = []
    train_accuracy1 = []
    train_accuracy2 = []
    min_loss = float('inf')
    min_epoch = 0

    # スライド幅を計算する
    width = int(train_length / args.batchsize)

    # 最初の時間情報を取得する
    start_at = time.time()
    cur_at = start_at

    # Learning loop
    print("going to train {} iterations".format(width * args.epoch))

    # training
    epoch = 1
    accum_loss = None
    sum_train_loss = 0.
    sum_train_accuracy1 = 0.
    sum_train_accuracy2 = 0.
    K = 0

    # RNN 状態を初期化する
    model.reset_state()

    for iteration in range(width * args.epoch):

        # logger.info('epoch {:} / {:}'.format(epoch, n_epoch))
        # handler1.flush()

        x_batch = xp.array([train_data[(width * x + iteration) % train_length] for x in range(args.batchsize)])
        y_batch = xp.array([train_data[(width * x + iteration + 1) % train_length] for x in range(args.batchsize)])

        # 順伝播させて誤差と精度を算出
        loss, accuracy = model(x_batch, y_batch)
        accum_loss = loss if accum_loss is None else accum_loss + loss
        sum_train_loss += float(loss.data)
        sum_train_accuracy1 += float(accuracy.data)
        sum_train_accuracy2 += math.exp(float(loss.data))
        K += 1

        # 誤差逆伝播で勾配を計算 (bproplen ごと)
        if (iteration + 1) % args.bproplen == 0:
            model.cleargrads()
            accum_loss.backward()
            accum_loss.unchain_backward()
            optimizer.update()
            # optimizer.alpha *= lr_decay

        # 訓練データの誤差と,正解精度を表示 (epoch ごと)
        if (iteration + 1) % width == 0:
            mean_train_loss = sum_train_loss / K
            mean_train_accuracy1 = sum_train_accuracy1 / K
            mean_train_accuracy2 = sum_train_accuracy2 / K
            train_loss.append(mean_train_loss)
            train_accuracy1.append(mean_train_accuracy1)
            train_accuracy2.append(mean_train_accuracy2)
            now = time.time()
            train_throughput = now - cur_at

            logger.info(''
                  '[{:>3d}] '
                  'T/loss={:.6f} '
                  'T/acc={:.6f} '
                  'T/perp={:.6f} '
                  'T/sec= {:.6f} '
                  'lr={:.6f}'
                  ''.format(
                        epoch,
                        mean_train_loss,
                        mean_train_accuracy1,
                        mean_train_accuracy2,
                        train_throughput,
                        optimizer.alpha
                    )
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

            if args.test:
                with chainer.no_backprop_mode(), chainer.using_config('train', False):
                    test(model.copy(), vocab, token2id, prime_text)

            # 精度と誤差をグラフ描画
            if True:
                # ylim1 = [min(train_loss), max(train_loss)]
                # ylim2 = [min(train_accuracy1), max(train_accuracy1)]
                # ylim3 = [min(train_accuracy2), max(train_accuracy2)]

                # グラフ左
                plt.figure(figsize=(10, 10))
                plt.subplots_adjust(wspace=0.4)
                plt.subplot(1, 2, 1)
                # plt.ylim(ylim1)
                plt.plot(range(1, len(train_loss) + 1), train_loss, color='C0', marker='')
                plt.grid(False)
                plt.ylabel('train loss')
                plt.legend(['train loss'], loc="lower left")
                plt.twinx()
                # plt.ylim(ylim2)
                plt.plot(range(1, len(train_accuracy1) + 1), train_accuracy1, color='C1', marker='')
                plt.grid(False)
                plt.ylabel('train accuracy')
                plt.legend(['train accuracy'], loc="upper right")
                plt.title('Loss and accuracy of train.')

                # グラフ右
                plt.subplot(1, 2, 2)
                plt.grid(False)
                plt.twinx()
                # plt.ylim(ylim3)
                plt.plot(range(1, len(train_accuracy2) + 1), train_accuracy2, color='C2', marker='')
                plt.grid(False)
                plt.ylabel('perplexity')
                plt.legend(['train perplexity'], loc="upper right")
                plt.title('Perplexity of train.')

                plt.savefig('{}.png'.format(args.out))
                # plt.savefig('{}.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
                # plt.show()

            epoch += 1
            sum_train_loss = 0.
            sum_train_accuracy1 = 0.
            sum_train_accuracy2 = 0.
            K = 0

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

    vocab = pickle.load(open(os.path.join(args.out, 'vocab.bin'), 'rb'))
    token2id = {}
    for i, token in enumerate(vocab):
        token2id[token] = i

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        test(model, vocab, token2id, prime_text)


if __name__ == '__main__':
    main()
