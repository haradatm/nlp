#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
"""

__version__ = '0.0.1'

import sys, time, logging, os, json, random
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
from chainer.backends import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import matplotlib.pyplot as plt
import collections, pickle


def block_embed(embed, x, dropout=0.):
    e = embed(x)
    e = F.dropout(e, ratio=dropout)
    e = F.transpose(e, (0, 2, 1))
    e = e[:, :, :, None]
    return e


class ContinuousBoW(chainer.Chain):
    """Definition of Continuous Bag of Words Model"""

    def __init__(self, e_vocab, w_vocab, n_units):
        super(ContinuousBoW, self).__init__()

        with self.init_scope():
            self.e_embed = L.EmbedID(e_vocab, n_units, ignore_label=-1, initialW=I.Uniform(1. / n_units))
            self.w_embed = L.EmbedID(w_vocab, n_units, ignore_label=-1, initialW=I.Uniform(1. / n_units))
            self.e_out = L.Linear(n_units, e_vocab, initialW=0)
            self.w_out = L.Linear(n_units, w_vocab, initialW=0)

    def __call__(self, e, w, es, ws):
        ey, wy, ew = self.forward(es, ws)
        loss_e = F.softmax_cross_entropy(ey, e)
        loss_w = F.softmax_cross_entropy(wy, w)
        loss_ew = F.softmax_cross_entropy(ew, e)
        loss = loss_e + loss_w + loss_ew
        accuracy = F.accuracy(F.softmax(ew), e)
        return loss, loss_e, loss_w, loss_ew, accuracy

    def forward(self, es, ws):
        x_block = chainer.dataset.convert.concat_examples(es, padding=-1)
        ex_block = block_embed(self.e_embed, x_block)
        x_len = self.xp.array([len(x) for x in es], np.int32)[:, None, None]
        eh = F.sum(ex_block, axis=2) / x_len

        x_block = chainer.dataset.convert.concat_examples(ws, padding=-1)
        ex_block = block_embed(self.w_embed, x_block)
        x_len = self.xp.array([len(x) for x in ws], np.int32)[:, None, None]
        wh = F.sum(ex_block, axis=2) / x_len

        ey = self.e_out(eh)
        wy = self.w_out(wh)
        ew = self.e_out(wh)
        return ey, wy, ew


def load_kg(path, kg=None, entity2id={}):
    if kg is None:
        kg = collections.defaultdict(set)

    for i, line in enumerate(open(path, 'r')):
        line = line.strip()

        if line == '':
            continue

        entity1, entity2, relation = line.split('\t')
        entity1 = entity1.strip().lower()
        entity2 = entity2.strip().lower()

        if entity1 == entity2:
            # logger.warning("[SKIP] Self-relation entity: {}, relation: {}".format(entity1, entity2, relation))
            continue

        for e in [entity1, entity2]:
            if e not in entity2id:
                entity2id[e] = len(entity2id)

        kg[entity2id[entity1]].add(entity2id[entity2])
        kg[entity2id[entity2]].add(entity2id[entity1])

    return kg, entity2id


EOS_TOKEN = '</s>'


import nltk
nltk.download('stopwords')

def stopwords():
    symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')', '*', '--', '\\']
    stopwords = nltk.corpus.stopwords.words('english')
    return stopwords + symbols


def load_data(path, window, kg, entity2id, mention2id={}, mid2name={}):
    dataset = []

    for i, line in enumerate(open(path, 'r')):
        # if i > 1000:
        #     break

        line = line.strip()
        if line == '':
            continue

        line = line.replace("###END###", EOS_TOKEN).strip()

        entity1, entity2, mention1, mention2, _, context = line.split('\t')[0:6]
        mention1 = mention1.strip().lower()
        mention2 = mention2.strip().lower()
        entity1  = entity1.strip().lower()
        entity2  = entity2.strip().lower()
        context  = context.strip().lower().split(' ')
        if context == '':
            continue

        context = [w for w in context if w not in stopwords()]
        if len(context) < 1:
            continue

        for word in [mention1, mention2]:
            if word not in mention2id:
                mention2id[word] = len(mention2id)

        if entity1 in entity2id:
            entity_id = entity2id[entity1]
            mention_id = mention2id[mention1]
            entity_ids = list(kg[entity_id])

            if mention1 in context:
                center = context.index(mention1)
                context_l = context[max(0, center - window): center]
                context_r = context[center + 1: min(center + window + 1, len(context))]
                word_ids = []
                for word in context_l + context_r:
                    if word not in mention2id:
                        mention2id[word] = len(mention2id)
                    word_ids.append(mention2id[word])
                dataset.append((entity_id, mention_id, entity_ids, word_ids))

        if entity2 in entity2id:
            entity_id = entity2id[entity2]
            mention_id = mention2id[mention2]
            entity_ids = list(kg[entity_id])

            if mention2 in context:
                center = context.index(mention2)
                context_l = context[max(0, center - window): center]
                context_r = context[center + 1: min(center + window + 1, len(context))]
                word_ids = []
                for word in context_l + context_r:
                    if word not in mention2id:
                        mention2id[word] = len(mention2id)
                    word_ids.append(mention2id[word])
                dataset.append((entity_id, mention_id, entity_ids, word_ids))

        if entity1 not in mid2name:
            mid2name[entity1] = mention1
        if entity2 not in mid2name:
            mid2name[entity2] = mention2

    return dataset, mention2id, mid2name


from sklearn.utils import shuffle as skshuffle


def get_minibatches(dataset, window, batch_size, shuffle=True):
    batch = []
    shuffled_data = np.copy(dataset)
    if shuffle:
        shuffled_data = skshuffle(shuffled_data)

    for line in shuffled_data:
        entity_id, mention_id, entity_ids, word_ids = line
        if len(entity_ids) > window * 2:
            entity_ids = random.sample(entity_ids, window * 2)

        batch.append((entity_id, mention_id, entity_ids, word_ids))

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: Word embedding model')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=100, type=int, help='number of units')
    parser.add_argument('--window', '-w', default=5, type=int, help='window size')
    parser.add_argument('--batchsize', '-b', type=int, default=1500, help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=50, type=int, help='number of epochs to learn')
    parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'], default='cbow', help='model type ("skipgram", "cbow")')
    parser.add_argument('--negative-size', default=5, type=int, help='number of negative samples')
    parser.add_argument('--out-type', '-o', choices=['hsm', 'ns', 'original'], default='original', help='output model type ("hsm": hierarchical softmax, "ns": negative sampling, "original": no approximation)')
    parser.add_argument('--kg', default='../datasets/nyt-fb60k/kg/train.txt', type=str, help='training kg (.txt)')
    parser.add_argument('--train', default='../datasets/nyt-fb60k/text/train.txt', type=str, help='training file (.txt)')
    parser.add_argument('--valid', default='../datasets/nyt-fb60k/text/test.txt', type=str, help='validation file (.txt)')
    parser.add_argument('--out', default='result-kg2v-3', help='Directory to output the result')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=True)
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))

    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()
        cuda.cupy.random.seed(seed)
        chainer.config.use_cudnn = 'never'

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('Window: {}'.format(args.window))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Training model: {}'.format(args.model))
    print('Output type: {}'.format(args.out_type))
    print('')
    sys.stdout.flush()

    # Load the dataset
    kg, entity2id = load_kg(args.kg)
    train, word2id, mid2name = load_data(args.train, args.window, kg, entity2id)
    val,   word2id, mid2name = load_data(args.valid, args.window, kg, entity2id, word2id, mid2name)
    
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    with open(os.path.join(args.out, 'kg.pkl'), 'wb') as f:
        pickle.dump(kg, f)
    with open(os.path.join(args.out, 'train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(args.out, 'val.pkl'), 'wb') as f:
        pickle.dump(val, f)
    with open(os.path.join(args.out, 'entity2id.bin'), 'wb') as f:
        pickle.dump(entity2id, f)
    with open(os.path.join(args.out, 'mention2id.bin'), 'wb') as f:
        pickle.dump(word2id, f)
    with open(os.path.join(args.out, 'mid2name.bin'), 'wb') as f:
        pickle.dump(mid2name, f)

    if args.test:
        train = train[:1000]
        val = val[:1000]

    e_vocab = len(entity2id)
    w_vocab = len(word2id)

    logger.info('entity size: %d' % e_vocab)
    logger.info('vocabulary size: %d' % w_vocab)
    logger.info('train data length: %d' % len(train))
    logger.info('vaid  data length: %d' % len(val))
    sys.stdout.flush()

    # Model Parameters
    model = ContinuousBoW(e_vocab, w_vocab, args.unit)

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
    train_loss_e = []
    train_loss_w = []
    train_loss_ew = []
    train_accuracy1 = []
    train_accuracy2 = []
    test_loss = []
    test_loss_e = []
    test_loss_w = []
    test_loss_ew = []
    test_accuracy1 = []
    test_accuracy2 = []

    min_loss = float('inf')
    best_accuracy = .0

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
        sum_train_loss_e = 0.
        sum_train_loss_w = 0.
        sum_train_loss_ew = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        # training
        for x1, x2, x3, x4 in train_iter:
            N = len(x1)
            x1 = to_device(args.gpu, np.array(x1, 'i'))
            x2 = to_device(args.gpu, np.array(x2, 'i'))
            x3 = to_device(args.gpu, ([np.array(x, 'i') for x in x3]))
            x4 = to_device(args.gpu, ([np.array(x, 'i') for x in x4]))

            # 誤差逆伝播で勾配を計算 (minibatch ごと)
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            loss, loss_e, loss_w, loss_ew, accuracy = model(x1, x2, x3, x4)
            sum_train_loss += float(loss.data) * N
            sum_train_loss_e += float(loss_e.data) * N
            sum_train_loss_w += float(loss_w.data) * N
            sum_train_loss_ew += float(loss_ew.data) * N
            sum_train_accuracy1 += float(accuracy.data) * N
            sum_train_accuracy2 += .0
            K += N

            loss.backward()
            optimizer.update()

        # 訓練データの誤差と,正解精度を表示
        mean_train_loss = sum_train_loss / K
        mean_train_loss_e = sum_train_loss_e / K
        mean_train_loss_w = sum_train_loss_w / K
        mean_train_loss_ew = sum_train_loss_ew / K
        mean_train_accuracy1 = sum_train_accuracy1 / K
        mean_train_accuracy2 = sum_train_accuracy2 / K
        train_loss.append(mean_train_loss)
        train_loss_e.append(mean_train_loss_e)
        train_loss_w.append(mean_train_loss_w)
        train_loss_ew.append(mean_train_loss_ew)
        train_accuracy1.append(mean_train_accuracy1)
        train_accuracy2.append(mean_train_accuracy2)
        now = time.time()
        train_throughput = now - cur_at
        cur_at = now

        # Set up an iterator
        val_iter = get_minibatches(val, args.window, args.batchsize)
        sum_test_loss = 0.
        sum_test_loss_e = 0.
        sum_test_loss_w = 0.
        sum_test_loss_ew = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        # evaluation
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for x1, x2, x3, x4 in val_iter:
                N = len(x1)
                x1 = to_device(args.gpu, np.array(x1, 'i'))
                x2 = to_device(args.gpu, np.array(x2, 'i'))
                x3 = to_device(args.gpu, ([np.array(x, 'i') for x in x3]))
                x4 = to_device(args.gpu, ([np.array(x, 'i') for x in x4]))

                # 順伝播させて誤差と精度を算出
                loss, loss_e, loss_w, loss_ew, accuracy = model(x1, x2, x3, x4)
                sum_test_loss += float(loss.data) * N
                sum_test_loss_e += float(loss_e.data) * N
                sum_test_loss_w += float(loss_w.data) * N
                sum_test_loss_ew += float(loss_ew.data) * N
                sum_test_accuracy1 += float(accuracy.data) * N
                sum_test_accuracy2 += .0
                K += N

        # テストデータでの誤差と正解精度を表示
        mean_test_loss = sum_test_loss / K
        mean_test_loss_e = sum_test_loss_e / K
        mean_test_loss_w = sum_test_loss_w / K
        mean_test_loss_ew = sum_test_loss_ew / K
        mean_test_accuracy1 = sum_test_accuracy1 / K
        mean_test_accuracy2 = sum_test_accuracy2 / K
        test_loss.append(mean_test_loss)
        test_loss_e.append(mean_test_loss_e)
        test_loss_w.append(mean_test_loss_w)
        test_loss_ew.append(mean_test_loss_ew)
        test_accuracy1.append(mean_test_accuracy1)
        test_accuracy2.append(mean_test_accuracy2)
        now = time.time()
        test_throughput = now - cur_at

        logger.info(''
                    '[{:>3d}] '
                    'T/loss={:.6f} '
                    'T/loss_e={:.6f} '
                    'T/loss_w={:.6f} '
                    'T/loss_ew={:.6f} '
                    'T/accuracy={:.6f} '
                    # 'T/acc2={:.6f} '
                    'T/sec= {:.6f} '
                    'D/loss={:.6f} '
                    'D/loss_e={:.6f} '
                    'D/loss_w={:.6f} '
                    'D/loss_ew={:.6f} '
                    'D/accuracy={:.6f} '
                    # 'D/acc2={:.6f} '
                    'D/sec= {:.6f} '
                    'lr={:.6f}'
                    ''.format(
            epoch,
            mean_train_loss,
            mean_train_loss_e,
            mean_train_loss_w,
            mean_train_loss_ew,
            mean_train_accuracy1,
            # mean_train_accuracy2,
            train_throughput,
            mean_test_loss,
            mean_test_loss_e,
            mean_test_loss_w,
            mean_test_loss_ew,
            mean_test_accuracy1,
            # mean_test_accuracy2,
            test_throughput,
            optimizer.alpha)
        )
        sys.stdout.flush()

        # model と optimizer を保存する
        if args.gpu >= 0: model.to_cpu()
        if mean_test_loss < min_loss:
            min_loss = mean_test_loss
            print('saving early-stopped model (loss) at epoch {}'.format(epoch))
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped-loss.model'), model)
        if mean_test_accuracy1 > best_accuracy:
            best_accuracy = mean_test_accuracy1
            print('saving early-stopped model (accuracy) at epoch {}'.format(epoch))
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped-accuracy.model'), model)
        # print('saving final model at epoch {}'.format(epoch))
        chainer.serializers.save_npz(os.path.join(args.out, 'final.model'), model)
        chainer.serializers.save_npz(os.path.join(args.out, 'final.state'), optimizer)
        if args.gpu >= 0: model.to_gpu()
        sys.stdout.flush()

        # 精度と誤差をグラフ描画
        if True:
            ylim1 = [min(train_loss + train_loss_e + train_loss_w + train_loss_ew + test_loss + test_loss_e + test_loss_w + test_loss_ew),
                     max(train_loss + train_loss_e + train_loss_w + train_loss_ew + test_loss + test_loss_e + test_loss_w + test_loss_ew)]
            # ylim2 = [min(train_accuracy1 + test_accuracy1), max(train_accuracy1 + test_accuracy1)]
            ylim2 = [0, 1]

            # グラフ左
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.ylim(ylim1)
            plt.plot(range(1, len(train_loss) + 1), train_loss, color='C1', marker='x')
            plt.plot(range(1, len(train_loss_e) + 1), train_loss_e, color='C2', marker='x')
            plt.plot(range(1, len(train_loss_w) + 1), train_loss_w, color='C3', marker='x')
            plt.plot(range(1, len(train_loss_ew) + 1), train_loss_ew, color='C4', marker='x')
            plt.grid(False)
            plt.ylabel('loss and accuracy')
            plt.legend(['train loss', 'train loss_e', 'train loss_w', 'train loss_ew'], loc="lower left")
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
            plt.plot(range(1, len(test_loss) + 1), test_loss, color='C1', marker='x')
            plt.plot(range(1, len(test_loss_e) + 1), test_loss_e, color='C2', marker='x')
            plt.plot(range(1, len(test_loss_w) + 1), test_loss_w, color='C3', marker='x')
            plt.plot(range(1, len(test_loss_ew) + 1), test_loss_ew, color='C4', marker='x')
            plt.grid(False)
            # plt.ylabel('loss and accuracy')
            plt.legend(['test loss', 'test loss_e', 'test loss_w', 'test loss_ew'], loc="lower left")
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
            plt.close()

        # optimizer.alpha *= lr_decay
        cur_at = now


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
