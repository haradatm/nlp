#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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


from nltk.translate import bleu_score

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import pickle


UNK = 0
EOS = 1


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_VOCAB, n_target_vocab, n_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_VOCAB, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_target_vocab)

        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, xs, ys):
        xs = [x[::-1] for x in xs]

        eos = self.xp.array([EOS], np.int32)
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_x, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        hx, cx, _ = self.encoder(None, None, exs)
        _, _, os = self.decoder(hx, cx, eys)

        # It is faster to concatenate data before calculating loss because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch

        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)

        # logger.debug('loss: {}'.format(loss.data))
        # logger.debug('perp: {}'.format(perp))

        return loss, perp, os

    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, c, _ = self.encoder(None, None, exs)
            ys = self.xp.full(batch, EOS, np.int32)

            result = []
            for i in range(max_length):
                eys = self.embed_x(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype(np.int32)
                result.append(ys)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to support NumPy 1.9.
        result = cuda.to_cpu(self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = np.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)

        return outs


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = UNK
    word_ids['<EOS>'] = EOS
    return word_ids


def load_data(vocabulary, path):
    data = []
    logger.info('loading...: %s' % path)
    with open(path) as f:
        for line in f:
            words = line.strip().split()
            array = np.array([vocabulary.get(w, UNK) for w in words], 'i')
            data.append(array)
    return data


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def batch_iter(generator, batch_size):
    batch = []
    for line in generator:
        batch.append(line)
        if len(batch) == batch_size:
            # yield tuple(list(x) for x in zip(*batch))
            yield batch
            batch = []
    if batch:
        # yield tuple(list(x) for x in zip(*batch))
        yield batch


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--SOURCE', default='datasets/soseki.preprocess.en', help='source sentence list')
    parser.add_argument('--TARGET', default='datasets/soseki.preprocess.de', help='target sentence list')
    parser.add_argument('--VOCAB', default='datasets/vocab.txt', help='source vocabulary file')
    parser.add_argument('--validation-source', default='datasets/soseki.preprocess.en', help='source sentence list for validation')
    parser.add_argument('--validation-target', default='datasets/soseki.preprocess.de', help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=64, help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='', help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024, help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3, help='number of layers')
    parser.add_argument('--min-source-sentence', type=int, default=1, help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=9999, help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1, help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=9999, help='maximum length of target sentence')
    parser.add_argument('--log-interval', type=int, default=200, help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=4000, help='number of iteration to evlauate the model with validation dataset')
    parser.add_argument('--out', '-o', default='result', help='directory to output the result')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()
        cuda.cupy.random.seed(seed)

    word_ids = load_vocabulary(args.VOCAB)
    train_source = load_data(word_ids, args.SOURCE)
    train_target = load_data(word_ids, args.TARGET)
    assert len(train_source) == len(train_target)
    train_data = [(s, t) for s, t in zip(train_source, train_target) if args.min_source_sentence <= len(s) <= args.max_source_sentence and args.min_source_sentence <= len(t) <= args.max_source_sentence]
    train_source_unknown = calculate_unknown_ratio([s for s, _ in train_data])
    train_target_unknown = calculate_unknown_ratio([t for _, t in train_data])
    logger.info('Source vocabulary size: %d' % len(word_ids))
    logger.info('Target vocabulary size: %d' % len(word_ids))
    logger.info('Train data size: %d' % len(train_data))
    logger.info('Train source unknown ratio: %.2f%%' % (train_source_unknown * 100))
    logger.info('Train target unknown ratio: %.2f%%' % (train_target_unknown * 100))
    sys.stdout.flush()

    target_words = {i: w for w, i in word_ids.items()}
    source_words = {i: w for w, i in word_ids.items()}

    test_source = load_data(word_ids, args.validation_source)
    test_target = load_data(word_ids, args.validation_target)
    assert len(test_source) == len(test_target)
    test_data = list(zip(test_source, test_target))
    test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]
    test_source_unknown = calculate_unknown_ratio([s for s, _ in test_data])
    test_target_unknown = calculate_unknown_ratio([t for _, t in test_data])
    logger.info('Validation data: %d' % len(test_data))
    logger.info('Validation source unknown ratio: %.2f%%' % (test_source_unknown * 100))
    logger.info('Validation target unknown ratio: %.2f%%' % (test_target_unknown * 100))
    sys.stdout.flush()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    with open(os.path.join(args.out, 'word_ids.bin'), 'wb') as f:
        pickle.dump(word_ids, f)

    # Setup model
    model = Seq2seq(args.layer, len(word_ids), len(word_ids), args.unit)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # Setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

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
        train_iter = batch_iter(train_data, args.batchsize)
        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        for batch in train_iter:
            batch = convert(batch, args.gpu)
            xs = batch['xs']
            ys = batch['ys']

            # 勾配を初期化
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            loss, perp, y = model(xs, ys)

            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                references = [[t.tolist()] for t in ys]
                hypotheses = [y.tolist() for y in model.translate(xs, max_length=100)]
                bleu = bleu_score.corpus_bleu(references, hypotheses, smoothing_function=bleu_score.SmoothingFunction().method1)

            sum_train_loss += float(loss.data) * len(ys)
            sum_train_accuracy1 += float(perp) * len(ys)
            sum_train_accuracy2 += float(bleu) * len(ys)
            K += len(ys)

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
        test_iter = batch_iter(test_data, args.batchsize)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for batch in test_iter:
                batch = convert(batch, args.gpu)
                xs = batch['xs']
                ys = batch['ys']

                # 順伝播させて誤差と精度を算出
                loss, perp, y = model(xs, ys)

                references = [[t.tolist()] for t in ys]
                hypotheses = [y.tolist() for y in model.translate(xs, max_length=100)]
                bleu = bleu_score.corpus_bleu(references, hypotheses, smoothing_function=bleu_score.SmoothingFunction().method1)

            sum_test_loss += float(loss.data) * len(ys)
            sum_test_accuracy1 += float(perp) * len(ys)
            sum_test_accuracy2 += float(bleu) * len(ys)
            K += len(ys)

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

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            source, target = test_data[np.random.choice(len(test_data))]
            result = model.translate([model.xp.array(source)])[0]
            source_sentence = ' '.join([source_words[x] for x in source])
            target_sentence = ' '.join([target_words[t] for t in target])
            result_sentence = ' '.join([target_words[y] for y in result])
            logger.info('# source : '  + source_sentence)
            logger.info('#  result : ' + result_sentence)
            logger.info('#  expect : ' + target_sentence)

        logger.info(''
              '[{:>3d}] '
              'T/loss={:.6f} '
              'T/perp={:.6f} '
              'T/bleu={:.6f} '
              'T/sec= {:.6f} '
              'D/loss={:.6f} '
              'D/perp={:.6f} '
              'D/bleu={:.6f} '
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
                    optimizer.alpha
                )
        )
        sys.stdout.flush()

        # optimizer.lr *= lr_decay

        # model と optimizer を保存する
        if mean_test_accuracy2 > best_accuracy:
            best_accuracy = mean_test_accuracy2
            min_epoch = epoch
            logger.info('saving early stopped-model at epoch {}'.format(min_epoch))
            if args.gpu >= 0: model.to_cpu()
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped.model'), model)
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped.state'), optimizer)
            if args.gpu >= 0: model.to_gpu()
            sys.stdout.flush()

        # 精度と誤差をグラフ描画
        if True:
            ylim1 = [min(train_loss + train_accuracy1 + test_loss + test_accuracy1), max(train_loss + train_accuracy1 + test_loss + test_accuracy1)]
            ylim2 = [min(train_accuracy2 + test_accuracy2), max(train_accuracy2 + test_accuracy2)]

            # グラフ左
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.ylim(ylim1)
            plt.plot(range(1, len(train_loss) + 1), train_loss, 'b')
            plt.plot(range(1, len(train_accuracy1) + 1), train_accuracy1, 'm')
            plt.grid()
            plt.ylabel('loss and perp')
            plt.legend(['train loss', 'train perp'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(train_accuracy2) + 1), train_accuracy2, 'r')
            plt.grid()
            # plt.ylabel('bleu')
            plt.legend(['train bleu'], loc="upper right")
            plt.title('Loss and accuracy of train.')

            # グラフ右
            plt.subplot(1, 2, 2)
            plt.ylim(ylim1)
            plt.plot(range(1, len(test_loss) + 1), test_loss, 'b')
            plt.plot(range(1, len(test_accuracy1) + 1), test_accuracy1, 'm')
            plt.grid()
            # plt.ylabel('loss and perp')
            plt.legend(['dev loss', 'dev perp'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(test_accuracy2) + 1), test_accuracy2, 'r')
            plt.grid()
            plt.ylabel('bleu')
            plt.legend(['dev bleu'], loc="upper right")
            plt.title('Loss and accuracy of dev.')

            plt.savefig('{}.png'.format(args.out))
            # plt.savefig('{}.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()

        cur_at = now

    # model と optimizer を保存する
    logger.info('saving final-model at epoch {}'.format(epoch))
    if args.gpu >= 0: model.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.out, 'final.model'), model)
    chainer.serializers.save_npz(os.path.join(args.out, 'final.state'), optimizer)
    if args.gpu >= 0: model.to_gpu()
    sys.stdout.flush()


if __name__ == '__main__':
    main()
