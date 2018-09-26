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

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
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

    def enc_dec(self, xs, ys, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            eos = self.xp.array([EOS], np.int32)
            ys_in = [F.concat([eos, y], axis=0) for y in ys]

            # Both xs and ys_in are lists of arrays.
            exs = sequence_embed(self.embed_x, xs)
            eys = sequence_embed(self.embed_x, ys_in)

            # None represents a zero vector in an encoder.
            h, c, _ = self.encoder(None, None, exs)
            h, c, ys = self.decoder(h, c, eys)
            cys = F.concat([y[-1:, :] for y in ys], axis=0)
            wy = self.W(cys)
            ys = self.xp.argmax(wy.data, axis=1).astype(np.int32)
            result = [ys]

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
    parser.add_argument('--VOCAB', default='models/word_ids.bin', help='vocabulary file (.bin)')
    parser.add_argument('--validation-source', default='datasets/test.preprocess.en', help='source sentence list for validation')
    parser.add_argument('--validation-target', default='datasets/test.preprocess.de', help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=64, help='number of sentence pairs in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=1024, help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3, help='number of layers')
    parser.add_argument('--model', '-m', default='models/final.model', help='trained model file (.model)')
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

    word_ids = pickle.load(open(args.VOCAB, 'rb'))
    logger.info('Source vocabulary size: %d' % len(word_ids))
    logger.info('Target vocabulary size: %d' % len(word_ids))
    sys.stdout.flush()

    source_words = {i: w for w, i in word_ids.items()}
    target_words = {i: w for w, i in word_ids.items()}

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

    # Setup model
    model = Seq2seq(args.layer, len(word_ids), len(word_ids), args.unit)
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    print('\n### full enc-dec ###\n')

    test_iter = batch_iter(test_data, args.batchsize)

    count = 1
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for batch in test_iter:
            batch = convert(batch, args.gpu)
            xs = batch['xs']
            ys = batch['ys']
            results = model.translate(xs, max_length=100)

            for i in range(len(xs)):
                source_sentence = ' '.join([source_words[x] for x in xs[i].tolist()])
                target_sentence = ' '.join([target_words[t] for t in ys[i].tolist()])
                result_sentence = ' '.join([target_words[y] for y in results[i]])
                score = bleu_score.corpus_bleu([[target_sentence]], [result_sentence], smoothing_function=bleu_score.SmoothingFunction().method1)
                print('No.{}\t(bleu {:.4f})'.format(count, score))
                print(' source: {}'.format(source_sentence))
                print(' result: {}'.format(result_sentence))
                print(' expect: {}'.format(target_sentence))
                count += 1

    print('\n### with preceding words ###\n')

    test_iter = batch_iter(test_data, args.batchsize)

    count = 1
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for batch in test_iter:
            batch = convert(batch, args.gpu)
            xs = batch['xs']
            ys = batch['ys']
            yl = [y[0:1] for y in ys]
            ye = [y[1:]  for y in ys]

            results = model.enc_dec(xs, yl, max_length=100)

            for i in range(len(xs)):
                source_sentence = ' '.join([source_words[x] for x in xs[i].tolist()])
                lead___sentence = ' '.join([target_words[l] for l in yl[i].tolist()])
                result_sentence = ' '.join([target_words[y] for y in results[i]])
                expect_sentence = ' '.join([target_words[e] for e in ye[i].tolist()])
                score = bleu_score.corpus_bleu([[expect_sentence]], [result_sentence], smoothing_function=bleu_score.SmoothingFunction().method1)
                print('No.{}\t(bleu {:.4f})'.format(count, score))
                print(' source: {}'.format(source_sentence))
                print('   lead: {}'.format(lead___sentence))
                print(' result: {}'.format(result_sentence))
                print(' expect: {}'.format(expect_sentence))
                count += 1


if __name__ == '__main__':
    main()
