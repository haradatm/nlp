#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Sample script of recurrent neural network language model. (using NStep-LSTM)

    usage: python3.6 train_rnnlm.py --gpu -1 --epoch 200 --batchsize 100 --unit 300 --train datasets/soseki/neko-word-train.txt --test datasets/soseki/neko-word-test.txt --w2v datasets/soseki/neko_w2v.bin --out model-neko
    usage: python3.6  test_rnnlm.py --gpu -1 --model "model-neko/final.model" --text "吾輩 は 猫 で ある 。"
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
import pickle

# UNK_ID = 0
# EOS_ID = 1
# UNK_TOKEN = '<unk>'
EOS_TOKEN = '</s>'


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


def to_device_batch(batch, device):
    if device is None:
        return batch
    elif device < 0:
        return [chainer.dataset.to_device(device, x) for x in batch]
    else:
        xp = cuda.cupy.get_array_module(*batch)
        concat = xp.concatenate(batch, axis=0)
        sections = xp.cumsum([len(x) for x in batch[:-1]], dtype='i')
        concat_dev = chainer.dataset.to_device(device, concat)
        batch_dev = cuda.cupy.split(concat_dev, sections)
        return batch_dev


# Definition of a recurrent net for language modeling
class RNNLM(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_units):
        super(RNNLM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_source_vocab, n_units)
            self.l1 = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.l2 = L.Linear(n_units, n_source_vocab)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, xs, ys):
        hx, cx, os = self.forward(xs)
        concat_os = self.l2(F.concat(os, axis=0))
        concat_ys_out = F.concat(ys, axis=0)

        batch = len(xs)
        n_words = concat_ys_out.shape[0]

        loss = F.sum(F.softmax_cross_entropy(concat_os, concat_ys_out, reduce='no')) / batch
        accuracy = F.accuracy(concat_os, concat_ys_out)
        perplexity = xp.exp(loss.data * batch / n_words)

        return loss, accuracy, perplexity

    def forward(self, xs, hx=None, cx=None):
        exs = sequence_embed(self.embed, xs)
        hx, cx, os = self.l1(hx=hx, cx=cx, xs=exs)
        return hx, cx, os

    def predict(self, xs, hx=None, cx=None):
        hx, cx, os = self.forward(xs, hx=hx, cx=cx)
        y = self.l2(F.concat(os, axis=0))
        return hx, cx, F.softmax(y)

    def set_word_embedding(self, data):
        self.embed.W.data = data


def main():
    global xp

    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: NStep RNNLM')
    parser.add_argument('--model', '-m', type=str, default='model/final.model', help='model data, saved by train.py')
    parser.add_argument('--text', '-t', type=str, default='吾 輩 は 猫 で あ る', help='base text data, used for text generation')
    parser.add_argument('--unit', '-u', type=int, default=200, help='number of dimensions')
    parser.add_argument('--layer', '-l', type=int, default=3, help='number of layers')
    parser.add_argument('--sample', type=int, default=1, help='negative value indicates NOT use random choice')
    parser.add_argument('--length', type=int, default=2000, help='length of the generated text')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    # print(json.dumps(args.__dict__, indent=2))

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np
    xp.random.seed(123)

    vocab = pickle.load(open(os.path.join(os.path.dirname(args.model), 'vocab.bin'), 'rb'))
    token2id = {}
    for i, token in enumerate(vocab):
        token2id[token] = i

    logger.info('Number of units: {}'.format(args.unit))
    logger.info('Vocabulary size: {}'.format(len(vocab)))

    # Recurrent neural net languabe model
    model = RNNLM(args.layer, len(vocab), args.unit)
    chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):

        prime_text = args.text.strip().split(' ')

        for token in prime_text:
            sys.stdout.write(token)

        hx, cx, prev_word = model.predict([xp.array([token2id[x] for x in prime_text], dtype=np.int32)])

        for i in range(args.length):
            if args.sample > 0:
                next_prob = cuda.to_cpu(prev_word.data)[-1].astype(np.float64)
                next_prob /= np.sum(next_prob)
                idx = np.random.choice(range(len(next_prob)), p=next_prob)
            else:
                idx = np.argmax(cuda.to_cpu(prev_word.data)[-1])

            if vocab[idx] == EOS_TOKEN:
                sys.stdout.write('\n')
                sys.stdout.flush()
            else:
                sys.stdout.write(vocab[idx])
                sys.stdout.flush()
            hx, cx, prev_word = model.predict([xp.array([idx], dtype=np.int32)], hx=hx, cx=cx)

        sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
