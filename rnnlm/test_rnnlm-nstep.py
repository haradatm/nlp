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


def make_candidates(candidates, beam_width):
    next_candidates = []

    for model, hx, cx, token_ids, likelihood in candidates:
        hx, cx, y = model.predict([xp.array(token_ids, dtype=np.int32)], hx=hx, cx=cx)
        y = model.predict(xp.array([token_ids[-1]], dtype=np.int32))
        next_prob = cuda.to_cpu(y.data)[0].astype(np.float64)
        next_prob /= np.sum(next_prob)
        next_likelihood = np.log(next_prob)

        # 上位 beam_width 個の枝を残す
        # order = np.argsort(next_prob)[::-1][:beam_width]
        order = np.random.choice(range(len(next_prob)), beam_width, p=next_prob)

        for i in order:
            ll = (likelihood * len(token_ids) + next_likelihood[i]) / (len(token_ids) + 1)
            next_candidates.append((model.copy(), hx.copy(), cx.copy(), token_ids + [i], ll))

        # 全ての枝の中から対数尤度の上位 beam_width 個を残す
        candidates = sorted(next_candidates, key=lambda x: -x[2])[:beam_width]

    return candidates


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
    token2id = {v: k for k, v in enumerate(vocab)}

    logger.info('Number of units: {}'.format(args.unit))
    logger.info('Vocabulary size: {}'.format(len(vocab)))

    # Recurrent neural net languabe model
    model = RNNLM(args.layer, len(vocab), args.unit)
    chainer.serializers.load_npz(args.model, model)

    beam_width = 5

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):

        prime_text = args.text.strip().split(' ')
        token_ids = [token2id[x] for x in prime_text]

        hx, cx, prev_word = model.predict([xp.array(token_ids, dtype=np.int32)])

        candidates = [(model.copy(), hx.copy(), cx.copy(), token_ids, 0)]

        for i in range(args.length):
            candidates = make_candidates(candidates, beam_width)

        for x in candidates[0][1][1:]:
            if x != token2id[EOS_TOKEN]:
                print(vocab[x], end='')
            else:
                print()
        sys.stdout.flush()


if __name__ == '__main__':
    main()
