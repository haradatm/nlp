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
from sklearn.metrics.pairwise import cosine_similarity


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: Word embedding model')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=100, type=int, help='number of units')
    parser.add_argument('--saved_entity', default="model/entity2id.bin", type=str, help='')
    parser.add_argument('--saved_vocabulary', default="model/mention2id.bin", type=str, help='')
    parser.add_argument('--saved_mid2name', default="model/mid2name.bin", type=str, help='')
    parser.add_argument('--saved_model', default="model/early_stopped.model", type=str, help='')
    parser.add_argument('--query', '-q', default='', help='word for query')
    parser.add_argument('--N', default=50, type=int, help='')
    parser.add_argument('--max_eval', default=50, type=str, help='number of evaluations')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()

    xp = cuda.cupy if args.gpu >= 0 else np

    # Set random seed
    xp.random.seed(123)

    # Load the models
    entity2id  = pickle.load(open(args.saved_entity, 'rb'))
    mention2id = pickle.load(open(args.saved_vocabulary, 'rb'))
    mid2name   = pickle.load(open(args.saved_mid2name, 'rb'))

    e_vocab = len(entity2id)
    w_vocab = len(mention2id)

    id2word   = {id: w for w, id in mention2id.items()}
    id2entity = {id: e for e, id in entity2id.items()}

    # Model Parameters
    model = ContinuousBoW(e_vocab, w_vocab, args.unit)

    # test (early_stopped model by loss)
    chainer.serializers.load_npz(args.saved_model, model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # w_vecs = cuda.to_cpu(model.w_embed.W.data)
    w_vecs = cuda.to_cpu(model.w_out.W.data)
    # e_vecs = cuda.to_cpu(model.e_embed.W.data)
    e_vecs = cuda.to_cpu(model.e_out.W.data)

    logger.info("mention: size={}, dim={}".format(w_vecs.shape[0], w_vecs.shape[1]))
    logger.info("entity : size={}, dim={}".format(e_vecs.shape[0], e_vecs.shape[1]))
    sys.stdout.flush()

    query = args.query if args.query else ""

    while True:
        plot_vecs, plot_labels = [], []

        if query == "":
            try:
                val = input('Enter query (mention) => ')
                if val == "":
                    continue
                query = val.strip()
                if query not in mention2id:
                    continue
            except KeyboardInterrupt:
                return

        q_vec = w_vecs[mention2id[query]]

        print("#query\t#rank\t#word\t#similarity")
        similarities = cosine_similarity(q_vec[None, :], w_vecs)
        for j, k in enumerate(similarities[0].argsort()[-1:-(args.max_eval + 1):-1]):
            plot_vecs.append(w_vecs[k])
            plot_labels.append(id2word[k])
            if j < args.N:
                print("{}\t{}\t{}\t{:.6f}".format(query, j+1, id2word[k], similarities[0][k]))
                sys.stdout.flush()
        print(); sys.stdout.flush()

        plot_vecs.append(q_vec)
        plot_labels.append(query)

        print("#query\t#rank\t#entity(mid)\t#similarity")
        similarities = cosine_similarity(q_vec[None, :], e_vecs)
        for j, k in enumerate(similarities[0].argsort()[-1:-(args.max_eval + 1):-1]):
            plot_vecs.append(e_vecs[k])
            plot_labels.append("{} ({})".format(mid2name[id2entity[k]], id2entity[k]))
            if j < args.N:
                print("{}\t{}\t{} ({})\t{:.6f}".format(query, j+1, mid2name[id2entity[k]], id2entity[k], similarities[0][k]))
                sys.stdout.flush()
        print(); sys.stdout.flush()

        from sklearn.manifold import TSNE
        X_reduced = TSNE(n_components=2, random_state=0).fit_transform(plot_vecs)
        # from sklearn.decomposition import PCA
        # X_reduced = PCA(n_components=2).fit_transform(plot_vecs)
        # from sklearn.decomposition import TruncatedSVD
        # X_reduced = TruncatedSVD(n_components=2).fit_transform(plot_vecs)

        plt.figure(figsize=(10, 10))

        for i in range(1, args.max_eval):
            plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c='C0', marker='.')
            plt.annotate(plot_labels[i], X_reduced[i]+[.1, .0], color='C0')

        i = args.max_eval
        plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c='C1', marker='x')
        plt.annotate(plot_labels[i], X_reduced[i]+[.1, .1], color='C1')

        for i in range(args.max_eval + 1, args.max_eval*2 + 1):
            plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c='C2', marker='.')
            plt.annotate(plot_labels[i], X_reduced[i]+[.1, -.3], color='C2')

        plt.title("t-SNE plot ({})".format(query))
        plt.savefig("plot_{}.png".format(query))
        plt.close()

        if args.query != "":
            break
        query = ""


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
