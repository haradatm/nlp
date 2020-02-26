#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: Text clustering using a BoW encoder with BERT pre-trained embedding.

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

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from collections import Counter


def main():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input',  default='features/rt-dml-04-test.txt', type=str, help='input file (.txt)')
    parser.add_argument('--output', default='result.png', type=str, help='output file (.png)')
    # parser.set_defaults(test=True)
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 43
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    X, y, label2id = [], [], {}
    for i, line in enumerate(open(args.input)):
        cols = line.strip().split("\t")
        label = cols[0]
        X.append(np.array(list(map(lambda x: float(x), cols[1:])), 'f'))

        if label not in label2id:
            label2id[label] = len(label2id)
        y.append(label2id[label])

    # id2label = {v: k for k, v in label2id.items()}

    print('# input: {}, class: {}, labels: {}'.format(len(X), len(label2id), label2id))
    sys.stdout.flush()

    tsne = TSNE(n_components=2, random_state=0).fit_transform(X)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']), int(len(label2id) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])

    plt.figure()
    plt.scatter(tsne[:, 0], tsne[:, 1], s=10, color=colors[y])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(args.output)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
