#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
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


def main():
    global xp

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='word2vec.model', type=str)
    # parser.set_defaults(test=True)
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))

    n_result = 5  # number of search result to show

    with open(args.model, 'r') as f:
        ss = f.readline().split()
        n_vocab, n_units = int(ss[0]), int(ss[1])
        word2index = {}
        index2word = {}
        w = np.empty((n_vocab, n_units), dtype=np.float32)
        for i, line in enumerate(f):
            ss = line.split()
            assert len(ss) == n_units + 1
            word = ss[0]
            word2index[word] = i
            index2word[i] = word
            w[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

    s = np.sqrt((w * w).sum(1))
    w /= s.reshape((s.shape[0], 1))  # normalize

    try:
        while True:
            q = input('>> ')
            if q not in word2index:
                print('"{0}" is not found'.format(q))
                continue
            v = w[word2index[q]]
            similarity = w.dot(v)
            print('query: {}'.format(q))
            count = 0
            for i in (-similarity).argsort():
                if np.isnan(similarity[i]):
                    continue
                if index2word[i] == q:
                    continue
                print('{0}: {1}'.format(index2word[i], similarity[i]))
                count += 1
                if count == n_result:
                    break

    except EOFError:
        pass


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
