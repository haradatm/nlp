#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stratified k-fold Splitter for a Japanese Text Classification Dataset
"""

__version__ = '0.0.1'

import sys, time, logging
import numpy as np
np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def pp(obj):
    import pprint
    pp = pprint.PrettyPrinter(indent=1, width=160)
    print(pp.pformat(obj))


start_time = time.time()


import time, six
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Stratified k-fold Splitter')
    parser.add_argument('file', default='', type=str, help='training file (.txt)')
    parser.add_argument('-N', default=10,  type=int, help='number of folds')
    args = parser.parse_args()

    path = args.file
    N = args.N

    data = []
    target = []

    for i, line in enumerate(open(path, 'r')):
        line = line.strip()
        if line == '':
            continue

        cols = line.split('\t')
        if len(cols) < 2:
            logger.error('invalid record: {}\n'.format(line))
            continue

        data.append(cols[1].strip())
        target.append(cols[0].strip())

    for i, (train_idx, test_idx) in enumerate(StratifiedKFold(n_splits=N, shuffle=True).split(data, target)):

        logger.debug('#train: {:}, #test: {:}, #total: {:}'.format(len(train_idx), len(test_idx), len(train_idx) + len(test_idx)))

        with open('{:02d}-train.txt'.format(i+1), 'w') as f_train:
            for j in range(len(train_idx)):
                f_train.write('{}\t{}\n'.format(target[train_idx[j]], data[train_idx[j]]))

        with open('{:02d}-test.txt'.format(i+1), 'w') as f_test:
            for j in range(len(test_idx)):
                f_test.write('{}\t{}\n'.format(target[test_idx[j]], data[test_idx[j]]))

logger.info('time spent: {}\n'.format(time.time() - start_time))
