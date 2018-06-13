#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, time, logging, re, json
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

    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('file', default='train.tsv', type=str, help='training file (.txt)')
    args = parser.parse_args()

    X, y = [], []

    for i, line in enumerate(open(args.file, 'rU')):
        line = line.strip()
        if line == "":
            continue

        row = line.split('\t')
        if len(row) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        if i != 0 and not i % 3:
            print("{:}\t{:}\t{:}\t{:}".format(len(X), '\t'.join(['\t'.join(x) for x in X]), '\t'.join(y), y[-1]))
            X.clear()
            y.clear()

        X.append(row[0:-1])
        y.append(row[-1])

    if len(X):
        print("{:}\t{:}\t{:}\t{:}".format(len(X), '\t'.join(['\t'.join(x) for x in X]), '\t'.join(y), y[-1]))


if __name__ == '__main__':
    main()
