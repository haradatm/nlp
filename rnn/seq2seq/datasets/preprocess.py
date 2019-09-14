#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, time, logging, os, json, re
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
    parser = argparse.ArgumentParser(description='Chainer example: BLSTM-CRF')
    parser.add_argument('file', default="", type=str, help='input file name (*.txt)')
    parser.add_argument('-N', default=3, type=int, help='number of previous sentences')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    buff = []
    for line in open(args.file, 'r'):
        line = line.strip()
        buff.append(line)

    for i in range(1, args.N):
        for j in range(0, len(buff) - i):
            enc = ' '.join(buff[j:j+i])
            dec = buff[j+i]
            print("{}\t{}".format(enc, dec))


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
