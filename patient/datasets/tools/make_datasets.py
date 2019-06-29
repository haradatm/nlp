#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, time, logging, re, os
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


import collections


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('file', type=str, help='input file (.txt)')
    args = parser.parse_args()

    q_ids, d_ids = [], []
    f_q = open("{}.queries".format(os.path.splitext(os.path.basename(args.file))[0]), 'w')
    f_d = open(   "{}.docs".format(os.path.splitext(os.path.basename(args.file))[0]), 'w')
    f_r = open(   "{}.qrel".format(os.path.splitext(os.path.basename(args.file))[0]), 'w')

    qrels = collections.defaultdict(lambda: [])

    for line in open(args.file, 'r'):
        line = line.strip()
        if line == '':
            continue

        cols = line.split('\t')
        if len(cols) < 2:
            continue

        q, d = cols[0:2]
        # q_ids.append(q)
        # d_ids.append(d)

        # f_q.write("Q-{:04d}\t{}\n".format(len(q_ids), q))
        # f_d.write("D-{:04d}\t{}\n".format(len(d_ids), d))
        # f_r.write("Q-{:04d}\t0\tD-{:04d}\t{}\n".format(len(q_ids), len(d_ids), 1))
        # f_q.flush(); f_d.flush(); f_r.flush()

        if q not in q_ids:
            q_ids.append(q)

        if d not in d_ids:
            d_ids.append(d)

        qrels[q_ids.index(q)].append(d_ids.index(d))

    for q in range(len(q_ids)):
        f_q.write("Q-{:04d}\t{}\n".format(q, q_ids[q]))
        f_q.flush()

    for d in range(len(d_ids)):
        f_d.write("D-{:04d}\t{}\n".format(d, d_ids[d]))
        f_d.flush()

    for q, ds in sorted(qrels.items(), key=lambda x: x[0]):
        for d in ds:
            f_r.write("Q-{:04d}\t0\tD-{:04d}\t{}\n".format(q, d, 1))
            f_r.flush()


sys.stderr.write('time spent: {}\n'.format(time.time() - start_time))
