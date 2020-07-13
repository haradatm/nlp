#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import logging
import os
import random
import sys
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

import glob, json, math, datetime
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import fasttext as ft


def load_data(filename, labels=[]):
    X, y = [], []

    for i, line in enumerate(open(filename, 'r')):
        # if i == 0:
        #     continue

        line = line.strip()
        if line == "":
            continue

        row = line.split(u',')
        if len(row) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        label = row[0].strip()
        text = row[1].strip()

        if label not in labels:
            labels.append(label)

        X.append(text)                  # Text
        y.append(labels.index(label))   # Class

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return X, y, labels


def main():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', default='datasets/train.txt', type=str, help='training file (.txt)')
    parser.add_argument('--test',  default='datasets/test.txt',  type=str, help='evaluating file (.txt)')
    # parser.add_argument('--analyzer', default='word', choices=['char', 'word'], help='type of analyzer')
    parser.add_argument('--out', '-o', default='models', type=str, help='output file name')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if args.gpu >= 0:
    #     torch.cuda.manual_seed_all(seed)

    output_dir = args.out
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 文章分類器の作成・学習
    print('Start time: {0}'.format(datetime.datetime.today()))
    classifier = ft.train_supervised(input=args.train, dim=100, epoch=20, loss='hs')
    results = classifier.test(args.test)
    print(results)
    print("End time: {0}".format(datetime.datetime.today()))

    classifier.save_model(os.path.join(output_dir, 'model.bin'))
    label_list = classifier.get_labels()
    with open(os.path.join(output_dir, 'labels.bin'), 'wb') as f:
        pickle.dump(label_list, f)

    # テストデータの読み込み
    X_test, y_test, label_list = load_data(args.test, labels=label_list)
    logger.debug(X_test[0:3])
    logger.debug(y_test[0:3])

    print_label_list = [x.replace("__label__", "") for x in label_list]

    # 文章分類器を用いたラベリング
    print('Start time: {0}'.format(datetime.datetime.today()))
    classifier = ft.load_model(os.path.join(output_dir, 'model.bin'))
    labels, probs = classifier.predict(X_test, k=1)
    print("End time: {0}".format(datetime.datetime.today()))
    sys.stdout.flush()

    y_true = [print_label_list[y] for y in y_test]
    y_pred = [y[0].replace("__label__", "") for y in labels]
    y_prob = [y[0] for y in probs]
    print(classification_report(y_true, y_pred))
    sys.stdout.flush()

    # for y, p, t in zip(y_pred, y_prob, y_true):
    #     print("%s\t%.6f\t%s" % (y, p, t))
    # sys.stdout.flush()


if __name__ == '__main__':
    main()
