#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys

reload(sys)
sys.setdefaultencoding('utf-8')
#print sys.getdefaultencoding()

import re
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

import pprint
def pp(obj):
    pp = pprint.PrettyPrinter(indent=1, width=160)
    str = pp.pformat(obj)
    print re.sub(r"\\u([0-9a-f]{4})", lambda x: unichr(int("0x"+x.group(1),16)), str)

import os, time
start_time = time.time()

import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
np.set_printoptions(precision=20)


def load_data(path, labels={}):
    X, Y = [], []

    for i, line in enumerate(open(path, 'rU')):
        # if i >= 100:
        #     break

        line = unicode(line).strip()
        if line == u'':
            continue

        line = line.replace(u'. . .', u'…')

        cols = line.split(u'\t')
        if len(cols) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        label = cols[0]
        text  = cols[1]
        X.append(text)

        if label not in labels:
            labels[label] = len(labels)
        Y.append(labels[label])

    print('Loading dataset ... done.')
    sys.stdout.flush()

    return X, Y, labels


import nltk
def stopwords():
    symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')', '*', '--', '\\']
    stopwords = nltk.corpus.stopwords.words('english')
    return stopwords + symbols


from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='N-gram')
    parser.add_argument('--train', default='train.txt', type=unicode, help='training file (.txt)')
    parser.add_argument('--test',  default='',  type=unicode, help='evaluating file (.txt)')
    parser.add_argument('--analyzer',  default='char', choices=['char', 'word'], help='type of analyzer')
    args = parser.parse_args()

    # データの読み込み
    if not args.test:
        # トレーニング+テストデータ
        X, y, labels = load_data(args.train)

        # トレーニングデータとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=123)

    else:
        # トレーニングデータ
        X_train, y_train, labels = load_data(args.train)

        # テストデータ
        X_test, y_test, labels = load_data(args.test, labels=labels)

    print('# train X: {}, y: {}, class: {}'.format(len(X_train), len(y_train), len(labels)))
    print('# test  X: {}, y: {}, class: {}'.format(len(X_test),  len(y_test),  len(labels)))
    sys.stdout.flush()

    features = TfidfVectorizer(analyzer=args.analyzer, binary=False, stop_words=stopwords(), ngram_range=(1,2), min_df=1, max_df=1.0, smooth_idf=True, use_idf=True, sublinear_tf=False)
    pipeline = Pipeline([('vect', features), ('clf', LinearSVC())])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__min_df': [1, 2],
        'vect__max_df': [50, 1.0],
        'vect__stop_words': [None],
        # 'vect__smooth_idf': [False, True],
        # 'vect__use_idf': [False, True],
        # 'vect__sublinear_tf': [False, True],
        # 'vect__binary': [False, True],
    }

    print('# Tuning hyper-parameters for %s' % 'accuracy')
    print('')

    cv = ShuffleSplit(n_splits=10, test_size=0.10, random_state=123)
    clf = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1, cv=cv, verbose=0, scoring='accuracy')
    # pp(clf.get_params().keys())

    clf.fit(X_train, y_train)
    print('Best parameters set found on development set:')
    print('')
    print(clf.best_params_)
    print('')
    print('Grid scores on development set:')
    print('')

    means  = clf.cv_results_['mean_test_score']
    stds   = clf.cv_results_['std_test_score']
    params = clf.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, param))
    print('')

    print('Detailed classification report:')
    print('')
    print('The model is trained on the full development set.')
    print('The scores are computed on the full evaluation set.')
    print('')

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    print('')

print('time spent: ', time.time() - start_time)
