#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Kaggole: Sentiment Analysis on Movie Reviews
    https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
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

# import csv
import nltk


def load_data(filename, labels={}):
    X, y = [], []

    for i, line in enumerate(open(filename, 'rU')):
        # if i == 0:
        #     continue

        line = line.strip()
        if line == u'':
            continue

        line = line.replace(u'. . .', u'…')

        row = line.split(u'\t')
        if len(row) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        if row[0] not in labels:
            labels[row[0]] = len(labels)

        X.append(row[1])            # Text
        y.append(labels[row[0]])    # Class

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return X, y, labels


def stopwords():
    symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')', '*', '--', '\\']
    stopwords = nltk.corpus.stopwords.words('english')
    return stopwords + symbols


def analyzer(text):
    return text.split(' ')


from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='LinearSVC')
    parser.add_argument('--train', default='train.tsv', type=str, help='training file (.txt)')
    parser.add_argument('--test',  default='test.tsv',  type=str, help='evaluating file (.txt)')
    parser.add_argument('--analyzer',  default='word', choices=['char', 'word'], help='type of analyzer')
    parser.add_argument('--type',    default='lsvc', choices=['lsvc', 'lgb', 'xgb'], help='type of classifier')
    args = parser.parse_args()

    # 訓練データの読み込み
    X, y, labels = load_data(args.train)
    logger.debug(X[0:3])
    logger.debug(y[0:3])

    # トレーニングデータとテストデータに分割
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.05, random_state=123)

    print('# train X: {}, y: {}, class: {}'.format(len(X_train), len(y_train), len(labels)))
    print('# eval  X: {}, y: {}, class: {}'.format(len(X_eval), len(y_eval), len(labels)))
    print('')
    sys.stdout.flush()

    def get_classifier(clf_type):

        if clf_type == "lgb":
            import lightgbm as lgb
            return lgb.LGBMClassifier()

        elif clf_type == "xgb":
            import xgboost as xgb
            return xgb.XGBClassifier()

        else:
            from sklearn.svm import LinearSVC
            return LinearSVC(C=1, loss='squared_hinge')

    pipeline = Pipeline([
        ('vec', TfidfVectorizer(analyzer=args.analyzer, binary=False, stop_words=stopwords(), ngram_range=(1,2), min_df=1, max_df=1.0, smooth_idf=True, use_idf=True, sublinear_tf=False)),
        # ('svd', TruncatedSVD(n_components=300, random_state=123, algorithm='arpack')),
        ('clf', get_classifier(args.type))
    ])

    parameters = {
        'vec__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vec__min_df': [1, 2],
        'vec__max_df': [50, 1.0],
        'vec__stop_words': [None],
        # 'vec__smooth_idf': [False, True],
        # 'vec__use_idf': [False, True],
        # 'vec__sublinear_tf': [False, True],
        # 'vec__binary': [False, True],
    }

    print('# Tuning hyper-parameters for %s' % 'accuracy')

    cv = ShuffleSplit(n_splits=10, test_size=0.10, random_state=123)
    clf = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, cv=cv, verbose=1, scoring='accuracy')

    clf.fit(X_train, y_train)
    print('Best parameters set found on development set:')
    print('')
    print(clf.best_params_)
    print('')
    print('Grid scores on development set:')
    print('')
    sys.stdout.flush()

    means  = clf.cv_results_['mean_test_score']
    stds   = clf.cv_results_['std_test_score']
    params = clf.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        logger.debug('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, param))

    print('Detailed classification report:')
    print('')
    print('The model is trained on the full development set.')
    print('The scores are computed on the full evaluation set.')
    print('')
    sys.stdout.flush()

    y_true, y_pred = y_eval, clf.predict(X_eval)
    print(classification_report(y_true, y_pred))

    print('')
    sys.stdout.flush()

    # すべての訓練データで fit
    clf = clf.best_estimator_
    clf.fit(X, y)

    # テストデータの読み込み
    X_test, y_test, labels = load_data(args.test, labels=labels)
    logger.debug(X_test[0:3])
    logger.debug(y_test[0:3])

    print('# test  X: {}, y: {}, class: {}'.format(len(X_test),  len(y_test),  len(labels)))
    print('')
    sys.stdout.flush()

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    sys.stdout.flush()

    # with open('submit.csv', 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n')
    #     writer.writerow(['PhraseId', 'Sentiment'])
    #     for index, id in enumerate(ids):
    #         survived = results[index]
    #         writer.writerow ([id, survived])

print('time spent: ', time.time() - start_time)
