#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Kaggole: Sentiment Analysis on Movie Reviews
    https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
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


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import fasttext as ft
import pickle


class VectorizerWrapper(BaseEstimator, VectorizerMixin):
    def __init__(self):
        super(VectorizerWrapper, self).__init__()

    def fit(self, raw_documents):
        return self

    def transform(self, raw_documents):
        X = raw_documents
        return X


# CLASS_LABEL_dic = {
#     'GPS_Si_Gi': '__label__GPS_Si_Gi',
#     'KWD_So_Gx': '__label__KWD_So_Gx',
# }
#
# CLASS_LABEL_list = [
#     '__label__GPS_Si_Gi',
#     '__label__KWD_So_Gx'
# ]


class ClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, label_list):
        super(ClassifierWrapper, self).__init__()
        self.model = model
        self.label_list = label_list

    def fit(self, X, y):
        return self

    def predict(self, X):
        ys = []
        for x in X:
            labels, probs = self.model.predict(x, k=len(self.label_list))
            ys.append(self.label_list.index(labels[0].strip()))
        return ys

    def predict_proba(self, X):
        y_pred = []
        for x in X:
            labels, probs = self.model.predict(x, k=len(self.label_list))
            p = np.zeros(len(self.label_list), 'f')
            for i, l in enumerate(self.label_list):
                if l not in labels:
                    continue
                p[i] = probs[labels.index(l)]
            y_pred.append(p)
        y_pred = np.array(y_pred)
        return y_pred


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
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Lime example for fastText')
    parser.add_argument('--train', default='datasets/train.txt', type=str, help='training file (.txt)')
    parser.add_argument('--test',  default='datasets/test.txt',  type=str, help='evaluating file (.txt)')
    parser.add_argument('--model', default='models/model.bin',  type=str, help='trained model file (.bin)')
    parser.add_argument('--label', default='models/labels.bin',  type=str, help='trained label file (.bin)')
    parser.add_argument('--topN', '-N', default=1, type=int, help='number of top labels')
    parser.add_argument('--out', '-o',  default='output', type=str, help='output file name')
    # parser.set_defaults(test=True)
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    output_dir = args.out
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 学習済みモデルの読み込み
    model = ft.load_model(args.model)

    # 学習時のラベルデータの読み込み
    with open(args.label, 'rb') as f:
        label_list = pickle.load(f)

    # テストデータの読み込み
    X_test, y_test, label_list = load_data(args.test, labels=label_list)
    logger.debug(X_test[0:3])
    logger.debug(y_test[0:3])

    print_label_list = [x.replace("__label__", "") for x in label_list]

    # テストデータの読み込み
    print('# test  X: {}, y: {}, class: {}'.format(len(X_test), len(y_test), len(print_label_list)))
    print('')
    sys.stdout.flush()

    vectorizer = VectorizerWrapper()
    classifier = ClassifierWrapper(model, label_list)
    test_vectors = vectorizer.transform(X_test)

    y_true, y_pred = y_test, classifier.predict(test_vectors)
    print(classification_report(
        [print_label_list[y] for y in y_true],
        [print_label_list[y] for y in y_pred]
    ))
    sys.stdout.flush()

    pipeline = make_pipeline(vectorizer, classifier)
    explainer = LimeTextExplainer(class_names=print_label_list)

    try:
        while True:
            val = input('Enter document ID [0..{}]=> '.format(len(y_test)))
            if val == "":
                idx = 0
            else:
                idx = int(val)

            exp = explainer.explain_instance(X_test[idx], pipeline.predict_proba, num_features=20, top_labels=args.topN)
            top_labels = exp.available_labels()

            print('Document id: {}'.format(idx))
            print('True class: {}'.format(print_label_list[y_test[idx]]))
            for label_id in top_labels:
                print('Probability ({}) = {:.6f}'.format(print_label_list[label_id], pipeline.predict_proba([X_test[idx]])[0, label_id]))
            print()
            sys.stdout.flush()

            for label_id in top_labels:
                print('Explanation of document id {} for class {}'.format(idx, print_label_list[label_id]))
                print('\n'.join(map(str, exp.as_list(label=label_id))))
                print()
                sys.stdout.flush()
                fig = exp.as_pyplot_figure(label=label_id)
                fig.savefig(os.path.join(output_dir, 'exp_show-docid_{}-class_{}.png'.format(idx, label_id)))

            exp.save_to_file(os.path.join(output_dir, 'exp_show-docid_{}.html'.format(idx)))

    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
