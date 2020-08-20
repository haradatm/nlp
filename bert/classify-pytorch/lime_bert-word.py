#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
import sys
import time
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


import requests
import pickle
import gzip

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.pipeline import make_pipeline
from marcotcr.lime.lime_text import LimeTextExplainer


import MeCab
m = MeCab.Tagger("-Owakati")


def load_data(filename, labels):
    X, y = [], []

    for i, line in enumerate(open(filename, 'r')):
        # if i >= 100:
        #     continue

        line = line.strip()
        if line == u'':
            continue

        line = line.replace(u'. . .', u'…')

        row = line.split(u'\t')
        if len(row) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        text = row[1]
        text_segmented = m.parse(text).strip()

        X.append(text_segmented)  # Text
        y.append(labels[row[0]])  # Class

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return X, y


class VectorizerWrapper(BaseEstimator, VectorizerMixin):
    def __init__(self, tokenizer, labels):
        super(VectorizerWrapper, self).__init__()
        self.tokenizer = tokenizer
        self.labels = labels

    def fit(self, raw_documents):
        return self

    def transform(self, raw_documents_segmented):
        raw_documents = [s.replace(' ', '') for s in raw_documents_segmented]
        encoded_data = self.tokenizer.batch_encode_plus(raw_documents, pad_to_max_length=True, add_special_tokens=True)
        x1 = torch.tensor(encoded_data["input_ids"])
        x2 = torch.tensor(encoded_data["token_type_ids"])
        x3 = torch.tensor(encoded_data["attention_mask"])
        return x1, x2, x3


def batch_iter(data, batch_size):
    batch = []
    for line in data:
        batch.append(line)
        if len(batch) == batch_size:
            # yield tuple(list(x) for x in zip(*batch))
            yield batch
            batch = []
    if batch:
        # yield tuple(list(x) for x in zip(*batch))
        yield batch


class ClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, batchsize, device):
        super(ClassifierWrapper, self).__init__()
        self.model = model
        self.batchsize = batchsize
        self.device = device

    def fit(self, X, y):
        return self

    def predict(self, X):
        y_pred = []
        test_iter = batch_iter(X, self.batchsize)
        for x1, x2, x3 in test_iter:
            input_ids = x1.to(self.device)
            token_type_ids = x2.to(self.device)
            attention_mask = x3.to(self.device)
            logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            _, preds = torch.max(logits[0], 1)
            y_pred += preds.cpu().numpy().tolist()
        return y_pred

    def predict_proba(self, X):
        y_prob = []
        test_iter = batch_iter(X, self.batchsize)
        for x1, x2, x3 in test_iter:
            input_ids = x1.to(self.device)
            token_type_ids = x2.to(self.device)
            attention_mask = x3.to(self.device)
            logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            probs = F.softmax(logits[0], dim=1)
            y_prob += probs.cpu().numpy().tolist()
        return np.array(y_prob)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    # parser.add_argument('--test', default='datasets/rt-polarity/04-test.txt', type=str, help='evaluating file (.txt)')
    # parser.add_argument('--pretrained', default='bert-base-uncased', type=str, help='pretrained model name or path')
    # parser.add_argument('--model', default='models/rt-polarity/early_stopped-loss.pth.tar', type=str, help='model path')
    parser.add_argument('--test', default='datasets/mlit/04-test.txt', type=str, help='evaluating file (.txt)')
    parser.add_argument('--pretrained', default='cl-tohoku/bert-base-japanese-whole-word-masking', type=str, help='pretrained model name or path')
    parser.add_argument('--model', default='models/mlit/early_stopped-uar.loss.tar', type=str, help='model path')
    parser.add_argument('--batchsize', '-b', default=64, type=int, help='learning batchsize size')
    parser.add_argument('--topN', '-N', default=2, type=int, help='number of top labels')
    parser.add_argument('--out', '-o', default='results_lime-bert-mlit', type=str, help='output file name')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = args.out
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Setup model
    state = torch.load(args.model, map_location=torch.device('cpu'))
    labels = state['labels']
    config = AutoConfig.from_pretrained(args.pretrained, num_labels=len(labels), output_attentions=False, output_hidden_states=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained, config=config)
    model.load_state_dict(state['state_dict'])
    print(model.classifier)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    vectorizer = VectorizerWrapper(tokenizer, labels)
    classifier = ClassifierWrapper(model, args.batchsize, device)

    # テストデータの読み込み
    X_test, y_test = load_data(args.test, labels)
    logger.debug(X_test[0:3])
    logger.debug(y_test[0:3])
    print('# test  X: {}, y: {}, class: {}'.format(len(X_test), len(y_test), len(labels)))
    print('')
    sys.stdout.flush()

    # 訓練モード OFF
    model.eval()
    with torch.no_grad():   # 勾配を計算しない
        # test_vectors = vectorizer.transform(X_test)
        # y_true, y_pred = y_test, classifier.predict(test_vectors)
        # print(classification_report(y_true, y_pred))
        # sys.stdout.flush()

        sorted_labels = [k for k, _ in sorted(labels.items(), key=lambda x: x[1], reverse=False)]
        pipeline = make_pipeline(vectorizer, classifier)
        explainer = LimeTextExplainer(class_names=sorted_labels, char_level=False)

        try:
            # for idx in range(100):
            while True:
                val = input('Enter document ID [0..{}]=> '.format(len(y_test)))
                if val == "":
                    idx = 0
                else:
                    idx = int(val)

                exp = explainer.explain_instance(X_test[idx], pipeline.predict_proba, num_features=20, top_labels=args.topN)
                top_labels = exp.available_labels()

                print('Document id: {}'.format(idx))
                print('True class: {}'.format(sorted_labels[y_test[idx]]))
                for label_id in top_labels:
                    print('Probability ({}) = {:.6f}'.format(sorted_labels[label_id],  pipeline.predict_proba([X_test[idx]])[0, label_id]))
                print()
                sys.stdout.flush()

                for label_id in top_labels:
                    print('Explanation of document id {} for class {}'.format(idx, sorted_labels[label_id]))
                    print('\n'.join(map(str, exp.as_list(label=label_id))))
                    print()
                    sys.stdout.flush()
                    fig = exp.as_pyplot_figure(label=label_id)
                    # fig.savefig(os.path.join(model_dir, 'exp_show-docid_{}-class_{}.png'.format(idx, label_id)))

                exp.save_to_file(os.path.join(output_dir, 'exp_show-docid_{}.html'.format(idx)))

        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    main()
    logger.info('time spent: %06f' % (time.time() - start_time))