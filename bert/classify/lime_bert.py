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


import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer


def load_data(filename, labels):
    X, y = [], []

    for i, line in enumerate(open(filename, 'r')):
        # if i >= 10:
        #     continue

        line = line.strip()
        if line == u'':
            continue

        line = line.replace(u'. . .', u'…')

        row = line.split(u'\t')
        if len(row) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        X.append(row[1])  # Text
        y.append(labels[row[0]])  # Class

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return X, y


class VectorizerWrapper(BaseEstimator, VectorizerMixin):
    def __init__(self, tokenizer, bert_config, labels, type_id=0):
        super(VectorizerWrapper, self).__init__()
        self.tokenizer = tokenizer
        self.bert_config = bert_config
        self.labels = labels
        self.type_id = type_id

    def fit(self, raw_documents):
        return self

    def transform(self, raw_documents):
        X = []

        for i, text in enumerate(raw_documents):
            # if i >= 10:
            #     break

            tokens_a = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] if self.type_id == 0 else []
            segment_ids = [self.type_id] if self.type_id == 0 else []

            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)

            tokens.append("[SEP]")
            segment_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            max_position_embeddings = self.bert_config.max_position_embeddings
            x1 = np.array(input_ids[:max_position_embeddings], 'i')
            x2 = np.array(input_mask[:max_position_embeddings], 'f')
            x3 = np.array(segment_ids[:max_position_embeddings], 'i')
            X.append((x1, x2, x3))

        return X


class ClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, batchsize, gpu=-1):
        super(ClassifierWrapper, self).__init__()
        self.model = model
        self.batchsize = batchsize
        self.gpu = gpu

    def fit(self, X, y):
        return self

    def predict(self, X):
        ys = []
        test_iter = batch_iter(X, self.batchsize)
        for x1, x2, x3 in test_iter:
            # x1, x2, x3 = tuple(list(x) for x in zip(*X))
            x1 = to_device(self.gpu, F.pad_sequence(x1, length=None, padding=0).array).astype('i')
            x2 = to_device(self.gpu, F.pad_sequence(x2, length=None, padding=0).array).astype('f')
            x3 = to_device(self.gpu, F.pad_sequence(x3, length=None, padding=0).array).astype('i')
            y = self.model.predict(x1, x2, x3)
            ys += np.argmax(cuda.to_cpu(y.data), axis=1).tolist()
        return ys

    def predict_proba(self, X):
        p = []
        test_iter = batch_iter(X, self.batchsize)
        for x1, x2, x3 in test_iter:
            x1 = to_device(self.gpu, F.pad_sequence(x1, length=None, padding=0).array).astype('i')
            x2 = to_device(self.gpu, F.pad_sequence(x2, length=None, padding=0).array).astype('f')
            x3 = to_device(self.gpu, F.pad_sequence(x3, length=None, padding=0).array).astype('i')
            y = self.model.predict(x1, x2, x3)
            p += cuda.to_cpu(y.data).tolist()
        y_pred = np.array(p)
        return y_pred


def batch_iter(data, batch_size):
    batch = []
    for line in data:
        batch.append(line)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch))
            # yield batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch))
        # yield batch


from bertlib.modeling import BertConfig, BertModel
from bertlib.tokenization import FullTokenizer
from bertlib.optimization import WeightDecayForMatrixAdam


class Linear3D(L.Linear):
    def __init__(self, *args, **kwargs):
        super(Linear3D, self).__init__(*args, **kwargs)

    def call(self, x):
        return super(Linear3D, self).__call__(x)

    def __call__(self, x):
        if x.ndim == 2:
            return self.call(x)
        assert x.ndim == 3

        x_2d = x.reshape((-1, x.shape[-1]))
        out_2d = self.call(x_2d)
        out_3d = out_2d.reshape(x.shape[:-1] + (out_2d.shape[-1],))
        # (B, S, W)
        return out_3d


# Network definition
class BertClassifier(chainer.Chain):
    def __init__(self, bert, num_labels):
        super(BertClassifier, self).__init__()
        with self.init_scope():
            self.bert = bert
            self.output = Linear3D(None, num_labels, initialW=chainer.initializers.Normal(0.02))

    def forward(self, input_ids, input_mask, token_type_ids):
        output_layer = self.bert.get_pooled_output(input_ids, input_mask, token_type_ids)
        output_layer = F.dropout(output_layer, 0.1)
        logits = self.output(output_layer)
        return logits

    def __call__(self, input_ids, input_mask, token_type_ids, labels):
        logits = self.forward(input_ids, input_mask, token_type_ids)
        return F.softmax_cross_entropy(logits, labels), F.accuracy(logits, labels)

    def predict(self, input_ids, input_mask, token_type_ids):
        logits = self.forward(input_ids, input_mask, token_type_ids)
        return F.softmax(logits)


def to_device(device, x):
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='lLIME for BERT')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=16, type=int, help='learning batchsize size')
    parser.add_argument('--label', default='models/rt-polarity/labels.bin', type=str, help='model directory')
    parser.add_argument('--model', default='models/rt-polarity/early_stopped-uar.model', type=str, help='model directory')
    parser.add_argument('--test', default='datasets/rt-polarity/04-test.txt', type=str, help='evaluating file (.txt)')
    parser.add_argument('--init_checkpoint', default='BERT/uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz', type=str, help='initial checkpoint (usually from a pre-trained BERT model (.npz)')
    parser.add_argument('--bert_config_file', default='BERT/uncased_L-12_H-768_A-12/bert_config.json', type=str, help='json file corresponding to the pre-trained BERT model (.json)')
    parser.add_argument('--vocab_file', default='BERT/uncased_L-12_H-768_A-12/vocab.txt', type=str, help='vocabulary file that the BERT model was trained on (.txt)')
    parser.add_argument('--topN', '-N', default=1, type=int, help='number of top labels')
    parser.add_argument('--out', '-o', default='results_lime-bert-rt', type=str, help='output file name')
    # parser.set_defaults(test=True)
    # args = parser.parse_args()
    args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()
        cuda.cupy.random.seed(seed)
        chainer.config.use_cudnn = 'never'

    output_dir = args.out
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(args.label, 'rb') as f:
        labels = pickle.load(f)

    # 学習済みモデルの読み込み
    vocab_file = args.vocab_file
    bert_config_file = args.bert_config_file
    init_checkpoint = args.init_checkpoint

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    print('# class: {}, labels: {}'.format(len(labels), labels))
    print('# vocab: {}'.format(len(tokenizer.vocab)))
    sys.stdout.flush()

    # Setup model
    bert = BertModel(config=bert_config)
    model = BertClassifier(bert, num_labels=len(labels))
    chainer.serializers.load_npz(init_checkpoint, model, ignore_names=['output/W', 'output/b'])

    # test (early_stopped model by uar)
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    vectorizer = VectorizerWrapper(tokenizer, bert_config, labels)
    classifier = ClassifierWrapper(model, args.batchsize, args.gpu)

    # 訓練データの読み込み
    X_test, y_test = load_data(args.test, labels)
    logger.debug(X_test[0:3])
    logger.debug(y_test[0:3])

    # トレーニングデータとテストデータに分割
    print('# test  X: {}, y: {}, class: {}'.format(len(X_test), len(y_test), len(labels)))
    print('')
    sys.stdout.flush()

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        test_vectors = vectorizer.transform(X_test)

        y_true, y_pred = y_test, classifier.predict(test_vectors)
        print(classification_report(y_true, y_pred))
        sys.stdout.flush()

        sorted_labels = [k for k, _ in sorted(labels.items(), key=lambda x: x[1], reverse=False)]
        pipeline = make_pipeline(vectorizer, classifier)
        explainer = LimeTextExplainer(class_names=sorted_labels)

        try:
            # for idx in range(100):
            while True:
                val = input('Enter document ID [0..{}]=> '.format(len(y_test)))
                if val == "":
                    idx = 0
                else:
                    idx = int(val)

                exp = explainer.explain_instance(X_test[idx], pipeline.predict_proba, num_features=20,
                                                 top_labels=args.topN)
                top_labels = exp.available_labels()

                print('Document id: {}'.format(idx))
                print('True class: {}'.format(sorted_labels[y_test[idx]]))
                for label_id in top_labels:
                    print('Probability ({}) = {:.6f}'.format(sorted_labels[label_id],
                                                             pipeline.predict_proba([X_test[idx]])[0, label_id]))
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
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
