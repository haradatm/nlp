#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: Text clustering using a BoW encoder with BERT pre-trained embedding.

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
import matplotlib.cm as cm
import pickle
from sklearn.metrics import confusion_matrix, classification_report


from bertlib.modeling import BertConfig, BertModel
from bertlib.tokenization import FullTokenizer


class BertEmbedding(chainer.Chain):
    def __init__(self, bert):
        super(BertEmbedding, self).__init__()
        with self.init_scope():
            self.bert = bert

    def __call__(self, x1, x2, x3):
        output_layer = self.bert.get_pooled_output(x1, x2, x3)
        return output_layer


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


class BertClassifier(chainer.Chain):
    def __init__(self, bert, num_labels):
        super(BertClassifier, self).__init__()
        with self.init_scope():
            self.bert = bert
            self.output = Linear3D(None, num_labels, initialW=chainer.initializers.Normal(0.02))

    # def forward(self, input_ids, input_mask, token_type_ids):
    #     output_layer = self.bert.get_pooled_output(input_ids, input_mask, token_type_ids)
    #     output_layer = F.dropout(output_layer, 0.1)
    #     logits = self.output(output_layer)
    #     return logits
    #
    # def __call__(self, input_ids, input_mask, token_type_ids, labels):
    #     logits = self.forward(input_ids, input_mask, token_type_ids)
    #     return F.softmax_cross_entropy(logits, labels), F.accuracy(logits, labels)
    #
    # def predict(self, input_ids, input_mask, token_type_ids):
    #     logits = self.forward(input_ids, input_mask, token_type_ids)
    #     return F.softmax(logits)

    def get_embeddings(self, x1, x2, x3):
        output_layer = self.bert.get_pooled_output(x1, x2, x3)
        return output_layer


def load_data(path, tokenizer, bert_config, labels={}, max_length=None):
    X, Y = [], []

    for i, line in enumerate(open(path)):
        if max_length is not None:
            if i >= max_length:
                break

        line = line.strip()
        line = line.replace(u'. . .', u'…')
        if line == '':
            continue

        cols = line.split('\t')

        if len(cols) > 1:
            label, text = cols

            if label not in labels:
                labels[label] = len(labels)
            Y.append(labels[label])
        else:
            text = cols[0]

        text = text.strip()

        tokens_a = tokenizer.tokenize(text)
        tokens = ["[CLS]"]
        segment_ids = [0]

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        max_position_embeddings = bert_config.max_position_embeddings
        x1 = np.array(input_ids[:max_position_embeddings], 'i')
        x2 = np.array(input_mask[:max_position_embeddings], 'f')
        x3 = np.array(segment_ids[:max_position_embeddings], 'i')
        X.append((x1, x2, x3))

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return X, Y, labels


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


def to_device(device, x):
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: Clustering w/BERT')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=64, type=int, help='learning batchsize size')
    parser.add_argument('--label', default='models/classified/rt-polarity/labels.bin', type=str, help='model directory')
    parser.add_argument('--model', default='models/classified/rt-polarity/early_stopped-uar.model', type=str, help='model directory')
    parser.add_argument('--input', default='datasets/rt-polarity/04-train.txt', type=str, help='training file (.txt)')
    parser.add_argument('--init_checkpoint', default='BERT/uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz', type=str, help='initial checkpoint (usually from a pre-trained BERT model (.npz)')
    parser.add_argument('--bert_config_file', default='BERT/uncased_L-12_H-768_A-12/bert_config.json', type=str, help='json file corresponding to the pre-trained BERT model (.json)')
    parser.add_argument('--vocab_file', default='BERT/uncased_L-12_H-768_A-12/vocab.txt', type=str, help='vocabulary file that the BERT model was trained on (.txt)')
    # parser.add_argument('--input', default='datasets/mlit/04-test.txt', type=str, help='training file (.txt)')
    # parser.add_argument('--init_checkpoint', default='BERT/Japanese_L-12_H-768_A-12_E-30_BPE/arrays_bert_model.ckpt.npz', type=str, help='initial checkpoint (usually from a pre-trained BERT model (.npz)')
    # parser.add_argument('--bert_config_file', default='BERT/Japanese_L-12_H-768_A-12_E-30_BPE/bert_config.json', type=str, help='json file corresponding to the pre-trained BERT model (.json)')
    # parser.add_argument('--vocab_file', default='BERT/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt', type=str, help='vocabulary file that the BERT model was trained on (.txt)')
    parser.add_argument('--max_length', default=None, type=int, help='maximum length of source data')
    # parser.set_defaults(test=True)
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 43
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()
        cuda.cupy.random.seed(seed)
        chainer.config.use_cudnn = 'never'

    # 学習済みモデルの読み込み
    vocab_file = args.vocab_file
    bert_config_file = args.bert_config_file
    init_checkpoint = args.init_checkpoint

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    with open(args.label, 'rb') as f:
        label2id = pickle.load(f)

    source_data, source_ids, label2id = load_data(args.input, tokenizer, bert_config, label2id, max_length=args.max_length)
    id2label = {v: k for k, v in label2id.items()}

    logger.info('# source: {}, class: {}, {}'.format(len(source_data), len(label2id), label2id))
    logger.info('# vocab: {}'.format(len(tokenizer.vocab)))
    sys.stdout.flush()

    # Setup model
    bert = BertModel(config=bert_config)
    model = BertClassifier(bert, num_labels=len(label2id))
    chainer.serializers.load_npz(init_checkpoint, model, ignore_names=['output/W', 'output/b'])

    # Loading early_stopped model by uar
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    input_iter = batch_iter(source_data, args.batchsize)

    count = 0
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for x1, x2, x3 in input_iter:
            x1 = to_device(args.gpu, F.pad_sequence(x1, length=None, padding=0).array).astype('i')
            x2 = to_device(args.gpu, F.pad_sequence(x2, length=None, padding=0).array).astype('f')
            x3 = to_device(args.gpu, F.pad_sequence(x3, length=None, padding=0).array).astype('i')
            y = model.get_embeddings(x1, x2, x3)
            features = cuda.to_cpu(y.data)

            for i in range(features.shape[0]):
                print("%s\t" % id2label[source_ids[count + i]] + '\t'.join("%.6f" % x for x in features[i].tolist()))
            sys.stdout.flush()

            count += features.shape[0]


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
