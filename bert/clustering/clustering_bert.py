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
    parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=64, type=int, help='learning batchsize size')
    parser.add_argument('--model', default='models', type=str, help='model directory')
    # parser.add_argument('--input', default='datasets/rt-polarity/04-train.txt', type=str, help='training file (.txt)')
    # parser.add_argument('--init_checkpoint', default='BERT/uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz', type=str, help='initial checkpoint (usually from a pre-trained BERT model (.npz)')
    # parser.add_argument('--bert_config_file', default='BERT/uncased_L-12_H-768_A-12/bert_config.json', type=str, help='json file corresponding to the pre-trained BERT model (.json)')
    # parser.add_argument('--vocab_file', default='BERT/uncased_L-12_H-768_A-12/vocab.txt', type=str, help='vocabulary file that the BERT model was trained on (.txt)')
    parser.add_argument('--input', default='datasets/mlit/04-test.txt', type=str, help='training file (.txt)')
    parser.add_argument('--init_checkpoint', default='BERT/Japanese_L-12_H-768_A-12_E-30_BPE/arrays_bert_model.ckpt.npz', type=str, help='initial checkpoint (usually from a pre-trained BERT model (.npz)')
    parser.add_argument('--bert_config_file', default='BERT/Japanese_L-12_H-768_A-12_E-30_BPE/bert_config.json', type=str, help='json file corresponding to the pre-trained BERT model (.json)')
    parser.add_argument('--vocab_file', default='BERT/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt', type=str, help='vocabulary file that the BERT model was trained on (.txt)')
    parser.add_argument('--K', '-K', default=None, type=int, help='number of cluster')
    parser.add_argument('--max_length', default=None, type=int, help='maximum length of source data')
    # parser.add_argument('--out', '-o', default='results_bert-rt-all', type=str, help='output prefix')
    parser.add_argument('--out', '-o', default='results_bert-mlit-all', type=str, help='output prefix')
    parser.add_argument('--noplot', action='store_true', help='disable PlotReport extension')
    # parser.set_defaults(test=True)
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))
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

    # model_dir = args.model
    # if not os.path.exists(model_dir):
    #     os.mkdir(model_dir)

    vocab_file = args.vocab_file
    bert_config_file = args.bert_config_file
    init_checkpoint = args.init_checkpoint

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    label2id = {}

    source_data, source_ids, label2id = load_data(args.input, tokenizer, bert_config, label2id, max_length=args.max_length)
    id2label = {v: k for k, v in label2id.items()}

    K = args.K if args.K is not None else len(label2id)
    sys.stdout.flush()

    print('# source: {}, id: {}, {}'.format(len(source_data), len(label2id), label2id))
    print('# vocab: {}'.format(len(tokenizer.vocab)))
    print('# K: {}'.format(K))
    sys.stdout.flush()

    # Setup model
    bert = BertModel(config=bert_config)
    model = BertEmbedding(bert)
    chainer.serializers.load_npz(init_checkpoint, model, ignore_names=['output/W', 'output/b'])

    if args.gpu >= 0:
        model.to_gpu()

    input_iter = batch_iter(source_data, args.batchsize)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        outputs = []
        for x1, x2, x3 in input_iter:
            x1 = to_device(args.gpu, F.pad_sequence(x1, length=None, padding=0).array).astype('i')
            x2 = to_device(args.gpu, F.pad_sequence(x2, length=None, padding=0).array).astype('f')
            x3 = to_device(args.gpu, F.pad_sequence(x3, length=None, padding=0).array).astype('i')
            y = model(x1, x2, x3)
            outputs.append(cuda.to_cpu(y.data))

    features = np.concatenate(outputs, axis=0)

    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=K, random_state=seed).fit(features)

    if len(source_ids) > 0:
        print("cluster_id\tsource_id")
        for cluster_id, source_id in zip(kmeans_model.labels_, source_ids):
            print("{}\t{}".format(cluster_id, source_id))
    else:
        print("cluster_id")
        for cluster_id in kmeans_model.labels_:
            print("{}".format(cluster_id))

    if not args.noplot:
        from sklearn.manifold import TSNE
        tsne_model = TSNE(n_components=2, random_state=0).fit_transform(features)

        if len(source_ids) > 0:
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.title("Input class")
            for x, y, name in zip(tsne_model[:, 0], tsne_model[:, 1], [id2label[x] for x in source_ids]):
            # for x, y, name in zip(tsne_model[:, 0], tsne_model[:, 1], source_ids):
                plt.text(x, y, name, alpha=0.8, size=10)
            plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=source_ids)
            # plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.title("K-means cluster")
            for x, y, name in zip(tsne_model[:, 0], tsne_model[:, 1], [id2label[x] for x in source_ids]):
            # for x, y, name in zip(tsne_model[:, 0], tsne_model[:, 1], source_ids):
                plt.text(x, y, name, alpha=0.8, size=10)
            plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=kmeans_model.labels_)
            # plt.colorbar()

        else:
            plt.figure(figsize=(10, 10))
            plt.title("K-means cluster")
            plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=kmeans_model.labels_)
            # plt.colorbar()

        plt.savefig('{}.png'.format(args.out))
        # plt.show()
        plt.close()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
