#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: Text classifier using a BoW encoder with BERT pre-trained embedding.

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


def load_data(path, tokenizer, bert_config, labels={}, type_id=0):
    data = []

    for i, line in enumerate(open(path)):
        # if i >= 10:
        #     break

        line = line.strip()
        line = line.replace(u'. . .', u'…')
        if line == '':
            continue

        label, text = line.split('\t')

        if label not in labels:
            labels[label] = len(labels)

        tokens_a = tokenizer.tokenize(text)
        tokens = ["[CLS]"] if type_id == 0 else []
        segment_ids = [type_id] if type_id == 0 else []

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        label_id = labels[label]

        max_position_embeddings = bert_config.max_position_embeddings
        x1 = np.array(input_ids[:max_position_embeddings], 'i')
        x2 = np.array(input_mask[:max_position_embeddings], 'f')
        x3 = np.array(segment_ids[:max_position_embeddings], 'i')
        y = np.array([label_id], 'i')
        data.append((x1, x2, x3, y))

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return data, labels


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
        out_3d = out_2d.reshape(x.shape[:-1] + (out_2d.shape[-1], ))
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


from sklearn.utils import shuffle as skshuffle


def batch_iter(data, batch_size, shuffle=True):
    batch = []
    shuffled_data = np.copy(data)
    if shuffle:
        shuffled_data = skshuffle(shuffled_data)

    for line in shuffled_data:
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
    parser = argparse.ArgumentParser(description='Chainer example: Classifier w/BERT')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=64, type=int, help='learning batchsize size')
    parser.add_argument('--model', default='models', type=str, help='model directory')
    parser.add_argument('--train', default='datasets/rt-polarity/04-train.txt', type=str, help='training file (.txt)')
    parser.add_argument('--eval', default='datasets/rt-polarity/04-test.txt', type=str, help='evaluating file (.txt)')
    parser.add_argument('--init_checkpoint', default='uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz', type=str, help='initial checkpoint (usually from a pre-trained BERT model (.npz)')
    parser.add_argument('--bert_config_file', default='uncased_L-12_H-768_A-12/bert_config.json', type=str, help='json file corresponding to the pre-trained BERT model (.json)')
    parser.add_argument('--vocab_file', default='uncased_L-12_H-768_A-12/vocab.txt', type=str, help='vocabulary file that the BERT model was trained on (.txt)')
    parser.add_argument('--output_prefix', '-o', default='results_bert-2', type=str, help='output prefix')
    parser.add_argument('--noplot', action='store_true', help='disable PlotReport extension')
    # parser.set_defaults(test=True)
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
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

    model_dir = args.model
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    vocab_file = args.vocab_file
    bert_config_file = args.bert_config_file
    init_checkpoint = args.init_checkpoint

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    labels = {}

    train, labels = load_data(args.train, tokenizer, bert_config, labels)
    eval,  labels = load_data(args.eval,  tokenizer, bert_config, labels)
    print('# train: {}, eval: {},'.format(len(train), len(eval)))
    print('# class: {}, labels: {}'.format(len(labels), labels))
    print('# vocab: {}'.format(len(tokenizer.vocab)))
    sys.stdout.flush()

    # Setup model
    bert = BertModel(config=bert_config)
    model = BertClassifier(bert, num_labels=len(labels))
    chainer.serializers.load_npz(init_checkpoint, model, ignore_names=['output/W', 'output/b'])

    index2label = {v: k for k, v in labels.items()}
    sorted_labels = [k for k, _ in sorted(labels.items(), key=lambda x: x[1], reverse=False)]

    # test (early_stopped model by uar)
    chainer.serializers.load_npz(os.path.join(model_dir, 'early_stopped-uar.model'), model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    test_iter = batch_iter(eval, args.batchsize, shuffle=False)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        y_true = []
        y_pred = []

        for x1, x2, x3, t in test_iter:
            x1 = to_device(args.gpu, F.pad_sequence(x1, length=None, padding=0).array).astype('i')
            x2 = to_device(args.gpu, F.pad_sequence(x2, length=None, padding=0).array).astype('f')
            x3 = to_device(args.gpu, F.pad_sequence(x3, length=None, padding=0).array).astype('i')
            t = to_device(args.gpu, F.pad_sequence(t, length=None, padding=0).array).astype('i')[:, 0]
            y = model.predict(x1, x2, x3)
            y_pred += np.argmax(cuda.to_cpu(y.data), axis=1).tolist()
            y_true += cuda.to_cpu(t).tolist()

        print("\n==== Confusion matrix 1 (early_stopped-uar) ====\n")
        cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm):
            print("{}\t{}".format(label, "\t".join(map(str, counts))))

        print("\n==== Confusion matrix 2 (early_stopped-uar) ====\n")
        cm2 = np.apply_along_axis(lambda x: x / sum(x), 1, cm)
        uar = np.nanmean(np.diag(cm2))

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm2):
            print("{}\t{}".format(label, "\t".join(map(lambda x: "%.2f" % x, counts))))

        print("\nUAR = {:.6f}".format(float(uar)))
        sys.stdout.flush()

        # グラフ描画
        if not args.noplot:
            import matplotlib as mpl
            mpl.rcParams['font.family'] = 'YuGothic'
            plt.figure()
            plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(cm2.shape[0]):
                for j in range(cm2.shape[1]):
                    plt.text(j, i, "{:.2f}".format(cm2[i, j]), horizontalalignment="center", color="white" if cm2[i, j] > cm2.max() / 2 else "black")
            plt.title('Confusion matrix: UAR = {:.6f}'.format(uar))
            plt.colorbar()
            tick_marks = np.arange(len(sorted_labels))
            plt.xticks(tick_marks, sorted_labels, rotation=45)
            plt.yticks(tick_marks, sorted_labels)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig('{}-cm-early_stopped-uar.png'.format(args.output_prefix))
            # plt.savefig('{}-train_cm.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        print("\n==== Classification report (early_stopped-uar) ====\n")
        print(classification_report(
            [sorted_labels[x] for x in y_true],
            [sorted_labels[x] for x in y_pred]
        ))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
