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
    parser.add_argument('--learnrate', '-l', type=float, default=5e-5, help='value of learning rate')
    parser.add_argument('--weightdecay', default=0.01, type=float, help='value of weight decay rate')
    parser.add_argument('--epoch', '-e', default=50, type=int, help='number of epochs to learn')
    parser.add_argument('--train', default='datasets/rt-polarity/04-train.txt', type=str, help='training file (.txt)')
    parser.add_argument('--eval', default='datasets/rt-polarity/04-test.txt', type=str, help='evaluating file (.txt)')
    parser.add_argument('--init_checkpoint', default='uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz', type=str, help='initial checkpoint (usually from a pre-trained BERT model (.npz)')
    parser.add_argument('--bert_config_file', default='uncased_L-12_H-768_A-12/bert_config.json', type=str, help='json file corresponding to the pre-trained BERT model (.json)')
    parser.add_argument('--vocab_file', default='uncased_L-12_H-768_A-12/vocab.txt', type=str, help='vocabulary file that the BERT model was trained on (.txt)')
    parser.add_argument('--out', '-o', default='results_bert-2', type=str, help='output directory')
    parser.add_argument('--resume', default='', type=str, help='path to resume models')
    parser.add_argument('--start_epoch', default=1, type=int, help='epoch number at start')
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

    model_dir = args.out
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

    if args.gpu >= 0:
        model.to_gpu()

    # 学習率
    lr = args.learnrate
    rate = (lr - 0.) / args.epoch

    # 重み減衰
    lr_decay = 0.0001

    # 勾配上限
    grad_clip = 1.

    # Setup optimizer
    # ignore alpha. instead, use eta as actual lr
    optimizer = WeightDecayForMatrixAdam(alpha=1., eps=1e-6, weight_decay_rate=0.01)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(1.))
    optimizer.eta = lr

    if args.resume:
        print('resuming model and state at: {}'.format(args.start_epoch - 1))
        if args.gpu >= 0: model.to_cpu()
        chainer.serializers.load_npz(os.path.join(args.resume, 'final.model'), model)
        chainer.serializers.load_npz(os.path.join(args.resume, 'final.state'), optimizer)
        if args.gpu >= 0: model.to_gpu()
        sys.stdout.flush()
        optimizer.eta = lr - (rate * (args.start_epoch - 1))

    # プロット用に実行結果を保存する
    train_loss = []
    train_accuracy1 = []
    train_accuracy2 = []
    test_loss = []
    test_accuracy1 = []
    test_accuracy2 = []

    min_loss = float('inf')
    best_accuracy = .0

    start_at = time.time()
    cur_at = start_at

    # Learning loop
    for epoch in range(args.start_epoch, args.epoch + 1):

        # training
        train_iter = batch_iter(train, args.batchsize, shuffle=True)
        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        for x1, x2, x3, t in train_iter:
            # if K > 1: break
            N = len(t)
            x1 = to_device(args.gpu, F.pad_sequence(x1, length=None, padding=0).array).astype('i')
            x2 = to_device(args.gpu, F.pad_sequence(x2, length=None, padding=0).array).astype('f')
            x3 = to_device(args.gpu, F.pad_sequence(x3, length=None, padding=0).array).astype('i')
            t =  to_device(args.gpu, F.pad_sequence(t,  length=None, padding=0).array).astype('i')[:, 0]

            # 勾配を初期化
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            loss, accuracy = model(x1, x2, x3, t)
            sum_train_loss += float(loss.data) * N
            sum_train_accuracy1 += float(accuracy.data) * N
            sum_train_accuracy2 += .0
            K += N

            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

        # 訓練データの誤差と,正解精度を表示
        mean_train_loss = sum_train_loss / K
        mean_train_accuracy1 = sum_train_accuracy1 / K
        mean_train_accuracy2 = sum_train_accuracy2 / K
        train_loss.append(mean_train_loss)
        train_accuracy1.append(mean_train_accuracy1)
        train_accuracy2.append(mean_train_accuracy2)
        now = time.time()
        train_throughput = now - cur_at
        cur_at = now

        # evaluation
        test_iter = batch_iter(eval, args.batchsize, shuffle=False)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        # evaluation
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            y_true = []
            y_pred = []

            for x1, x2, x3, t in test_iter:
                # if K > 1: break
                N = len(t)
                x1 = to_device(args.gpu, F.pad_sequence(x1, length=None, padding=0).array).astype('i')
                x2 = to_device(args.gpu, F.pad_sequence(x2, length=None, padding=0).array).astype('f')
                x3 = to_device(args.gpu, F.pad_sequence(x3, length=None, padding=0).array).astype('i')
                t =  to_device(args.gpu, F.pad_sequence(t,  length=None, padding=0).array).astype('i')[:, 0]

                # 順伝播させて誤差と精度を算出
                loss, accuracy = model(x1, x2, x3, t)
                sum_test_loss += float(loss.data) * N
                sum_test_accuracy1 += float(accuracy.data) * N
                sum_test_accuracy2 += .0
                K += N

                y = model.predict(x1, x2, x3)
                y_pred += np.argmax(cuda.to_cpu(y.data), axis=1).tolist()
                y_true += cuda.to_cpu(t).tolist()

        cm = confusion_matrix(y_true, y_pred)
        cm2 = np.apply_along_axis(lambda x: x / sum(x), 1, cm)
        uar = np.nanmean(np.diag(cm2))
        sum_test_accuracy2 += uar * K

        # テストデータでの誤差と正解精度を表示
        mean_test_loss = sum_test_loss / K
        mean_test_accuracy1 = sum_test_accuracy1 / K
        mean_test_accuracy2 = sum_test_accuracy2 / K
        test_loss.append(mean_test_loss)
        test_accuracy1.append(mean_test_accuracy1)
        test_accuracy2.append(mean_test_accuracy2)
        now = time.time()
        test_throughput = now - cur_at
        cur_at = now

        logger.info(''
                    '[{:>3d}] '
                    'T/loss={:.6f} '
                    'T/acc1={:.6f} '
                    'T/acc2={:.6f} '
                    'T/sec= {:.6f} '
                    'D/loss={:.6f} '
                    'D/acc1={:.6f} '
                    'D/acc2={:.6f} '
                    'D/sec= {:.6f} '
                    'lr={:.6f} '
                    'eta={:.6f}'
                    ''.format(
            epoch,
            mean_train_loss,
            mean_train_accuracy1,
            mean_train_accuracy2,
            train_throughput,
            mean_test_loss,
            mean_test_accuracy1,
            mean_test_accuracy2,
            test_throughput,
            optimizer.lr,
            optimizer.eta,
        )
        )
        sys.stdout.flush()

        optimizer.eta -= rate
        if optimizer.eta < 0.:
            optimizer.eta = 0.

        # model と optimizer を保存する
        if args.gpu >= 0: model.to_cpu()
        if mean_test_loss < min_loss:
            min_loss = mean_test_loss
            print('saving early-stopped model (loss) at epoch {}'.format(epoch))
            chainer.serializers.save_npz(os.path.join(model_dir, 'early_stopped-loss.model'), model)
        if mean_test_accuracy2 > best_accuracy:
            best_accuracy = mean_test_accuracy2
            print('saving early-stopped model (uar) at epoch {}'.format(epoch))
            chainer.serializers.save_npz(os.path.join(model_dir, 'early_stopped-uar.model'), model)
        # print('saving final model at epoch {}'.format(epoch))
        chainer.serializers.save_npz(os.path.join(model_dir, 'final.model'), model)
        chainer.serializers.save_npz(os.path.join(model_dir, 'final.state'), optimizer)
        if args.gpu >= 0: model.to_gpu()
        sys.stdout.flush()

        # 精度と誤差をグラフ描画
        if True:
            ylim1 = [min(train_loss + test_loss), max(train_loss + test_loss)]
            # ylim2 = [min(train_accuracy1 + test_accuracy1 + train_accuracy2 + test_accuracy2), max(train_accuracy1 + test_accuracy1 + train_accuracy2 + test_accuracy2)]
            ylim2 = [0.5, 1.0]

            # グラフ左
            plt.figure(figsize=(10, 10))

            plt.subplot(1, 2, 1)
            plt.ylim(ylim1)
            plt.plot(range(1, len(train_loss) + 1), train_loss, color='C1', marker='x')
            # plt.grid()
            plt.ylabel('loss')
            plt.legend(['train loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(train_accuracy1) + 1), train_accuracy1, color='C0', marker='x')
            # plt.plot(range(1, len(train_accuracy2) + 1), train_accuracy2, color='C2', marker='x')
            plt.yticks(np.arange(ylim2[0], ylim2[1], .1))
            plt.grid(True)
            # plt.ylabel('accuracy')
            plt.legend(['train acc'], loc="upper right")
            plt.title('Loss and accuracy of train.')

            # グラフ右
            plt.subplot(1, 2, 2)
            plt.ylim(ylim1)
            plt.plot(range(1, len(test_loss) + 1), test_loss, color='C1', marker='x')
            # plt.grid()
            # plt.ylabel('loss')
            plt.legend(['test loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(test_accuracy1) + 1), test_accuracy1, color='C0', marker='x')
            plt.plot(range(1, len(test_accuracy2) + 1), test_accuracy2, color='C2', marker='x')
            plt.yticks(np.arange(ylim2[0], ylim2[1], .1))
            plt.grid(True)
            plt.ylabel('accuracy')
            plt.legend(['test acc', 'test uar'], loc="upper right")
            plt.title('Loss and accuracy of test.')

            plt.savefig('{}.png'.format(args.out))
            # plt.savefig('{}-train.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        cur_at = now

    index2label = {v: k for k, v in labels.items()}
    sorted_labels = [k for k, _ in sorted(labels.items(), key=lambda x: x[1], reverse=False)]

    # test (early_stopped model by loss)
    chainer.serializers.load_npz(os.path.join(model_dir, 'early_stopped-loss.model'), model)
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

        print("\n==== Confusion matrix 1 (early_stopped-loss) ====\n")
        cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm):
            print("{}\t{}".format(label, "\t".join(map(str, counts))))

        print("\n==== Confusion matrix 2 (early_stopped-loss) ====\n")
        cm2 = np.apply_along_axis(lambda x: x / sum(x), 1, cm)
        uar = np.nanmean(np.diag(cm2))

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm2):
            print("{}\t{}".format(label, "\t".join(map(lambda x: "%.2f" % x, counts))))

        print("\nUAR = {:.6f}".format(float(uar)))
        sys.stdout.flush()

        # グラフ描画
        if not args.noplot:
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
            plt.savefig('{}-cm-early_stopped-loss.png'.format(args.out))
            # plt.savefig('{}-train_cm.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        print("\n==== Classification report (early_stopped-loss) ====\n")
        print(classification_report(
            [sorted_labels[x] for x in y_true],
            [sorted_labels[x] for x in y_pred]
        ))
        sys.stdout.flush()

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
            plt.savefig('{}-cm-early_stopped-uar.png'.format(args.out))
            # plt.savefig('{}-train_cm.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        print("\n==== Classification report (early_stopped-uar) ====\n")
        print(classification_report(
            [sorted_labels[x] for x in y_true],
            [sorted_labels[x] for x in y_pred]
        ))
        sys.stdout.flush()

    # test (final model)
    chainer.serializers.load_npz(os.path.join(model_dir, 'final.model'), model)
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

        print("\n==== Confusion matrix 1 (final model) ====\n")
        cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm):
            print("{}\t{}".format(label, "\t".join(map(str, counts))))

        print("\n==== Confusion matrix 2 (final model) ====\n")
        cm2 = np.apply_along_axis(lambda x: x / sum(x), 1, cm)
        uar = np.nanmean(np.diag(cm2))

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm2):
            print("{}\t{}".format(label, "\t".join(map(lambda x: "%.2f" % x, counts))))

        print("\nUAR = {:.6f}".format(float(uar)))
        sys.stdout.flush()

        # グラフ描画
        if not args.noplot:
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
            plt.savefig('{}-cm-final.png'.format(args.out))
            # plt.savefig('{}-cm-final.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        print("\n==== Classification report (final model) ====\n")
        print(classification_report(
            [sorted_labels[x] for x in y_true],
            [sorted_labels[x] for x in y_pred]
        ))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
