#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, time, logging, os, json, re, random
import numpy as np
np.set_printoptions(precision=20)
np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
# handler = logging.FileHandler(filename="log.txt")
handler.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'))
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


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
import collections


def load_conll2003(path, id_list, tokenizer, bert_config, type_id=0):
    X, y = [], []

    data, tokens, labels = [], [], []
    for i, line in enumerate(open(path)):
        line = line.strip()
        line = line.replace(u'. . .', u'…')

        if line.startswith("-DOCSTART-"):
            continue

        if line == '':
            if len(tokens) > 0:
                data.append((tokens, labels))
                tokens, labels = [], []
            continue

        cols = line.split()
        word, label = cols[0], cols[3]

        if label not in id_list:
            id_list += [label]

        tokens.append(word)
        labels.append(id_list.index(label))

    if len(tokens) > 0:
        data.append((tokens, labels))

    for i, (tokens_a, labels) in enumerate(data):
        # if i >= 100:
        #     break

        tokens = ["[CLS]"] if type_id == 0 else []
        segment_ids = [type_id] if type_id == 0 else []
        label_ids = [id_list.index("[CLS]")]

        for j, (token, label_id) in enumerate(zip(tokens_a, labels)):
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) == 1:
                tokens.append(sub_tokens[0])
                segment_ids.append(0)
                label_ids.append(label_id)
            else:
                tags = id_list[label_id].split('-')
                for k, sub_token in enumerate(sub_tokens):
                    tokens.append(sub_token)
                    segment_ids.append(0)

                    if k == 0:                          # start
                        if tags[0] in ['B', 'I', 'O']:
                            label_ids.append(label_id)
                        elif tags[0] in ['E']:
                            label_ids.append(id_list.index('I-' + tags[1]))
                        elif tags[0] in ['S']:
                            label_ids.append(id_list.index('B-' + tags[1]))
                    elif k == len(sub_tokens) - 1:      # end
                        if tags[0] in ['E', 'I', 'O']:
                            label_ids.append(label_id)
                        elif tags[0] in ['B']:
                            label_ids.append(id_list.index('I-' + tags[1]))
                        elif tags[0] in ['S']:
                            label_ids.append(id_list.index('E-' + tags[1]))
                    else:                               # middle
                        if tags[0] in ['O']:
                            label_ids.append(label_id)
                        else:
                            label_ids.append(id_list.index('I-' + tags[1]))

        tokens.append("[SEP]")
        segment_ids.append(type_id)
        label_ids.append(id_list.index("[SEP]"))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        max_position_embeddings = bert_config.max_position_embeddings
        x1 = np.array(input_ids[:max_position_embeddings], 'i')
        x2 = np.array(input_mask[:max_position_embeddings], 'f')
        x3 = np.array(segment_ids[:max_position_embeddings], 'i')
        y = np.array(label_ids[:max_position_embeddings], 'i')
        X.append((x1, x2, x3, y))

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return X, id_list


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
class BertNER(chainer.Chain):
    def __init__(self, bert, n_label, use_crf=False):
        super(BertNER, self).__init__()
        with self.init_scope():
            self.bert = bert
            self.output = Linear3D(None, n_label, initialW=chainer.initializers.Normal(0.02))
            self.crf = L.CRF1d(n_label)
            self.n_label = n_label
            self.use_crf = use_crf

    def forward(self, input_ids, input_mask, token_type_ids):
        final_hidden = self.bert.get_sequence_output(input_ids, input_mask, token_type_ids)
        batch_size, seq_length, hidden_size = final_hidden.shape
        final_hidden_matrix = F.reshape(final_hidden, [batch_size * seq_length, hidden_size])
        output_layer = self.output(final_hidden_matrix)
        output_layer = F.reshape(output_layer, [batch_size, seq_length, self.n_label])
        return output_layer

    def __call__(self, input_ids, input_mask, token_type_ids, label_ids):
        logits = self.forward(input_ids, input_mask, token_type_ids)
        if self.use_crf:
            ys, ts = [], []
            for y, t, m in zip(logits, label_ids, input_mask):
                mask = m.astype('bool')
                ys.append(F.get_item(y, mask))
                ts.append(F.get_item(t, mask))
            sorted_list = sorted(zip(ys, ts), key=lambda x: len(x[0]), reverse=True)
            sorted_ys, sorted_ts = tuple(list(x) for x in zip(*sorted_list))
            loss = self.crf(sorted_ys, sorted_ts)
        else:
            batch_size, seq_length, class_size = logits.shape
            y = F.reshape(logits, [batch_size * seq_length, class_size])
            t = F.reshape(label_ids, [batch_size * seq_length])
            loss = F.softmax_cross_entropy(y, t, ignore_label=0)
        return loss, logits


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


from seqeval.metrics import f1_score, accuracy_score, classification_report


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: BERT NER')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=32, type=int, help='learning batchsize size')
    parser.add_argument('--learnrate', '-l', type=float, default=5e-5, help='value of learning rate')
    parser.add_argument('--weightdecay', default=0.01, type=float, help='value of weight decay rate')
    parser.add_argument('--epoch', '-e', default=50, type=int, help='number of epochs to learn')
    parser.add_argument('--train', default='datasets/eng.train', type=str, help='dataset to train (.txt)')
    parser.add_argument('--test', default='datasets/eng.testb', type=str, help='dataset to validate (.txt)')
    parser.add_argument('--init_checkpoint', default='uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz', type=str, help='initial checkpoint (usually from a pre-trained BERT model (.npz)')
    parser.add_argument('--bert_config_file', default='uncased_L-12_H-768_A-12/bert_config.json', type=str, help='json file corresponding to the pre-trained BERT model (.json)')
    parser.add_argument('--vocab_file', default='uncased_L-12_H-768_A-12/vocab.txt', type=str, help='vocabulary file that the BERT model was trained on (.txt)')
    parser.add_argument('--out', '-o', default='results_ner-5', type=str, help='output directory')
    parser.add_argument('--resume', default='', type=str, help='path to resume models')
    parser.add_argument('--start_epoch', default=1, type=int, help='epoch number at start')
    parser.add_argument('--crf', action='store_true', help='use crf loss')
    parser.add_argument('--noplot', action='store_true', help='disable PlotReport extension')
    # parser.set_defaults(crf=True)
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

    id_list = ['[PAD]', '[CLS]', '[SEP]']
    train, id_list = load_conll2003(args.train, id_list, tokenizer, bert_config, type_id=0)
    test, id_list = load_conll2003(args.test, id_list, tokenizer, bert_config, type_id=0)
    n_label = len(id_list)

    print('# train: {}, test: {}, class: {}'.format(len(train), len(test), n_label))
    print('# vocab: {}'.format(len(tokenizer.vocab)))
    sys.stdout.flush()

    # Setup model
    bert = BertModel(config=bert_config)
    model = BertNER(bert, n_label=n_label, use_crf=args.crf)
    chainer.serializers.load_npz(init_checkpoint, model, ignore_names=['output/W', 'output/b', 'crf/cost'])

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
        train_iter = batch_iter(train, args.batchsize)
        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        y_true, y_pred = [], []
        for x1, x2, x3, t in train_iter:
            # if K > 1: break
            N = len(t)
            x1 = to_device(args.gpu, F.pad_sequence(x1, length=None, padding=0).array).astype('i')
            x2 = to_device(args.gpu, F.pad_sequence(x2, length=None, padding=0).array).astype('f')
            x3 = to_device(args.gpu, F.pad_sequence(x3, length=None, padding=0).array).astype('i')
            t = to_device(args.gpu, F.pad_sequence(t, length=None, padding=0).array).astype('i')

            # 勾配を初期化
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            loss, y = model(x1, x2, x3, t)
            sum_train_loss += float(loss.data) * N

            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

            masks = cuda.to_cpu(x2).astype('bool')
            y_pred += [[id_list[label_id] for label_id in np.argmax(label_ids[mask], axis=1)][1:-1] for label_ids, mask in zip(cuda.to_cpu(y.data), masks)]
            y_true += [[id_list[label_id] for label_id in label_ids[mask][1:-1]] for label_ids, mask in zip(cuda.to_cpu(t), masks)]
            K += N

        sum_train_accuracy1 += f1_score(y_true, y_pred)
        sum_train_accuracy2 += accuracy_score(y_true, y_pred)

        # 訓練データの誤差と,正解精度を表示
        mean_train_loss = sum_train_loss / K
        mean_train_accuracy1 = sum_train_accuracy1
        mean_train_accuracy2 = sum_train_accuracy2
        train_loss.append(mean_train_loss)
        train_accuracy1.append(mean_train_accuracy1)
        train_accuracy2.append(mean_train_accuracy2)
        now = time.time()
        train_throughput = now - cur_at
        cur_at = now

        # evaluation
        test_iter = batch_iter(test, args.batchsize)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        test_y_true, test_y_pred = [], []
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for x1, x2, x3, t in test_iter:
                # if K > 1: break
                N = len(t)
                x1 = to_device(args.gpu, F.pad_sequence(x1, length=None, padding=0).array).astype('i')
                x2 = to_device(args.gpu, F.pad_sequence(x2, length=None, padding=0).array).astype('f')
                x3 = to_device(args.gpu, F.pad_sequence(x3, length=None, padding=0).array).astype('i')
                t = to_device(args.gpu, F.pad_sequence(t, length=None, padding=0).array).astype('i')

                # 順伝播させて誤差と精度を算出
                loss, y = model(x1, x2, x3, t)
                sum_test_loss += float(loss.data) * N

                masks = cuda.to_cpu(x2).astype('bool')
                test_y_pred += [[id_list[label_id] for label_id in np.argmax(label_ids[mask], axis=1)][1:-1] for label_ids, mask in zip(cuda.to_cpu(y.data), masks)]
                test_y_true += [[id_list[label_id] for label_id in label_ids[mask][1:-1]] for label_ids, mask in zip(cuda.to_cpu(t), masks)]
                K += N

        sum_test_accuracy1 += f1_score(test_y_true, test_y_pred)
        sum_test_accuracy2 += accuracy_score(test_y_true, test_y_pred)

        # テストデータでの誤差と正解精度を表示
        mean_test_loss = sum_test_loss / K
        mean_test_accuracy1 = sum_test_accuracy1
        mean_test_accuracy2 = sum_test_accuracy2
        test_loss.append(mean_test_loss)
        test_accuracy1.append(mean_test_accuracy1)
        test_accuracy2.append(mean_test_accuracy2)
        now = time.time()
        test_throughput = now - cur_at
        cur_at = now

        logger.info(''
                    '[{:>3d}] '
                    'T/loss={:.6f} '
                    'T/f1={:.6f} '
                    'T/acc={:.6f} '
                    'T/sec= {:.6f} '
                    'D/loss={:.6f} '
                    'D/f1={:.6f} '
                    'D/acc={:.6f} '
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
        if mean_test_accuracy1 > best_accuracy:
            best_accuracy = mean_test_accuracy1
            print('saving early-stopped model (f1) at epoch {}'.format(epoch))
            chainer.serializers.save_npz(os.path.join(model_dir, 'early_stopped-f1.model'), model)
            print("\n==== Classification report (early-stopped model) ====\n")
            print(classification_report(test_y_true, test_y_pred))
            sys.stdout.flush()
        # print('saving final model at epoch {}'.format(epoch))
        chainer.serializers.save_npz(os.path.join(model_dir, 'final.model'), model)
        chainer.serializers.save_npz(os.path.join(model_dir, 'final.state'), optimizer)
        if args.gpu >= 0: model.to_gpu()
        sys.stdout.flush()

        # 精度と誤差をグラフ描画
        if not args.noplot:
            ylim1 = [min(train_loss + test_loss), max(train_loss + test_loss)]
            # ylim2 = [min(train_accuracy1 + test_accuracy1 + train_accuracy2 + test_accuracy2), max(train_accuracy1 + test_accuracy1 + train_accuracy2 + test_accuracy2)]
            ylim2 = [0., 1.]

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
            plt.plot(range(1, len(train_accuracy2) + 1), train_accuracy2, color='C2', marker='x')
            plt.yticks(np.arange(ylim2[0], ylim2[1], .1))
            plt.grid(True)
            # plt.ylabel('accuracy')
            plt.legend(['train f1', 'train acc'], loc="upper right")
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
            plt.legend(['test f1', 'test acc'], loc="upper right")
            plt.title('Loss and accuracy of test.')

            plt.savefig('{}.png'.format(args.out))
            # plt.savefig('{}-train.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        cur_at = now

    print("\n==== Classification report (final model) ====\n")
    print(classification_report(test_y_true, test_y_pred))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
