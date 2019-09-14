#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: Recurrent Convolutional Neural Networks for Sentence Classification with BERT pre-trained embedding.

http://ai2-s2-pdfs.s3.amazonaws.com/eba3/6ac75bf22edf9a1bfd33244d459c75b98305.pdf

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
from sklearn.utils import shuffle as skshuffle


class RNNClassifier(chainer.Chain):

    """A classifier using a LSTM-RNN Encoder with Word Embedding.

    This chain encodes a sentence and classifies it into classes.
    This model encodes a sentence sequentially using LSTM.

    Args:
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.
        n_class (int): The number of classes to be predicted.

    """

    def __init__(self, n_layers, n_dims, n_units, n_class, dropout=0.1, pre_embed=None):
        super(RNNClassifier, self).__init__()
        with self.init_scope():
            self.bert = pre_embed.bert
            self.encoder = L.NStepBiLSTM(n_layers, n_dims, n_units, dropout)
            self.output = L.Linear(n_units * 2 + n_dims, n_class)

        self.dropout = dropout

    def __call__(self, xs1, xs2, xs3, ts):
        outputs = self.predict(xs1, xs2, xs3)
        loss = F.softmax_cross_entropy(outputs, ts)
        accuracy = F.accuracy(outputs, ts)
        return loss, accuracy

    def predict(self, xs1, xs2, xs3, softmax=False, argmax=False):
        embedding = self.bert.get_embedding_output(xs1, xs2, xs3)
        x_len = [len(x) for x in xs1]
        x_section = np.cumsum(x_len[:-1])
        ex_block = F.split_axis(F.concat(embedding, axis=0), x_section, 0)

        last_h, last_c, ys = self.encoder(None, None, ex_block)
        h_list = []
        for y, ex in zip(ys, ex_block):
            h_list.append(F.max(F.concat((y, ex), axis=1), axis=0)[None, :])
        h = F.concat(h_list, axis=0)

        encodings = F.dropout(h, ratio=self.dropout)
        outputs = self.output(encodings)
        if softmax:
            return F.softmax(outputs).data
        elif argmax:
            return self.xp.argmax(outputs.data, axis=1)
        else:
            return outputs


def load_data(path, labels, tokenizer):
    features = []

    print('loading...: %s' % path)
    for i, line in enumerate(open(path)):
        # if i > 100:
        #     break

        line = line.strip()
        line = line.replace(u'. . .', u'…')
        if line == '':
            continue

        label, text = line.split('\t')

        if label not in labels:
            labels += [label]

        tokens_a = tokenizer.tokenize(text)
        tokens = ["[CLS]"]
        segment_ids = [0]

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        label_id = labels.index(label)

        feature = (np.array(input_ids, 'i'),
                   np.array(input_mask, 'f'),
                   np.array(segment_ids, 'i'),
                   np.array([label_id], 'i'))
        features.append(feature)

    return features


def load_glove_model(path, vocab, width=100):
    w_shape = (len(vocab), width)
    w = np.random.uniform(-np.sqrt(6. / sum(w_shape)), np.sqrt(6. / sum(w_shape)), w_shape).astype(np.float32)

    print('loading...: %s' % path)
    for i, line in enumerate(open(path, 'r')):
        line = line.strip()
        if line == '':
            continue
        cols = line.split(' ')

        token = cols[0]
        if token not in vocab:
            continue

        w[int(vocab.get(token))] = cols[1:]

    l2 = np.linalg.norm(w, axis=1)
    return w / l2.repeat(w_shape[1]).reshape(w_shape)


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


class BertEmbedding(chainer.Chain):

    def __init__(self, bert):
        super(BertEmbedding, self).__init__()
        with self.init_scope():
            self.bert = bert

    def __call__(self, x1, x2, x3, ts):
        output_layer = self.bert.get_embedding_output(x1, x2, x3)
        return output_layer


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: RCNN Classifier w/BERT')
    parser.add_argument('--train',            default='',  type=str, help='training file (.txt)')
    parser.add_argument('--test',             default='',  type=str, help='evaluating file (.txt)')
    parser.add_argument('--init_checkpoint',  default='',  type=str, help='initial checkpoint (usually from a pre-trained BERT model (.npz)')
    parser.add_argument('--bert_config_file', default='',  type=str, help='json file corresponding to the pre-trained BERT model (.json)')
    parser.add_argument('--vocab_file',       default='',  type=str, help='vocabulary file that the BERT model was trained on (.txt)')
    parser.add_argument('--gpu',       '-g',  default=-1,  type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch',     '-e',  default=50,  type=int, help='number of epochs to learn')
    parser.add_argument('--unit',      '-u',  default=300, type=int, help='number of output channels')
    parser.add_argument('--batchsize', '-b',  default=64, type=int, help='learning batchsize size')
    parser.add_argument('--out',       '-o', default='model-rcnn-bert',  type=str, help='output directory')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.gpu >= 0 and cuda.check_cuda_available():
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.cupy.random.seed(seed)

    vocab_file = args.vocab_file
    bert_config_file = args.bert_config_file
    init_checkpoint = args.init_checkpoint

    from tokenization import FullTokenizer
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    vocab = tokenizer.vocab
    labels = []

    train = load_data(args.train, labels, tokenizer)
    test  = load_data(args.test,  labels, tokenizer)
    # assert labels == ["0", "1"]

    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    print('# class: {}'.format(len(labels)))
    sys.stdout.flush()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    # 学習の繰り返し回数
    n_epoch = args.epoch

    # 確率的勾配降下法で学習させる際の1回分のバッチサイズ
    batchsize = args.batchsize

    n_units = args.unit
    n_vocab = len(vocab)
    n_class = len(labels)

    # Setup bert
    pre_embed = None
    import modeling
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    bert = BertEmbedding(modeling.BertModel(config=bert_config))
    with np.load(init_checkpoint) as f:
        d = chainer.serializers.NpzDeserializer(f, path='', strict=True)
        d.load(bert)

    # Setup model
    model = RNNClassifier(n_layers=1, n_dims=bert_config.hidden_size, n_units=n_units, dropout=0.4, n_class=n_class, pre_embed=bert)
    if args.gpu >= 0:
        model.to_gpu()

    # 重み減衰
    decay = 0.0001

    # 勾配上限
    grad_clip = 3

    # Setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

    # プロット用に実行結果を保存する
    train_loss = []
    train_accuracy1 = []
    train_accuracy2 = []
    test_loss = []
    test_accuracy1 = []
    test_accuracy2 = []
    best_accuracy = .0
    min_epoch = 0

    start_at = time.time()
    cur_at = start_at
    sys.stdout.flush()

    # Learning loop
    for epoch in range(1, args.epoch + 1):

        # logger.info('epoch {:} / {:}'.format(epoch, n_epoch))
        # handler1.flush()

        # training
        train_iter = batch_iter(train, args.batchsize, shuffle=True)
        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        for x1, x2, x3, t in train_iter:
            x1 = to_device(args.gpu, F.pad_sequence(x1, length=None, padding=0).array).astype('i')
            x2 = to_device(args.gpu, F.pad_sequence(x2, length=None, padding=0).array).astype('f')
            x3 = to_device(args.gpu, F.pad_sequence(x3, length=None, padding=0).array).astype('i')
            t  = to_device(args.gpu, F.pad_sequence(t , length=None, padding=0).array).astype('i')[:, 0]

            # 勾配を初期化
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            loss, accuracy = model(x1, x2, x3, t)
            sum_train_loss += float(loss.data) * len(t)
            sum_train_accuracy1 += float(accuracy.data) * len(t)
            sum_train_accuracy2 += .0
            K += len(t)

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
        test_iter = batch_iter(test, args.batchsize, shuffle=False)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        # evaluation
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for x1, x2, x3, t in test_iter:
                x1 = to_device(args.gpu, F.pad_sequence(x1, length=None, padding=0).array).astype('i')
                x2 = to_device(args.gpu, F.pad_sequence(x2, length=None, padding=0).array).astype('f')
                x3 = to_device(args.gpu, F.pad_sequence(x3, length=None, padding=0).array).astype('i')
                t =  to_device(args.gpu, F.pad_sequence(t,  length=None, padding=0).array).astype('i')[:, 0]

                # 順伝播させて誤差と精度を算出
                loss, accuracy = model(x1, x2, x3, t)
                sum_test_loss += float(loss.data) * len(t)
                sum_test_accuracy1 += float(accuracy.data) * len(t)
                sum_test_accuracy2 += .0
                K += len(t)

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
                    'lr={:.6f}'
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
            optimizer.alpha)
        )
        sys.stdout.flush()

        # model と optimizer を保存する
        if mean_test_accuracy1 > best_accuracy:
            best_accuracy = mean_test_accuracy1
            min_epoch = epoch
            print('saving early stopped-model at epoch {}'.format(min_epoch))
            if args.gpu >= 0: model.to_cpu()
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped.model'), model)
            chainer.serializers.save_npz(os.path.join(args.out, 'early_stopped.state'), optimizer)
            if args.gpu >= 0: model.to_gpu()
            sys.stdout.flush()

        # 精度と誤差をグラフ描画
        if True:
            ylim1 = [min(train_loss + test_loss), max(train_loss + test_loss)]
            # ylim2 = [min(train_accuracy1 + test_accuracy2), max(train_accuracy1 + test_accuracy2)]
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
            plt.legend(['train acc1', 'train acc2'], loc="upper right")
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
            # plt.plot(range(1, len(test_accuracy2) + 1), test_accuracy2, color='C2', marker='x')
            plt.yticks(np.arange(ylim2[0], ylim2[1], .1))
            plt.grid(True)
            plt.ylabel('accuracy')
            plt.legend(['test acc1', 'test acc2'], loc="upper right")
            plt.title('Loss and accuracy of test.')

            # plt.savefig('{}.png'.format(args.out))
            plt.savefig('{}.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        cur_at = now

    # model と optimizer を保存する
    print('saving final-model at epoch {}'.format(epoch))
    if args.gpu >= 0: model.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.out, 'final.model'), model)
    chainer.serializers.save_npz(os.path.join(args.out, 'final.state'), optimizer)
    if args.gpu >= 0: model.to_gpu()
    sys.stdout.flush()

print('time spent:', time.time() - start_time)
