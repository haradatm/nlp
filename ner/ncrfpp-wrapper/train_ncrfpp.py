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

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import collections

from NCRFpp.model.seqlabel import SeqLabel
from NCRFpp.model.sentclassifier import SentClassifier
from NCRFpp.utils.data import Data
from seqeval.metrics import f1_score, accuracy_score, classification_report
import gc


def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    if sentence_classification:
        pred_tag = pred_variable.cpu().data.numpy().tolist()
        gold_tag = gold_variable.cpu().data.numpy().tolist()
        pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
        gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
    else:
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert (len(pred) == len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
    return pred_label, gold_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='NCRF++: Neural Named entity recognition')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--config',  help='configuration file', default='')
    parser.add_argument('--train', default='datasets/bioes/eng.train.bioes', type=str, help='dataset to train (.txt)')
    parser.add_argument('--dev', default='datasets/bioes/eng.testb.bioes', type=str, help='use tiny datasets to evaluate (.txt)')
    parser.add_argument('--test', default='datasets/bioes/eng.testb.bioes', type=str, help='use tiny datasets to test (.txt)')
    parser.add_argument('--glove', default='datasets/glove.6B.100d.txt', type=str, help='initialize word embedding layer with glove (.txt)')
    parser.add_argument('--batchsize', '-b', type=int, default=10, help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=30, type=int, help='number of epochs to learn')
    parser.add_argument('--out', default='results_ncrfpp-2', help='Directory to output the result')
    parser.add_argument('--noplot', action='store_true', help='disable PlotReport extension')
    # parser.set_defaults(test=True)
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_gpu = torch.cuda.is_available() if args.gpu >= 0 else False
    if use_gpu:
        torch.cuda.manual_seed(seed)

    model_dir = args.out
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    data = Data()

    if args.config:
        data.read_config(args.config)
        data.train_dir = args.train
        data.dev_dir = args.dev
        data.test_dir = args.test
        # data.model_dir = args.out
        data.word_emb_dir = args.glove
        data.status = 'train'
        # data.HP_iteration = args.epoch
        data.HP_gpu = use_gpu
    else:
        # I/O
        data.seg = True
        data.train_dir = args.train
        data.dev_dir = args.dev
        data.test_dir = args.test
        # data.model_dir = args.out
        data.word_emb_dir = args.glove
        data.word_emb_dim = 50
        data.char_emb_dim = 30

        data.number_normalized = True
        data.norm_word_emb = False
        data.norm_char_emb = False

        # Networks
        data.word_seq_feature = 'LSTM'
        data.use_char = True
        data.char_seq_feature = 'CNN'
        data.use_crf = True
        data.nbest = 1

        # Training
        data.ave_batch_loss = False
        # data.optimizer = 'SGD'
        data.status = 'train'

        # Hyperparameters
        data.HP_cnn_layer = 4
        # data.HP_iteration = args.epoch
        data.HP_batch_size = 10
        data.HP_char_hidden_dim = 50
        data.HP_hidden_dim = 200
        data.HP_dropout = 0.5
        data.HP_lstm_layer = 1
        data.HP_bilstm = True
        data.HP_gpu = use_gpu
        data.HP_learning_rate = 0.015
        data.HP_lr_decay = 0.05
        data.momentum = 0
        data.l2 = 1e-8

    print("Save dset directory: ", data.dset_dir)
    print("Seed num: ", seed)
    print("MODEL: ", data.status)

    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()
    data.generate_instance('train')
    data.generate_instance('dev')
    data.generate_instance('test')
    data.build_pretrain_emb()
    data.show_data_summary()
    sys.stdout.flush()

    data.save(os.path.join(args.out, "data.bin"))

    model = SeqLabel(data)
    optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
    # optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    # optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    # optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    # optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)

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
    for epoch in range(1, args.epoch + 1):

        if data.optimizer == "SGD":
            lr = data.HP_lr / (1 + data.HP_lr_decay * epoch)
            print(" Learning rate is set as:", lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # training
        random.shuffle(data.train_Ids)
        print("Shuffle: first input word list: ", data.train_Ids[0][0])

        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        model.train()
        model.zero_grad()
        batch_size = args.batchsize
        train_num = len(data.train_Ids)
        total_batch = train_num // args.batchsize + 1

        y_true, y_pred = [], []
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue

            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_sequence_labeling_with_label(instance, data.HP_gpu, True)
            loss, tag_seq = model.calculate_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)

            y_pred += pred_label
            y_true += gold_label

            sum_train_loss += loss.item()
            K += len(instance)

            loss.backward()
            optimizer.step()
            model.zero_grad()

        if sum_train_loss > 1e8 or str(sum_train_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)

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
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        model.eval()
        batch_size = args.batchsize
        train_num = len(data.dev_Ids)
        total_batch = train_num // args.batchsize + 1

        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance = data.dev_Ids[start:end]
                if not instance:
                    continue

                batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_sequence_labeling_with_label(instance, data.HP_gpu, True)
                loss, tag_seq = model.calculate_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
                pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)

                y_pred += pred_label
                y_true += gold_label

                sum_test_loss += loss.item()
                K += len(instance)

        sum_test_accuracy1 += f1_score(y_true, y_pred)
        sum_test_accuracy2 += accuracy_score(y_true, y_pred)

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
            lr,
        )
        )
        sys.stdout.flush()

        gc.collect()

        # model と optimizer を保存する
        if mean_test_loss < min_loss:
            min_loss = mean_test_loss
            print('saving early-stopped model (loss) at epoch {}'.format(epoch))
            torch.save(model.state_dict(), os.path.join(args.out, 'early_stopped-loss.model'))
        if mean_test_accuracy1 > best_accuracy:
            best_accuracy = mean_test_accuracy1
            print('saving early-stopped model (f1) at epoch {}'.format(epoch))
            torch.save(model.state_dict(), os.path.join(args.out, 'early_stopped-f1.model'))
            print("\n==== Classification report (early-stopped model) ====\n")
            print(classification_report(y_true, y_pred))
            sys.stdout.flush()
        # print('saving final model at epoch {}'.format(epoch))
        torch.save(model.state_dict(), os.path.join(args.out, 'final.model'))
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
    print(classification_report(y_true, y_pred))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
