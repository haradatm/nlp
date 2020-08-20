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


class MyDatasaet(Dataset):

    def __init__(self, path, labels):
        self.text = []
        self.label = []
        self.labels = labels

        for i, line in enumerate(open(path)):
            # if i > 100:
            #     break

            line = line.strip()
            line = line.replace(u'. . .', u'…')
            if line == '':
                continue

            label, text = line.split('\t')

            if label not in labels:
                labels[label] = len(labels)

            self.text.append(text)
            self.label.append(labels[label])

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.text[idx], self.label[idx]


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('--train', default='datasets/mlit/04-train.txt', type=str, help='training file (.txt)')
    parser.add_argument('--valid', default='datasets/mlit/04-test.txt', type=str, help='evaluating file (.txt)')
    parser.add_argument('--pretrained', default='cl-tohoku/bert-base-japanese-whole-word-masking', type=str, help='pretrained model name or path')
    parser.add_argument('--batchsize', '-b', default=64, type=int, help='learning batchsize size')
    parser.add_argument('--learnrate', '-l', type=float, default=2e-5, help='value of learning rate')
    parser.add_argument('--epoch', '-e', default=4, type=int, help='number of epochs to learn')
    parser.add_argument('--out', '-o', default='results_bert-3', type=str, help='output directory')
    parser.add_argument('--noplot', action='store_true', help='disable PlotReport extension')
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

    model_dir = args.out
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    def generate_batch(batch):
        x_batch, y_batch = [x for x, _ in batch], [y for _, y in batch]
        encoded_data = tokenizer.batch_encode_plus(x_batch, pad_to_max_length=True, add_special_tokens=True)
        return map(torch.tensor, (encoded_data["input_ids"], encoded_data["token_type_ids"], encoded_data["attention_mask"], y_batch))

    labels = {}
    train_ds = MyDatasaet(args.train, labels)
    valid_ds = MyDatasaet(args.valid, train_ds.labels)
    train_dl = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True,  collate_fn=generate_batch)
    valid_dl = DataLoader(valid_ds, batch_size=args.batchsize, shuffle=False, collate_fn=generate_batch)

    print('# train: {}, valid: {}'.format(len(train_ds), len(valid_ds)))
    print('# class: {}, labels: {}'.format(len(labels), labels))
    print('# vocab: {}'.format(len(tokenizer.vocab)))
    sys.stdout.flush()

    # Setup model
    config = AutoConfig.from_pretrained(args.pretrained, num_labels=len(labels), output_attentions=False, output_hidden_states=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained, config=config)
    print(model)
    # tokenizer.save_pretrained('./models')
    # net.save_pretrained('./models')
    # tokenizer = BertJapaneseTokenizer.from_pretrained('./models')    # re-load
    # net = BertForSequenceClassification.from_pretrained('./models')  # re-load

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.to(device)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.learnrate)

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

    start_epoch = 1

    # Learning loop
    for epoch in range(start_epoch, args.epoch + 1):

        # training
        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        # 訓練モード ON
        model.train()
        for input_ids, token_type_ids, attention_mask, ts in train_dl:
            N = len(ts)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            ts = ts.to(device)

            # 勾配を初期化
            optimizer.zero_grad()

            # 順伝播させて誤差と精度を算出
            loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=ts)
            _, preds = torch.max(logits, 1)
            accuracy = (torch.sum(preds == ts)).double() / N
            sum_train_loss += float(loss.item()) * N
            sum_train_accuracy1 += float(accuracy.item()) * N
            sum_train_accuracy2 += .0
            K += N

            # 誤差逆伝播で勾配を計算
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

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
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        # 訓練モード OFF
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():   # 勾配を計算しない
            for input_ids, token_type_ids, attention_mask, ts in valid_dl:
                N = len(ts)
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                ts = ts.to(device)

                loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=ts)
                _, preds = torch.max(logits, 1)
                accuracy = (torch.sum(preds == ts)).double() / N
                sum_test_loss += float(loss.item()) * N
                sum_test_accuracy1 += float(accuracy.item()) * N
                sum_test_accuracy2 += .0
                K += N

                y_pred += preds.cpu().numpy().tolist()
                y_true += ts.cpu().numpy().tolist()

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
                    # 'lr={:.6f} '
                    # 'eta={:.6f}'
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
            # optimizer.lr,
            # optimizer.eta,
        )
        )
        sys.stdout.flush()

        # model と optimizer を保存する
        if torch.cuda.is_available(): model.to('cpu')
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'labels': labels}
        if mean_test_loss < min_loss:
            min_loss = mean_test_loss
            print('saving early-stopped model (loss) at epoch {}'.format(epoch))
            torch.save(state, os.path.join(model_dir, 'early_stopped-loss.pth.tar'))
        if mean_test_accuracy2 > best_accuracy:
            best_accuracy = mean_test_accuracy2
            print('saving early-stopped model (uar) at epoch {}'.format(epoch))
            torch.save(state, os.path.join(model_dir, 'early_stopped-uar.pth.tar'))
        # print('saving final model at epoch {}'.format(epoch))
        torch.save(state, os.path.join(model_dir, 'final.pth.tar'))
        if torch.cuda.is_available(): model.to(device)
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
    state = torch.load(os.path.join(model_dir, 'early_stopped-loss.pth.tar'))
    model.load_state_dict(state['state_dict'])
    if torch.cuda.is_available():
        model.to(device)

    y_true = []
    y_pred = []
    with torch.no_grad():  # 勾配を計算しない
        for input_ids, token_type_ids, attention_mask, ts in valid_dl:
            N = len(ts)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            ts = ts.to(device)

            loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=ts)
            _, preds = torch.max(logits, 1)
            y_pred += preds.cpu().numpy().tolist()
            y_true += ts.cpu().numpy().tolist()

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
                plt.text(j, i, "{:.2f}".format(cm2[i, j]), horizontalalignment="center",
                         color="white" if cm2[i, j] > cm2.max() / 2 else "black")
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
    state = torch.load(os.path.join(model_dir, 'early_stopped-uar.pth.tar'))
    model.load_state_dict(state['state_dict'])
    if torch.cuda.is_available():
        model.to(device)

    y_true = []
    y_pred = []
    with torch.no_grad():  # 勾配を計算しない
        for input_ids, token_type_ids, attention_mask, ts in valid_dl:
            N = len(ts)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            ts = ts.to(device)

            loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=ts)
            _, preds = torch.max(logits, 1)
            y_pred += preds.cpu().numpy().tolist()
            y_true += ts.cpu().numpy().tolist()

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
                plt.text(j, i, "{:.2f}".format(cm2[i, j]), horizontalalignment="center",
                         color="white" if cm2[i, j] > cm2.max() / 2 else "black")
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
    state = torch.load(os.path.join(model_dir, 'final.pth.tar'))
    model.load_state_dict(state['state_dict'])
    if torch.cuda.is_available():
        model.to(device)

    y_true = []
    y_pred = []
    with torch.no_grad():  # 勾配を計算しない
        for input_ids, token_type_ids, attention_mask, ts in valid_dl:
            N = len(ts)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            ts = ts.to(device)

            loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=ts)
            _, preds = torch.max(logits, 1)
            y_pred += preds.cpu().numpy().tolist()
            y_true += ts.cpu().numpy().tolist()

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
    logger.info('time spent: %06f' % (time.time() - start_time))