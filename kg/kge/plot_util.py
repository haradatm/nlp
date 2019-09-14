#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, time, logging, os, json
import numpy as np

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


import matplotlib.pyplot as plt


# プロット用に実行結果を保存する
train_loss = []
train_accuracy1 = []
train_accuracy2 = []
train_accuracy3 = []
valid_loss = []
valid_accuracy1 = []


def plot(tra_loss, tra_acc1, tra_acc2, tra_acc3, val_loss, val_acc1, filename):

    global train_loss, train_accuracy1, train_accuracy2, train_accuracy2, valid_loss, valid_accuracy1

    train_loss.append(tra_loss)
    train_accuracy1.append(tra_acc1)
    train_accuracy2.append(tra_acc2)
    train_accuracy3.append(tra_acc3)
    valid_loss.append(val_loss)
    valid_accuracy1.append(val_acc1)

    ylim1 = [min(train_loss + valid_loss), max(train_loss + valid_loss)]
    ylim2 = [0, 1]

    # グラフ左
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.ylim(ylim1)

    plt.plot(range(1, len(train_loss) + 1), train_loss, color='orange', marker='')
    plt.yticks(np.arange(ylim1[0], ylim1[1], .1))
    plt.grid(True)
    plt.ylabel('loss')
    plt.legend(['train loss'], loc="lower left")

    plt.twinx()

    plt.ylim(ylim2)
    plt.plot(range(1, len(train_accuracy1) + 1), train_accuracy1, color='red',   marker='')
    plt.plot(range(1, len(train_accuracy2) + 1), train_accuracy2, color='blue',  marker='x', linestyle='dashed')
    plt.plot(range(1, len(train_accuracy3) + 1), train_accuracy3, color='green', marker='x', linestyle='dashed')
    plt.yticks(np.arange(ylim2[0], ylim2[1], .1))
    plt.grid(True)
    # plt.ylabel('accuracy')
    plt.legend(['train acc', 'pos acc', 'neg acc'], loc="upper right")

    plt.title('Loss and accuracy of train.')

    # グラフ右
    plt.subplot(1, 2, 2)
    plt.ylim(ylim1)

    plt.plot(range(1, len(valid_loss) + 1), valid_loss, color='orange', marker='')
    plt.yticks(np.arange(ylim1[0], ylim1[1], .1))
    plt.grid(True)
    # plt.ylabel('loss')
    plt.legend(['valid loss'], loc="lower left")

    plt.twinx()

    plt.ylim(ylim2)
    plt.plot(range(1, len(valid_accuracy1) + 1), valid_accuracy1, color='red', marker='')
    plt.yticks(np.arange(ylim2[0], ylim2[1], .1))
    plt.grid(True)
    plt.ylabel('accuracy')
    plt.legend(['valid acc'], loc="upper right")

    plt.title('Loss and accuracy of valid.')

    plt.savefig(filename)
    # plt.show()
