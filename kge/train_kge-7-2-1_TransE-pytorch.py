#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" KGE: Knowledge Graph Embedding example
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


start_time = time.time()

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
# from utils import get_minibatches, sample_negatives, accuracy, auc
# from time import time


class TransE(nn.Module):
    """
    TransE embedding model
    ----------------------
    Bordes, Antoine, et al.
    "Translating embeddings for modeling multi-relational data." NIPS. 2013.
    """

    def __init__(self, n_e, n_r, k, margin, distance='l2', gpu=False):
        """
        TransE embedding model
        ----------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.

            margin: float
                Margin size for TransE's hinge loss.

            distance: {'l1', 'l2'}
                Distance measure to be used in the loss.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(TransE, self).__init__()
        # Parameters
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.gpu = gpu
        self.distance = distance
        self.gamma = margin
        # Embedding Layer
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)
        # Initialization
        r = 6 / np.sqrt(self.k)
        self.emb_E.weight.data.uniform_(-r, r)
        self.emb_R.weight.data.uniform_(-r, r)
        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X):
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X
        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)
        f = self.energy(e_hs, e_ls, e_ts).view(-1, 1)
        return f

    def energy(self, h, l, t):
        if self.distance == 'l1':
            out = torch.sum(torch.abs(h + l - t), 1)
        else:
            out = torch.sqrt(torch.sum((h + l - t) ** 2, 1))
        return out

    def ranking_loss(self, y_pos, y_neg, C=1, average=True):
        """
        Compute loss max margin ranking loss.

        Params:
        -------
        y_pos: vector of size Mx1
            Contains scores for positive samples.

        y_neg: np.array of size Mx1 (binary)
            Contains the true labels.

        margin: float, default: 1
            Margin used for the loss.

        C: int, default: 1
            Number of negative samples per positive sample.

        average: bool, default: True
            Whether to average the loss or just summing it.

        Returns:
        --------
        loss: float
        """
        M = y_pos.size(0)

        y_pos = y_pos.view(-1).repeat(C)  # repeat to match y_neg
        y_neg = y_neg.view(-1)
        target = Variable(torch.from_numpy(-np.ones(M * C, dtype=np.float32)))
        loss = nn.MarginRankingLoss(margin=self.gamma)
        loss = loss(y_pos, y_neg, target)
        return loss

    def normalize_embeddings(self):
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def predict(self, X, sigmoid=False):

        y_pred = self.forward(X).view(-1, 1)

        if sigmoid:
            y_pred = F.sigmoid(y_pred)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()


from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import roc_auc_score


def get_minibatches(X, mb_size, shuffle=True):
    """
    Generate minibatches from given dataset for training.

    Params:
    -------
    X: np.array of M x 3
        Contains the triplets from dataset. The entities and relations are
        translated to its unique indices.

    mb_size: int
        Size of each minibatch.

    shuffle: bool, default True
        Whether to shuffle the dataset before dividing it into minibatches.

    Returns:
    --------
    mb_iter: generator
        Example usage:
        --------------
        mb_iter = get_minibatches(X_train, mb_size)
        for X_mb in mb_iter:
            // do something with X_mb, the minibatch
    """
    minibatches = []
    X_shuff = np.copy(X)

    if shuffle:
        X_shuff = skshuffle(X_shuff)

    for i in range(0, X_shuff.shape[0], mb_size):
        yield X_shuff[i:i + mb_size]


def sample_negatives(X, n_e):
    """
    Perform negative sampling by corrupting head or tail of each triplets in
    dataset.

    Params:
    -------
    X: int matrix of M x 3, where M is the (mini)batch size
        First column contains index of head entities.
        Second column contains index of relationships.
        Third column contains index of tail entities.

    n_e: int
        Number of entities in dataset.

    Returns:
    --------
    X_corr: int matrix of M x 3, where M is the (mini)batch size
        Similar to input param X, but at each column, either first or third col
        is subtituted with random entity.
    """
    M = X.shape[0]

    corr = np.random.randint(n_e, size=M)
    e_idxs = np.random.choice([0, 2], size=M)

    X_corr = np.copy(X)
    X_corr[np.arange(M), e_idxs] = corr

    return X_corr


def accuracy(y_pred, y_true, thresh=0.5, reverse=False):
    """
    Compute accuracy score.

    Params:
    -------
    y_pred: np.array
        Predicted (Bernoulli) probabilities.

    y_true: np.array, binary
        True (Bernoulli) labels.

    thresh: float, default: 0.5
        Classification threshold.

    reverse: bool, default: False
        If it is True, then classify (y <= thresh) to be 1.
    """
    y = (y_pred >= thresh) if not reverse else (y_pred <= thresh)
    return np.mean(y == y_true)


def auc(y_pred, y_true):
    """
    Compute area under ROC curve score.

    Params:
    -------
    y_pred: np.array
        Predicted (Bernoulli) probabilities.

    y_true: np.array, binary
        True (Bernoulli) labels.
    """
    return roc_auc_score(y_true, y_pred)


def main():
    # Set random seed
    randseed = 9999
    np.random.seed(randseed)
    torch.manual_seed(randseed)

    # Data Loading
    # Load dictionary lookups
    idx2ent = np.load('datasets/SDA/wordnet/bin/idx2ent.npy')
    idx2rel = np.load('datasets/SDA/wordnet/bin/idx2rel.npy')

    n_e = len(idx2ent)
    n_r = len(idx2rel)

    # Load dataset
    X_train = np.load('datasets/SDA/wordnet/bin/train.npy')
    X_val = np.load('datasets/SDA/wordnet/bin/val.npy')
    y_val = np.load('datasets/SDA/wordnet/bin/y_val.npy')

    X_val_pos = X_val[y_val.ravel() == 1, :]  # Take only positive samples

    M_train = X_train.shape[0]
    M_val = X_val.shape[0]

    # Model Parameters
    k = 50
    distance = 'l2'
    margin = 1.0
    model = TransE(n_e=n_e, n_r=n_r, k=k, margin=margin, distance=distance, gpu=False)

    normalize_embed = True
    C = 5  # Negative Samples
    n_epoch = 20
    lr = 0.1
    lr_decay_every = 20
    # weight_decay = 1e-4
    mb_size = 100
    print_every = 100
    average = False
    # Optimizer Initialization
    # solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    solver = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Begin training
    for epoch in range(n_epoch):
        print('Epoch-{}'.format(epoch + 1))
        print('----------------')
        it = 0
        # Shuffle and chunk data into minibatches
        mb_iter = get_minibatches(X_train, mb_size, shuffle=True)

        # Anneal learning rate
        lr = lr * (0.5 ** (epoch // lr_decay_every))
        for param_group in solver.param_groups:
            param_group['lr'] = lr

        for X_mb in mb_iter:
            start = time.time()

            # Build batch with negative sampling
            m = X_mb.shape[0]
            # C x M negative sampls
            X_neg_mb = np.vstack([sample_negatives(X_mb, n_e) for _ in range(C)])
            X_train_mb = np.vstack([X_mb, X_neg_mb])

            y_true_mb = np.vstack([np.zeros([m, 1]), np.ones([C * m, 1])])

            # Training step
            y = model.forward(X_train_mb)
            y_pos, y_neg = y[:m], y[m:]
            loss = model.ranking_loss(y_pos, y_neg, C=C, average=average)
            loss.backward()
            solver.step()
            solver.zero_grad()
            if normalize_embed:
                model.normalize_embeddings()

            end = time.time()
            # Training logs
            if it % print_every == 0:
                # Training auc
                pred = model.predict(X_train_mb, sigmoid=True)
                train_acc = auc(pred, y_true_mb)

                # Validation auc
                y_pred_val = model.forward(X_val)
                y_prob_val = F.sigmoid(y_pred_val)
                y_prob_val = 1 - y_prob_val
                val_acc = auc(y_prob_val.data.numpy(), y_val)

                print('Iter-{}; loss: {:.4f}; train_auc: {:.4f}; val_auc: {:.4f}; time per batch: {:.2f}s'
                      .format(it, loss.data[0], train_acc, val_acc, end - start))
                sys.stdout.flush()

                import plot_util
                plot_util.plot(loss.data, train_acc, .0, .0, .0, val_acc,
                               '{}.png'.format(os.path.splitext(os.path.basename(__file__))[0]))

            it += 1

    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
