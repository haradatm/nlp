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

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt


# Network definition
class RESCAL(chainer.Chain):
    """
    RESCAL: bilinear model
    ----------------------
    Nickel, Maximilian, Volker Tresp, and Hans-Peter Kriegel.
    "A three-way model for collective learning on multi-relational data."
    ICML. 2011.
    """

    def __init__(self, n_e, n_r, k, lam, gpu=False):
        """
        RESCAL: bilinear model
        ----------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.

            lam: float
                Prior strength of the embeddings. Used to constaint the
                embedding norms inside a (euclidean) unit ball. The prior is
                Gaussian, this param is the precision.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(RESCAL, self).__init__()
        with self.init_scope():

            # Hyperparams
            self.n_e = n_e
            self.n_r = n_r
            self.k = k
            self.lam = lam
            self.gpu = gpu

            # Nets
            self.emb_E = L.EmbedID(self.n_e, self.k, initialW=chainer.initializers.HeNormal())
            self.emb_R = L.EmbedID(self.n_r, self.k ** 2, initialW=chainer.initializers.HeNormal())

    def forward(self, X):
        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        W = F.reshape(self.emb_R(ls), (-1, self.k, self.k)) # M x k x k

        # Forward
        out = F.matmul(F.expand_dims(e_hs, 2), W, transa=True, transb=False)        # h^T W
        out = F.matmul(out, F.expand_dims(e_ts, 2), transa=False, transb=False)     # (h^T W) h
        out = F.reshape(out, (-1, 1))

        return out

    def predict(self, X, sigmoid=False):
        y_pred = self.forward(X)
        if sigmoid:
            y_pred = F.sigmoid(y_pred)

        return y_pred

    def log_loss(self, y_pred, y_true, average=True):
        """
        Compute log loss (Bernoulli NLL).

        Params:
        -------
        y_pred: vector of size Mx1
            Contains prediction logits.

        y_true: np.array of size Mx1 (binary)
            Contains the true labels.

        average: bool, default: True
            Whether to average the loss or just summing it.

        Returns:
        --------
        loss: float
        """

        # nll = F.binary_cross_entropy_with_logits(y_pred, y_true, size_average=average)
        nll = F.sigmoid_cross_entropy(y_pred, y_true)

        norm_E = xp.linalg.norm(self.emb_E.W.data, axis=1)
        norm_R = xp.linalg.norm(self.emb_R.W.data, axis=1)

        # Penalize when embeddings norms larger than one
        nlp1 = xp.sum(xp.clip(norm_E - 1, a_min=0, a_max=None))
        nlp2 = xp.sum(xp.clip(norm_R - 1, a_min=0, a_max=None))

        if average:
            nlp1 /= norm_E.shape[0]
            nlp2 /= norm_R.shape[0]

        return nll + self.lam*nlp1 + self.lam*nlp2

    # def ranking_loss(self, y_pos, y_neg, margin=1, C=1, average=True):
    #     """
    #     Compute loss max margin ranking loss.
    #
    #     Params:
    #     -------
    #     y_pos: vector of size Mx1
    #         Contains scores for positive samples.
    #
    #     y_neg: np.array of size Mx1 (binary)
    #         Contains the true labels.
    #
    #     margin: float, default: 1
    #         Margin used for the loss.
    #
    #     C: int, default: 1
    #         Number of negative samples per positive sample.
    #
    #     average: bool, default: True
    #         Whether to average the loss or just summing it.
    #
    #     Returns:
    #     --------
    #     loss: float
    #     """
    #     M = y_pos.size(0)
    #
    #     y_pos = y_pos.view(-1).repeat(C)  # repeat to match y_neg
    #     y_neg = y_neg.view(-1)
    #
    #     # target = [-1, -1, ..., -1], i.e. y_neg should be higher than y_pos
    #     target = -np.ones(M*C, dtype=np.float32)
    #
    #     loss = F.margin_ranking_loss(y_pos, y_neg, target, margin=margin, size_average=average)
    #
    #     return loss
    #
    # def normalize_embeddings(self):
    #     for e in self.embeddings:
    #         e.W.data.renorm_(p=2, dim=0, maxnorm=1)
    #
    # def initialize_embeddings(self):
    #     r = 6/np.sqrt(self.k)
    #
    #     for e in self.embeddings:
    #         e.W.data.uniform_(-r, r)
    #
    #     self.normalize_embeddings()


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
    global xp

    import argparse
    parser = argparse.ArgumentParser(description='KGE example: RESCAL')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np

    # Set random seed
    xp.random.seed(123)

    # Data Loading
    # Load dictionary lookups
    idx2ent = xp.load('datasets/SDA/kinship/bin/idx2ent.npy')
    idx2rel = xp.load('datasets/SDA/kinship/bin/idx2rel.npy')

    n_e = len(idx2ent)
    n_r = len(idx2rel)

    # Load dataset
    X_train = xp.load('datasets/SDA/kinship/bin/train.npy')
    X_val = xp.load('datasets/SDA/kinship/bin/val.npy')
    y_val = xp.load('datasets/SDA/kinship/bin/y_val.npy').astype(np.int32)

    X_val_pos = X_val[y_val.ravel() == 1, :]  # Take only positive samples

    M_train = X_train.shape[0]
    M_val = X_val.shape[0]

    # Model Parameters
    k = 50
    embeddings_lambda = 0

    model = RESCAL(n_e=n_e, n_r=n_r, k=k, lam=embeddings_lambda, gpu=False)

    # Negative Samples
    C = 10

    # Optimizer Initialization
    n_epoch = 20
    lr = 0.1
    lr_decay_every = 20
    # weight_decay = 1e-4
    mb_size = 100
    print_every = 100
    average = True

    # Setup optimizer (Optimizer の設定)
    # optimizer = chainer.optimizers.Adam()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Begin training
    for epoch in range(n_epoch):
        print('Epoch-{}'.format(epoch + 1))
        print('----------------')
        it = 0

        # Shuffle and chunk data into minibatches
        mb_iter = get_minibatches(X_train, mb_size, shuffle=True)

        # # Anneal learning rate
        # lr = lr * (0.5 ** (epoch // lr_decay_every))
        # for param_group in solver.param_groups:
        #     param_group['lr'] = lr

        for X_mb in mb_iter:
            start = time.time()

            # Build batch with negative sampling
            m = X_mb.shape[0]

            # C x M negative samples
            X_neg_mb = sample_negatives(X_mb, n_e)
            X_train_mb = xp.vstack([X_mb, X_neg_mb])

            y_true_mb = xp.vstack([xp.ones([m, 1], dtype=np.int32), xp.zeros([m, 1], dtype=np.int32)])

            # 勾配を初期化
            model.cleargrads()

            # Training step
            y = model.forward(X_train_mb)
            loss = model.log_loss(y, y_true_mb, average=average)

            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

            # if normalize_embed:
            #     model.normalize_embeddings()

            end = time.time()

            # Training logs
            if it % print_every == 0:

                with chainer.no_backprop_mode(), chainer.using_config('train', False):

                    # Training auc
                    pred = model.predict(X_train_mb, sigmoid=True)
                    train_acc = auc(pred.data, y_true_mb)

                    # Per class accuracy
                    pos_acc = accuracy(pred.data[:m], y_true_mb.data[:m])
                    neg_acc = accuracy(pred.data[m:], y_true_mb.data[m:])

                    # Validation auc
                    y_pred_val = model.forward(X_val)
                    y_prob_val = F.sigmoid(y_pred_val)

                    val_acc = auc(y_prob_val.data, y_val)

                    # Validation loss
                    val_loss = model.log_loss(y_pred_val, y_val, average)

                    print('Iter-{}; loss: {:.4f}; train_auc: {:.4f}; val_auc: {:.4f}; val_loss: {:.4f}; pos_acc: {:.4f}; neg_acc: {:.4f}; time per batch: {:.2f}s'.format(
                            it, loss.data, train_acc, val_acc, val_loss.data, pos_acc, neg_acc, end - start))
                    sys.stdout.flush()

                    import plot_util
                    plot_util.plot(loss.data, train_acc, pos_acc, neg_acc, val_loss.data, val_acc, '{}.png'.format(os.path.splitext(os.path.basename(__file__))[0]))

            it += 1

    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
