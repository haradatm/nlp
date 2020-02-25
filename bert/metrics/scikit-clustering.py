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

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from collections import Counter


def main():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input',  default='datasets/rt-bertembed.txt', type=str, help='input file (.txt)')
    parser.add_argument('--output', default='result.png', type=str, help='output file (.png)')
    parser.add_argument('--samples', default=None, type=int, help='number of samples')
    parser.add_argument('--plots', default=None, type=int, help='number of samples')
    parser.add_argument('--dim2', default=False, action='store_true', help='use t-SNE features')
    # parser.set_defaults(test=True)
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 43
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    x, y = [], []
    for i, line in enumerate(open(args.input)):
        cols = line.strip().split("\t")
        label = cols[0]
        feature = np.array(list(map(lambda x: float(x), cols[1:])), 'f')
        y.append(label)
        x.append(feature)

    n_samples = len(x)
    if args.samples is not None and (n_samples > args.samples):
        n_samples = args.samples
    class_count = Counter(y)
    n_class = len(class_count)
    logger.info('Loading dataset ... done.')
    logger.info('# input: {}, class: {}, {}'.format(n_samples, n_class, class_count))

    if len(x) > n_samples:
        x_sample, _, y_sample, _ = train_test_split(x, y, train_size=n_samples, random_state=seed, stratify=y)
        X, y = np.vstack(x_sample), y_sample
    else:
        X = np.vstack(x)

    n_samples = X.shape[0]
    class_count = Counter(y)
    n_class = len(class_count)
    logger.info('# sample: {}, class: {}, {}'.format(n_samples, n_class, class_count))

    Y, label2id = [], {}
    for label in y:
        if label not in label2id:
            label2id[label] = len(label2id)
        Y.append(label2id[label])
    Y = np.array(Y, 'i')

    # id2label = {v: k for k, v in label2id.items()}
    logger.info('# labels: {}'.format(label2id))

    tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
    if args.dim2:
        dataset = (tsne, Y)
    else:
        dataset = (X, Y)

    n_plots = args.plots if args.plots is not None else n_samples
    perm = np.random.permutation(n_samples)[0:n_plots]

    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(10 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': n_class,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}

    # _datasets = [
    #     (noisy_circles, {'damping': .77, 'preference': -240, 'quantile': .2, 'n_clusters': 2, 'min_samples': 20, 'xi': 0.25}),
    #     (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    #     (varied, {'eps': .18, 'n_neighbors': 2, 'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
    #     (aniso, {'eps': .15, 'n_neighbors': 2, 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
    #     (blobs, {}),
    #     (no_structure, {})
    # ]

    _datasets = [
        (dataset, {'damping': .77, 'preference': -240, 'quantile': .2, 'n_clusters': n_class, 'min_samples': 20, 'xi': 0.25}),
        (dataset, {'damping': .75, 'preference': -220, 'n_clusters': n_class}),
        (dataset, {'eps': .18, 'n_neighbors': 2, 'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (dataset, {'eps': .15, 'n_neighbors': 2, 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (dataset, {}),
        (dataset, {})
    ]

    for i_dataset, (dataset, algo_params) in enumerate(_datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # # normalize dataset for easier parameter selection
        # X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # ============
        # Create cluster objects
        # ============
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
        spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=params['eps'])
        optics = cluster.OPTICS(min_samples=params['min_samples'], xi=params['xi'], min_cluster_size=params['min_cluster_size'])
        affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
        average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=params['n_clusters'], connectivity=connectivity)
        birch = cluster.Birch(n_clusters=params['n_clusters'])
        gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

        clustering_algorithms = (
            ('None', None),
            ('MiniBatchKMeans', two_means),
            ('AffinityPropagation', affinity_propagation),
            ('MeanShift', ms),
            ('SpectralClustering', spectral),
            ('Ward', ward),
            ('AgglomerativeClustering', average_linkage),
            ('DBSCAN', dbscan),
            ('OPTICS', optics),
            ('Birch', birch),
            ('GaussianMixture', gmm)
        )

        for i_algorithm, (name, algorithm) in enumerate(clustering_algorithms):

            if i_algorithm == 0:
                if n_samples > n_plots:
                    X_plot, y_plot = tsne[perm, :], y[perm]
                else:
                    X_plot, y_plot = tsne, y

                plt.subplot(len(_datasets), len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)

                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']), int(max(y_plot) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(X_plot[:, 0], X_plot[:, 1], s=10, color=colors[y_plot])

                # plt.xlim(-2.5, 2.5)
                # plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())

            elif not args.dim2 and name == 'MeanShift':
                logger.info('%s algorithm ... is skipped.' % name)

                plt.subplot(len(_datasets), len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)

            else:
                logger.info('%s algorithm ...' % name)

                t0 = time.time()

                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="the number of connected components of the connectivity matrix is [0-9]{1,2} > 1. Completing it to avoid stopping the tree early.", category=UserWarning)
                    warnings.filterwarnings("ignore", message="Graph is not fully connected, spectral embedding may not work as expected.", category=UserWarning)
                    algorithm.fit(X)

                t1 = time.time()

                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                else:
                    y_pred = algorithm.predict(X)

                if n_samples > n_plots:
                    X_plot, y_plot = tsne[perm, :], y_pred[perm]
                else:
                    X_plot, y_plot = tsne, y_pred

                plt.subplot(len(_datasets), len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)

                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']), int(max(y_plot) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(X_plot[:, 0], X_plot[:, 1], s=10, color=colors[y_plot])

                # plt.xlim(-2.5, 2.5)
                # plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=15, horizontalalignment='right')

            plot_num += 1

    plt.savefig(args.output)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
