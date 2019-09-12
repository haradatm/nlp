#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Patient (Convert Patient disease expression to Standard expression using BERT fine-tuning)
    by "triplet loss".
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
from sklearn.metrics.pairwise import cosine_similarity


def load_data(path, tokenizer, bert_config, type_id=0):
    X = []
    id_list = []

    for i, line in enumerate(open(path)):
        # if i >= 100:
        #     break

        line = line.strip()
        line = line.replace(u'. . .', u'…')
        if line == '':
            continue

        label, text = line.split('\t')

        if label not in id_list:
            id_list += [label]

        tokens_a = tokenizer.tokenize(text)
        tokens = ["[CLS]"] if type_id == 0 else []
        segment_ids = [type_id]

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(type_id)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # label_id = labels[label]

        max_position_embeddings = bert_config.max_position_embeddings
        x1 = np.array(input_ids[:max_position_embeddings], 'i')
        x2 = np.array(input_mask[:max_position_embeddings], 'f')
        x3 = np.array(segment_ids[:max_position_embeddings], 'i')
        # y = np.array([label_id], 'i')
        X.append((x1, x2, x3))

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return X, id_list


def load_qrels(path, que_ids, doc_ids):

    qrels = collections.defaultdict(lambda: [])

    for i, line in enumerate(open(path)):
        # if i > 100:
        #     break

        line = line.strip()
        line = line.replace(u'. . .', u'…')
        if line == '':
            continue

        cols = line.split('\t')
        if len(cols) < 4:
            logger.error('invalid record: {}\n'.format(line))
            continue

        que_id = cols[0].strip()
        doc_id = cols[2].strip()
        rel = cols[3].strip()

        if que_id in que_ids and doc_id in doc_ids:
            qrels[que_ids.index(que_id)].append((doc_ids.index(doc_id), int(rel)))

    pos, neg = collections.defaultdict(lambda: []), collections.defaultdict(lambda: [])

    for que_idx, rels in qrels.items():
        neg_doc_ids = list(range(len(doc_ids)))

        for doc_idx, rel in rels:

            if rel > 0:
                pos[que_idx].append(doc_idx)
            else:
                neg[que_idx].append(doc_idx)

            neg_doc_ids.remove(doc_idx)

        for doc_idx in neg_doc_ids:
            neg[que_idx].append(doc_idx)

    logger.info('loading qrels: {}'.format(path))
    sys.stdout.flush()

    return pos, neg


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
class BertRanking(chainer.Chain):
    def __init__(self, bert):
        super(BertRanking, self).__init__()
        with self.init_scope():
            self.bert = bert
            self.output = Linear3D(None, 512, initialW=chainer.initializers.Normal(0.02))

    def forward(self, input_ids, input_mask, token_type_ids):
        output_layer = self.bert.get_pooled_output(input_ids, input_mask, token_type_ids)
        output_layer = F.dropout(output_layer, 0.1)
        output_layer = self.output(output_layer)
        return output_layer

    def __call__(self, q_x1, q_x2, q_x3, d_x1, d_x2, d_x3, n_x1, n_x2, n_x3):
        y_0 = self.forward(q_x1, q_x2, q_x3)
        y_1 = self.forward(d_x1, d_x2, d_x3)
        y_2 = self.forward(n_x1, n_x2, n_x3)
        loss = F.triplet(y_0, y_1, y_2)
        return loss


def batch_iter(queries, docs, qrels, batch_size):

    pos, neg = qrels

    data = []
    for que_idx in pos.keys():

        pos_indexes = pos[que_idx]
        neg_indexes = neg[que_idx]

        for i in pos_indexes:
            j = np.random.choice(neg_indexes)
            q_x1, q_x2, q_x3 = queries[que_idx]
            p_x1, p_x2, p_x3 = docs[i]
            n_x1, n_x2, n_x3 = docs[j]

            # data.append((q_x1, q_x2, q_x3, p_x1, p_x2, p_x3, np.array(1, 'i')))
            # data.append((q_x1, q_x2, q_x3, n_x1, n_x2, n_x3, np.array(0, 'i')))

            data.append((q_x1, q_x2, q_x3, p_x1, p_x2, p_x3, n_x1, n_x2, n_x3))

    batch = []
    for line in sorted(data, key=lambda x: len(x[0]), reverse=True):
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


def trec_eval(qrel, results, k):

    import pytrec_eval

    with open(qrel, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(results, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run)

    measures = ['map_cut_{:}'.format(k), 'ndcg_cut_{:}'.format(k)]

    acc1 = pytrec_eval.compute_aggregated_measure(measures[0], [query_measures[measures[0]] for query_measures in results.values()])
    acc2 = pytrec_eval.compute_aggregated_measure(measures[1], [query_measures[measures[1]] for query_measures in results.values()])

    return float(acc1), float(acc2)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='BERT example: Text Retrieval (by triplet loss)')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=32, type=int, help='learning batchsize size')
    parser.add_argument('--model', default='models/early_stopped-ngcd.model', type=str, help='model directory')
    parser.add_argument('--train_docs', default='datasets/patient/train-wakachi.docs', type=str, help='training document data file (.txt)')
    parser.add_argument('--train_queries', default='datasets/patient/train-wakachi.queries', type=str, help='training query data file (.txt)')
    parser.add_argument('--train_qrels', default='datasets/patient/train.qrel', type=str, help='training query relevance file (.qrel)')
    parser.add_argument('--test_docs', default='datasets/patient/test-wakachi.docs', type=str, help='testing document data file (.txt)')
    parser.add_argument('--test_queries', default='datasets/patient/test-wakachi.queries', type=str, help='testing query data file (.txt)')
    parser.add_argument('--test_qrels', default='datasets/patient/test.qrel', type=str, help='testing query relevance file (.qrel)')
    parser.add_argument('--init_checkpoint', default='BERT/Japanese_L-12_H-768_A-12_E-30_BPE/arrays_bert_model.ckpt.npz', type=str, help='initial checkpoint (usually from a pre-trained BERT model (.npz)')
    parser.add_argument('--bert_config_file', default='BERT/Japanese_L-12_H-768_A-12_E-30_BPE/bert_config.json', type=str, help='json file corresponding to the pre-trained BERT model (.json)')
    parser.add_argument('--vocab_file', default='BERT/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt', type=str, help='vocabulary file that the BERT model was trained on (.txt)')
    parser.add_argument('--out', '-o', default='results_patient-6', type=str, help='output directory')
    parser.add_argument('--max_eval', default=1000, type=int, help='number of evaluations')
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

    vocab_file = args.vocab_file
    bert_config_file = args.bert_config_file
    init_checkpoint = args.init_checkpoint

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    train_queries, train_que_ids = load_data(args.train_queries, tokenizer, bert_config, type_id=0)
    train_docs, train_doc_ids = load_data(args.train_docs, tokenizer, bert_config, type_id=0)
    train_qrels = load_qrels(args.train_qrels, train_que_ids, train_doc_ids)

    test_queries, test_que_ids = load_data(args.test_queries, tokenizer, bert_config, type_id=0)
    test_docs, test_doc_ids = load_data(args.test_docs, tokenizer, bert_config, type_id=0)
    test_qrels = load_qrels(args.test_qrels, test_que_ids, test_doc_ids)

    print('# train queries: {}, docs: {}, pos: {}, neg: {}'.format(len(train_queries), len(train_docs), len(train_qrels[0]), len(train_qrels[1])))
    print('# test  queries: {}, docs: {}, pos: {}, neg: {}'.format(len(test_queries),   len(test_docs),  len(test_qrels[0]),  len(test_qrels[1])))
    print('# vocab: {}'.format(len(tokenizer.vocab)))
    sys.stdout.flush()

    # Setup model
    bert = BertModel(config=bert_config)
    model = BertRanking(bert)
    chainer.serializers.load_npz(init_checkpoint, model, ignore_names=['output/W', 'output/b'])

    if args.gpu >= 0:
        model.to_gpu()

    # test (early_stopped model by uar)
    chainer.serializers.load_npz(os.path.join(args.model), model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    start_at = time.time()
    cur_at = start_at

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        with open('test.results.txt', 'w') as f:
            q_hs = []
            for x1, x2, x3 in test_queries:
                x1 = to_device(args.gpu, F.pad_sequence([x1], length=None, padding=0).array).astype('i')
                x2 = to_device(args.gpu, F.pad_sequence([x2], length=None, padding=0).array).astype('f')
                x3 = to_device(args.gpu, F.pad_sequence([x3], length=None, padding=0).array).astype('i')
                q_vec = model.forward(x1, x2, x3)
                q_vec = cuda.to_cpu(q_vec.data[0])
                q_hs.append(q_vec)
            q_vecs = np.asarray(q_hs, 'f')

            doc_hs = []
            for x1, x2, x3 in test_docs:
                x1 = to_device(args.gpu, F.pad_sequence([x1], length=None, padding=0).array).astype('i')
                x2 = to_device(args.gpu, F.pad_sequence([x2], length=None, padding=0).array).astype('f')
                x3 = to_device(args.gpu, F.pad_sequence([x3], length=None, padding=0).array).astype('i')
                doc_vec = model.forward(x1, x2, x3)
                doc_vec = cuda.to_cpu(doc_vec.data[0])
                doc_hs.append(doc_vec)
            doc_vecs = np.asarray(doc_hs, 'f')

            for i, q_vec in enumerate(q_vecs):
                similarities = cosine_similarity(q_vec[None, :], doc_vecs)
                for j, k in enumerate(similarities[0].argsort()[-1:-(args.max_eval + 1):-1]):
                    f.write("{}\tQ0\t{}\t{}\t{:.6f}\tSTANDARD\n".format(test_que_ids[i], test_doc_ids[k], j, similarities[0][k]))
                    f.flush()

        now = time.time()
        throughput = now - cur_at

        at_K = 20
        acc1, acc2 = trec_eval(args.test_qrels, 'test.results.txt', k=at_K)
        print('#=> map@{0:}={1:.6f}, ngcd@{0:}={2:.6f}, sec={3:.6f}'.format(at_K, acc1, acc2, throughput))


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
