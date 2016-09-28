#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# 実験:
# 結果:
#
__version__ = '0.0.1'

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#print sys.getdefaultencoding()

# usage:

import re
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

import pprint
def pp(obj):
    pp = pprint.PrettyPrinter(indent=1, width=160)
    str = pp.pformat(obj)
    print re.sub(r"\\u([0-9a-f]{4})", lambda x: unichr(int("0x"+x.group(1),16)), str)

import os, math, time
start_time = time.time()

import struct
import numpy as np

import struct
FLOAT_SIZE = struct.calcsize('f')

from summarize import sent_splitter

import os, MeCab
CWD = os.getcwd()
tagger = MeCab.Tagger('-r ' + os.path.join(CWD, 'mecabrc'))


def ___mecab____(sent):
    ret = []
    encoded = sent.encode('utf-8')
    node = tagger.parseToNode(encoded)
    while node:
        str = [node.surface.decode('utf-8')]
        str += re.split('[\s,]', (node.feature.decode('utf-8')).strip())
        ret.append(str)
        node = node.next
    return ret


# for MeCab, CaboCha
MCB_FORM = 0
MCB_READ = 8
MCB_BASE = 7
MCB_POS1 = 1
MCB_POS2 = 2
MCB_POS3 = 3
MCB_POS4 = 5


class Morph():
    def __init__(self):
        self.form = None
        self.read = None
        self.base = None
        self.pos1 = None
        self.pos2 = None
        self.pos3 = None


def morph_mecab(str):
    mcb = ___mecab____(str)
    morphs = []
    ret = []

    for i, node in enumerate(mcb):
        morph = Morph()

        if len(node) < 2:
            continue

        if u'EOS' in node[MCB_POS1] or u'BOS' in node[MCB_POS1]:
            if i == 0:
                continue
            else:
                ret.append(morphs)
                morphs = []
                continue

        # 未知語 ('-x 未知語')
        elif len(node) == 2:
            node.append(u'*')  # (2) MCB_POS2
            node.append(u'*')  # (3) MCB_POS3
            node.append(u'*')  # (4)
            node.append(u'*')  # (5) MCB_POS4
            node.append(u'*')  # (6)
            node.append(u'*')  # (7) MCB_BASE
            node.append(u'*')  # (8) MCB_READ

        # 文節 (表層,読み,品詞,その他の素性)
        morph.form = node[MCB_FORM]

        if len(node) < (MCB_READ+1) or node[MCB_READ] == u'*':
            morph.read = node[MCB_FORM]
        else:
            morph.read = node[MCB_READ]

        if node[MCB_BASE] == u'*':
            morph.base = node[MCB_FORM]
        else:
            morph.base = node[MCB_BASE]

        morph.pos1 = node[MCB_POS1]
        morph.pos2 = node[MCB_POS2]
        morph.pos3 = node[MCB_POS4]

        # 補正
        if morph.read == u'*':
            morph.read = morph.form

        if morph.base == u'*':
            morph.base = morph.form
        morph.case = u''

        if morph.pos2 == u'格助詞' or morph.pos2 == u'副助詞':
            if len(morphs) > 0 and morphs[-1].pos1 == u'名詞':
                # 主題(ハ格)
                if morph.form == u'は':
                    morphs[-1].case = u'ハ格'
                # ガ格
                elif morph.form == u'が':
                    morphs[-1].case = u'ガ格'
                # ニ格
                elif morph.form == u'に':
                    morphs[-1].case = u'ニ格'
                # ヲ格
                elif morph.form == u'を':
                    morphs[-1].case = u'ヲ格'
                # デ格
                elif morph.form == u'で':
                    morphs[-1].case = u'デ格'
        morphs.append(morph)

    if len(morphs) > 0:
        ret.append(morphs)

    return ret


POS_FILTER = [
    u'名詞',
    u'動詞',
    u'形容詞',
    u'副詞',
    u'未知語',
]


def word_segmenter(text):
    sents = morph_mecab(text)

    selected = []
    for morphs in sents:
        for morph in morphs:

            if morph.form == u'':
                pass
            elif re.search(ur'^[\s!-@\[-`\{-~　、-〜！-＠［-｀]+$', morph.form):
                pass
            elif re.search(ur'^(接尾|非自立)', morph.pos2):
                pass
            elif u'サ変・スル' == morph.pos3 or u'ある' == morph.base:
                pass
            elif morph.pos1 not in POS_FILTER:
                pass

            else:
                selected.append(morph)

    words = []
    for morph in selected:
        # word = morph.base if morph.base != u'' else morph.form
        word = morph.form
        words.append(word)

    return words


def load_w2v_model(path):
    logging.error('# W2V model: {}'.format(path))

    with open(path, 'rb') as f:
        n_vocab, n_units = map(int, f.readline().split())

        w2i = {}
        i2w = {}
        w = np.empty((n_vocab, n_units), dtype=np.float32)

        for i in xrange(n_vocab):
            word = ''
            while True:
                ch = f.read(1)
                if ch == ' ':
                    break
                word += ch

            try:
                w2i[unicode(word)] = i
                i2w[i] = unicode(word)

            except UnicodeError:
                logging.error('Error unicode(): %s', word)
                w2i[word] = i
                i2w[i] = word

            w[i] = np.zeros(n_units)
            for j in xrange(n_units):
                w[i][j] = struct.unpack('f', f.read(FLOAT_SIZE))[0]

            # 改行を strip する
            assert f.read(1) == '\n'

    # ベクトルを正規化する
    s = np.sqrt((w * w).sum(1))
    w /= s.reshape((s.shape[0], 1))

    logging.error('# W2V vocab size: {}'.format(n_vocab))

    return w, w2i, i2w


def make_one_vector(w2v, vocab, words):

    vec = [0] * len(w2v[0])
    for word in words:
        if word not in vocab:
            logger.error('Word: "{0}" is not found'.format(word))
        else:
            seed_idx = vocab[word]
            seed_vector = w2v[seed_idx]
            vec += seed_vector

    vlen = np.linalg.norm(vec, 2)
    if vlen > 0:
        vec /= vlen
    return vec


from cleanse import cleanse


def build_vector(w2v, vocab, documents, mode=1):

    vectors = np.zeros((len(documents), len(w2v[0])), dtype=np.float)

    for i, text in enumerate(documents):
        words = word_segmenter(cleanse(text))
        if len(words) == 0:
            words = [u'']

        vec = make_one_vector(w2v, vocab, words)
        vectors[i] = vec

    return vectors


def make_cpt_model(doc, w2v):
    X = []
    Y = []
    documents = []

    for i, line in enumerate(open(doc, 'rU')):
        text = unicode(line).strip()

        if text.startswith(u'#'):
            continue

        row = text.split(u'\t')
        if len(row) > 1:
            x = row[1]
            y = row[0]
        else:
            x = row[0]
            y = u'SENT_%s' % (i + 1)

        logger.debug(u'%s\t%s'.format(y, x))

        documents.append(x)

        X.append(x)
        Y.append(y)

    (w2v_model, vocab, inv_vocab) = load_w2v_model(w2v)

    print('# dim %d' % len(w2v_model[0]))

    vector = build_vector(w2v_model, vocab, documents)

    model = {
        'w2v': w2v_model,
        'vocab': vocab,
        'vector': vector,
        'text':  X,
        'label': Y,
    }

    return model


def search(text, model=None, N=10, sort=True):

    for sent in sent_splitter(cleanse(text)):
        print(u'{}'.format(sent))

    words = word_segmenter(cleanse(text))
    if len(words) == 0:
        words = [u'']

    docvec = make_one_vector(model['w2v'], model['vocab'], words)

    vectors = model['vector']
    similarity = vectors.dot(docvec)

    if sort:
        indexes = (-similarity).argsort()
    else:
        indexes = xrange(len(similarity))

    ret = []
    for i, idx in enumerate(indexes):
        if i >= N:
            break

        ret.append({
            'score': '%0.6f' % similarity[idx],
            'label': model['label'][idx],
            'text': model['text'][idx],
        })

    return ret


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--text', type=unicode, default=None, help='text to search')
    parser.add_argument('--train', type=unicode, default=None, help='file for training')
    parser.add_argument('--eval', type=unicode, default=None, help='file for evaluation')
    parser.add_argument('--tqa', action='store_true', default=False, help='evaluation for tqa')
    parser.add_argument('--w2v', type=unicode, default=None, help='word2vec model file (.bin)')
    args = parser.parse_args()

    cpt_model = make_cpt_model(args.train, args.w2v)

    if args.text:
        results = search(args.text, model=cpt_model, N=10)
        pp(results)

    elif args.eval:
        for line in open(args.eval, 'r'):
            text = unicode(line).strip()

            if text.startswith(u'#'):
                continue

            if not args.tqa:
                results = search(text, model=cpt_model, N=10, sort=True)
                for ret in results:
                    print(u'{}\t{}\t{}'.format(ret['label'], ret['score'], ret['text']))
                print

            else:
                results = search(text, model=cpt_model, N=10, sort=True)
                scores = [x['score'] for x in results]
                labels = [x['label'] for x in results]
                texts  = [x['text']  for x in results]
                print(u'{}\t{}\t{}'.format('\t'.join(scores), '\t'.join(labels), '\t'.join(texts)))

sys.stderr.write('time spent: {}\n'.format(time.time() - start_time))
