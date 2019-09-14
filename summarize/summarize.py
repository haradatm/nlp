#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys

reload(sys)
sys.setdefaultencoding('utf-8')
#print sys.getdefaultencoding()

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

import numpy as np
import collections
np.set_printoptions(precision=20)

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import pairwise_distances
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot

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


def morph_mecab(str):
    mcb = ___mecab____(str)
    terms = []
    ret = []

    for i, morph in enumerate(mcb):
        mp = {}
        if len(morph) < 2:
            continue

        if u'EOS' in morph[MCB_POS1] or u'BOS' in morph[MCB_POS1]:
            if i == 0:
                continue
            else:
                ret.append(terms)
                terms = []
                continue

        # 未知語 ('-x 未知語')
        elif len(morph) == 2:
            morph.append(u'*')  # (2) MCB_POS2
            morph.append(u'*')  # (3) MCB_POS3
            morph.append(u'*')  # (4)
            morph.append(u'*')  # (5) MCB_POS4
            morph.append(u'*')  # (6)
            morph.append(u'*')  # (7) MCB_BASE
            morph.append(u'*')  # (8) MCB_READ

        # 文節 (表層,読み,品詞,その他の素性)
        mp[u'form'] = morph[MCB_FORM]

        if len(morph) < (MCB_READ+1) or morph[MCB_READ] == u'*':
            mp[u'read'] = morph[MCB_FORM]
        else:
            mp[u'read'] = morph[MCB_READ]

        if morph[MCB_BASE] == u'*':
            mp[u'base'] = morph[MCB_FORM]
        else:
            mp[u'base'] = morph[MCB_BASE]

        mp[u'pos1'] = morph[MCB_POS1]
        mp[u'pos2'] = morph[MCB_POS2]
        mp[u'pos3'] = morph[MCB_POS4]

        # 補正
        if mp[u'read'] == u'*':
            mp[u'read'] = mp[u'form']

        if mp[u'base'] == u'*':
            mp[u'base'] = mp[u'form']
        mp[u'case'] = u''

        if mp[u'pos2'] == u'格助詞' or mp[u'pos2'] == u'副助詞':
            if len(terms) > 0 and terms[-1][u'pos1'] == u'名詞':
                # 主題(ハ格)
                if mp[u'form'] == u'は':
                    terms[-1][u'case'] = u'ハ格'
                # ガ格
                elif mp[u'form'] == u'が':
                    terms[-1][u'case'] = u'ガ格'
                # ニ格
                elif mp[u'form'] == u'に':
                    terms[-1][u'case'] = u'ニ格'
                # ヲ格
                elif mp[u'form'] == u'を':
                    terms[-1][u'case'] = u'ヲ格'
                # デ格
                elif mp[u'form'] == u'で':
                    terms[-1][u'case'] = u'デ格'
        terms.append(mp)

    if len(terms) > 0:
        ret.append(terms)

    return ret


def word_segmenter(text):
    sents = morph_mecab(text)

    nodes = []
    for morphs in sents:
        for morph in morphs:

            if morph[u'form'] == u'':
                pass

            elif re.search(ur'^[\s!-@\[-`\{-~　、-〜！-＠［-｀]+$', morph[u'form']):
                pass

            elif re.search(ur'^(接尾|非自立)', morph[u'pos2']):
                pass

            elif u'サ変・スル' == morph[u'pos3'] or u'ある' == morph[u'base']:
                pass

            elif re.search(ur'^(名詞|動詞|形容詞)', morph[u'pos1']):
                nodes.append(morph)

            else:
                pass

    words = []
    for node in nodes:
        word = node[u'base'] if node[u'base'] != u'' else node[u'form']
        words.append(word)

    return words


def sent_splitter(text):
    parenthesis = u'（）「」『』【】［］〈〉《》〔〕｛｝””'
    close2open = dict(zip(parenthesis[1::2], parenthesis[0::2]))
    paren_chars = set(parenthesis)
    delimiters = set(u'。．？！\n\r')
    pstack = []
    buff = []

    ret = []

    for i, c in enumerate(text):
        c_next = None
        if i+1 < len(text):
            c_next = text[i + 1]

        # check correspondence of parenthesis
        if c in paren_chars:
            # close
            if c in close2open:
                if len(pstack) > 0 and pstack[-1] == close2open[c]:
                    pstack.pop()
            # open
            else:
                pstack.append(c)

        buff.append(c)
        if c in delimiters:
            if len(pstack) == 0 and c_next not in delimiters:
                ret.append(u''.join(buff).strip())
                buff = []

    if len(buff) > 0:
        ret.append(u''.join(buff).strip())

    return ret


def lexrank(sentences, continuous=True, sim_threshold=0.1, alpha=0.9, plot=False):

    ranker_params = {'max_iter': 1000}
    ranker = nx.pagerank_scipy
    ranker_params['alpha'] = alpha

    graph = nx.DiGraph()

    # sentence -> tf
    sent_tf_list = []
    for sent in sentences:
        words = word_segmenter(sent)
        if len(words) == 0:
            words = [u'']
        tf = collections.Counter(words)
        sent_tf_list.append(tf)

    sent_vectorizer = DictVectorizer(sparse=True)
    sent_vecs = sent_vectorizer.fit_transform(sent_tf_list)

    # compute similarities between senteces
    sim_mat = 1 - pairwise_distances(sent_vecs, sent_vecs, metric='cosine')

    if continuous:
        linked_rows, linked_cols = np.where(sim_mat > 0)
    else:
        linked_rows, linked_cols = np.where(sim_mat >= sim_threshold)

    # create similarity graph
    graph.add_nodes_from(range(sent_vecs.shape[0]))
    for i, j in zip(linked_rows, linked_cols):
        if i == j:
            continue
        weight = sim_mat[i, j] if continuous else 1.0
        graph.add_edge(i, j, {'weight': weight})

    if plot:
        dot = to_pydot(graph, strict='true')
        dot.set_rankdir('TB')
        for e in dot.get_edge_list():
            e.set_label(e.get_attributes()['weight'])
        dot.write_png('lexrank_graph.png', prog='dot')

    scores = ranker(graph, **ranker_params)
    return scores, sim_mat


def summarize(text, sent_limit=None, char_limit=None, imp_require=None, **lexrank_params):

    sentences = list(sent_splitter(text))
    if len(sentences) > 1:
        scores, sim_mat = lexrank(sentences, **lexrank_params)

        if scores:
            sum_scores = sum(scores.itervalues())
            acc_scores = 0.0
            indexes = set()
            num_sent, num_char = 0, 0

            for i in sorted(scores, key=lambda i: scores[i], reverse=True):
                num_sent += 1
                num_char += len(sentences[i])

                if sent_limit is not None and num_sent > sent_limit:
                    break

                if char_limit is not None and num_char > char_limit:
                    break

                if imp_require is not None and acc_scores / sum_scores >= imp_require:
                    break

                indexes.add(i)
                acc_scores += scores[i]
        else:
            indexes = []

    else:
        indexes = [0]

    if len(indexes) > 0:
        summary_sents = [sentences[i] for i in sorted(indexes)]
    else:
        summary_sents = []

    # pp({'sentences': sentences, 'scores': scores})

    return summary_sents


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('--file', default='', type=str, help='plain text file to be summarized')
    parser.add_argument('--sentences', '-s', default=None, type=int, help='summary length (the number of sentences)')
    parser.add_argument('--characters', '-c', default=None, type=int, help='summary length (the number of charactors)')
    parser.add_argument('--imp', '-i', default=None, type=float, help='cumulative LexRank score [0.0-1.0]')
    parser.add_argument('--debug', action='store_true', default=False, help='for debug')
    parser.add_argument('--oneline', action='store_true', default=False, help='for one line text')
    parser.add_argument('--eval', default='', type=unicode, help='text for evaluation')
    args = parser.parse_args()

    path = args.file
    sent_limit = args.sentences
    char_limit = args.characters
    imp_require = args.imp

    lexrank_params = {}

    lines = []

    if args.eval:
        text = unicode(args.eval).strip()
        sentences = summarize(text, sent_limit=sent_limit, char_limit=char_limit, imp_require=imp_require, **lexrank_params)
        print(u'{}'.format(u' '.join(sentences)))
        sys.stdout.flush()

    elif args.oneline:
        for i, line in enumerate(open(path, 'rU')):
            text = unicode(line).strip()
            if text.startswith(u'#'):
                continue

            sentences = summarize(text, sent_limit=sent_limit, char_limit=char_limit, imp_require=imp_require, **lexrank_params)

            print(u'{}\t{}'.format(i+1, u' '.join(sentences)))
            sys.stdout.flush()
    else:
        for line in open(path, 'rU'):
            line = unicode(line).strip()
            if line.startswith(u'#'):
                continue
            lines.append(line)
        text = u'\n'.join(lines)

        sentences = summarize(text, sent_limit=sent_limit, char_limit=char_limit, imp_require=imp_require, **lexrank_params)

        print(u'{}'.format(u'\n'.join(sentences)))
        sys.stdout.flush()

sys.stderr.write('time spent: {}\n'.format(time.time() - start_time))
