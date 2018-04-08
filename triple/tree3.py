#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, time, logging, re
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

import os, re
import unicodedata

han_alpha_lower = 'abcdefghijklmnopqrstuvwxyz'
han_alpha_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
han_number      = '0123456789'
han_kigou       = '!\"#$%&\'()=~|`{+*}<>?_-‐^\\@[;:],./'
han_blank       = ' '
zen_alpha_lower = 'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
zen_alpha_upper = 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
zen_number      = '０１２３４５６７８９'
zen_kigou       = '！”＃＄％＆’（）＝〜｜｀｛＋＊｝＜＞？＿−−＾￥＠［；：］，．／'
zen_blank       = '　'

trans_h2z = [
    [han_alpha_lower, zen_alpha_lower],
    [han_alpha_upper, zen_alpha_upper],
    [han_number,      zen_number     ],
    [han_kigou,       zen_kigou      ],
    [han_blank,       zen_blank      ],
]
trans_map_h2z = {}
for pairs in trans_h2z:
    for l in range(len(pairs[0])):
        from_c = ord(pairs[0][l])
        to_c = pairs[1][l]
        trans_map_h2z[from_c] = to_c

trans_z2h = [
    [zen_alpha_lower, han_alpha_lower],
    [zen_alpha_upper, han_alpha_upper],
    [zen_number,      han_number     ],
  # [zen_kigou,       han_kigou      ],
  # [zen_blank,       han_blank      ],
]
trans_map_z2h = {}
for pairs in trans_z2h:
    for l in range(len(pairs[0])):
        from_c = ord(pairs[0][l])
        to_c = pairs[1][l]
        trans_map_z2h[from_c] = to_c


def normalize(text):

    # 半角カナ→全角カナ変換する
    text = unicodedata.normalize('NFKC', text)

    # 半角→全角変換する (全て)
    text = text.translate(trans_map_h2z)

    return text


def cleans(text):

    # 改行コードを除去する
    text = re.sub('\r\n', ' ', text)
    text = re.sub('\r', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)

    # 半角カナ→全角カナ変換する
    text = normalize(text)

    # URL 文字列を除去する
    text = re.sub(r'(ｈｔｔｐｓ?|ｆｔｐ)(：／／[^　\\S]+)', '', text)

    # パス文字列を除去する
    text = re.sub(r'([ａ-ｚＡ-Ｚ]：)*￥[^　\\S]*', '', text)

    # <br> を除去する
    text = re.sub(r'＜[Ｂｂ][Ｒｒ]＞', '　', text)

    # RT 除去する
    text = re.sub(r'^ＲＴ[　\\S]*＠[^　\\S]+', '', text)

    # RT 文字列を除去する (あれば)
    text = re.sub(r'ＲＴ[　\\S]*＠[^　\\S]+', '', text)

    # アカウント名を除去する
    text = re.sub(r'＠[^　\\S]+', '', text)

    # ハッシュ文字列を除去する
    text = re.sub(r'＃[^　\\S]+', '', text)

    # タグ文字のサニタイズを復元する
    text = re.sub(r'＆ｌｔ；', '＜', text)
    text = re.sub(r'＆ｇｔ；', '＞', text)
    text = re.sub(r'＆ａｍｐ；', '＆', text)

    # 連続する長音を1つにする
    text = re.sub(r'ー+', 'ー', text)

    # トリミングする
    text = text.strip()

    return text


def split(text):

    patterns = r'(' \
           r'(?:.*?)' \
           r'(?:' \
               r'[！？。．…；ｗ♪→←]+[！？。．…；ｗ♪→←]*' \
               r'|' \
               r'＜ｅｏｓ＞' \
           r')' \
           r')'

    result = []
    match = re.split(patterns, text)
    for m in match:
        m = m.strip(r'＜ｅｏｓ＞')
        if m != '':
            result.append(m.strip())
    return result


# for CaboCha
import CaboCha
CWD = os.getcwd()
cabocha = CaboCha.Parser('-f1 -O4 -n0 -r ' + CWD + '/nlplib.conf/cabocharc -b ' + CWD + '/nlplib.conf/mecabrc')
def __cabocha__(sent):
    ret = []
    tree = cabocha.parse(sent)
    lines = tree.toString(CaboCha.FORMAT_LATTICE).split('\n')
    for str in lines:
        str = str.strip()
        if str:
            arr = re.split('[\s,]', str)
            ret.append(arr)
    return ret

# for MeCab, CaboCha
MCB_FORM = 0
MCB_READ = 8
MCB_BASE = 7
MCB_POS1 = 1
MCB_POS2 = 2
MCB_POS3 = 3
MCB_POS4 = 5


class Chunk():
    def __init__(self):
        self.fr = -1
        self.to = -1
        self.form = []
        self.read = []
        self.base = []
        self.pos1 = []
        self.pos2 = []
        self.pos3 = []
        self.pos4 = []
        self.case = []
        self.predicate = []
        self.argment = []


def parse_cabocha(str):
    morphs = __cabocha__(str)
    chunks = []
    ret = []

    chunk = Chunk()
    j = 1
    for i, morph in enumerate(morphs):
        # 文節処理
        if morph[MCB_FORM] == '#' or \
           morph[MCB_FORM] == '*' or \
           morph[MCB_FORM] == 'EOS':

            if i > 0:
                # 文節情報
                chunks.append(chunk)
                chunk = Chunk()
                j += 1

                if morph[MCB_FORM] == 'EOS':
                    ret.append(chunks)
                    chunks = []

            if morph[MCB_FORM] == '*':
                # 文節番号
                chunk.fr = j
                chunk.to = int(morph[2][:-1])+1
        else:
            # 未知語 ('-x 未知語')
            if len(morph) == 2:
                morph.append('*')  # (2) MCB_POS2
                morph.append('*')  # (3) MCB_POS3
                morph.append('*')  # (4)
                morph.append('*')  # (5) MCB_POS4
                morph.append('*')  # (6)
                morph.append('*')  # (7) MCB_BASE
                morph.append('*')  # (8) MCB_READ

            # if morph[MCB_POS2] == '句点' or morph[MCB_POS2] == '読点':
            #     continue
            if morph[MCB_FORM] in ['。', '．', '、', '，']:
                continue

            # 文節 (表層,読み,品詞,その他の素性)
            if len(chunk.form) > 0:
                chunk.form.append(morph[MCB_FORM])
            else:
                chunk.form = [morph[MCB_FORM]]

            st = ''
            if len(morph) < (MCB_READ+1) or morph[MCB_READ] == '*':
                st = morph[MCB_FORM]
            else:
                st = morph[MCB_READ]
            chunk.read.append(st)

            st = ''
            if morph[MCB_BASE] == '*':
                st = morph[MCB_FORM]
            else:
                st = morph[MCB_BASE]
            chunk.base.append(st)

            chunk.pos1.append(morph[MCB_POS1])
            chunk.pos2.append(morph[MCB_POS2])
            chunk.pos3.append(morph[MCB_POS3])
            chunk.pos4.append(morph[MCB_POS4])

            # 格, 項
            # if morph[MCB_POS2] == '格助詞' \
            #         or morph[MCB_POS2] == '係助詞' \
            #         or morph[MCB_POS2] == '接続助詞' \
            #         or morph[MCB_POS2] == '副詞化' \
            #         or morph[MCB_POS2] == '並立助詞' \
            #         or morph[MCB_POS2] == '連体化':
            if morph[MCB_POS2] == '格助詞' \
                    or morph[MCB_POS2] == '係助詞' \
                    or morph[MCB_POS2] == '接続助詞' \
                    or morph[MCB_POS2] == '副詞化' \
                    or morph[MCB_POS2] == '並立助詞':

                # ガ格
                if morph[MCB_FORM] == 'が':
                    chunk.case.append('ガ格')
                    chunk.argment.append('')
                    if len(chunk.argment) >= 2:
                        chunk.argment[-2] = 'ガ格'
                # ニ格
                elif morph[MCB_FORM] == 'に':
                    chunk.case.append('ニ格')
                    chunk.argment.append('')
                    if len(chunk.argment) >= 2:
                        chunk.argment[-2] = 'ニ格'
                # ヲ格
                elif morph[MCB_FORM] == 'を':
                    chunk.case.append('ヲ格')
                    chunk.argment.append('')
                    if len(chunk.argment) >= 2:
                        chunk.argment[-2] = 'ヲ格'
                # # デ格
                elif morph[MCB_FORM] == 'で':
                    chunk.case.append('デ格')
                    chunk.argment.append('')
                    if len(chunk.argment) >= 2:
                        chunk.argment[-2] = 'デ格'
                # # 主題(ハ格)
                elif morph[MCB_FORM] == 'は':
                    chunk.case.append('ハ格')
                    chunk.argment.append('')
                    if len(chunk.argment) >= 2:
                        chunk.argment[-2] = 'ハ格'
                # # 連体化「の」
                # elif morph[MCB_FORM] == 'の' and morph[MCB_POS2] == '連体化' and len(chunk.pos1) >= 2 and chunk.pos1[-2] == '名詞' and chunk.pos2[-2] != '代名詞':
                #     chunk.case.append('連体化「の」')
                #     chunk.argment.append('')
                #     chunk.argment[-2] = '連体化「の」'

                else:
                    chunk.case.append('')
                    chunk.argment.append('')

            else:
                chunk.case.append('')
                chunk.argment.append('')

            # 述語
            if morph[MCB_POS1] == '動詞' or \
                morph[MCB_POS1] == '形容詞' or \
                morph[MCB_POS1] == '形容動詞' or \
                morph[MCB_POS2] == 'サ変接続':
                chunk.predicate.append('述語')
            else:
                chunk.predicate.append('')

    if len(chunks) > 0:
        ret.append(chunks)

    return ret


def merge_chunk(chunk, edge=False):
    forms = []
    # bases = []
    cases = []

    # ノードラベルは,助詞以外の品詞と述語
    if not edge:
        i = 0
        while True:
            if i >= len(chunk.form):
                break

            # 名詞の連続は連結する (最大 N個まで連結)
            N = 5
            if chunk.pos1[i] == '名詞':
                word = chunk.form[i]
                i += 1
                k = 1
                for j in range(i, len(chunk.form)):
                    if k >= N:
                        break
                    if chunk.pos1[j] == '名詞':
                        word += chunk.form[j]
                        i += 1
                        k += 1
                    else:
                        break

                forms.append(word)
                continue

            # 動詞,サ変名詞は終止形にして後続をスキップ
            if chunk.pos1[i] == '動詞' or chunk.pos2[i] == 'サ変接続':
                # forms.append(chunk.form[i])
                # i += 1
                # continue
                forms.append(chunk.base[i])
                break

            # # 格助詞および連体形「の」は除外する
            # if chunk.case[i] != 'ガ格' and \
            #     chunk.case[i] != 'ニ格' and \
            #     chunk.case[i] != 'ヲ格' and \
            #     chunk.case[i] != '連体化「の」':
            #     forms.append(chunk.form[i])
            if chunk.case[i] == '':
                forms.append(chunk.form[i])
            i += 1

        return '-'.join(forms)

    # エッジラベルは,格助詞
    else:
        for i in range(len(chunk.case)):
            cases.append(chunk.case[i])
        label = ''.join(cases)
        return ' ' if label == '' else label


# ノードラベルは,助詞以外の品詞と述語
def merge_node_label(chunk):
    label = []
    i = 0

    while True:
        if i >= len(chunk.form):
            break

        # 名詞の連続は連結する (最大 N個まで連結)
        N = 5
        if chunk.pos1[i] == '名詞':
            word = chunk.form[i]
            i += 1
            k = 1
            for j in range(i, len(chunk.form)):
                if k >= N:
                    break
                if chunk.pos1[j] == '名詞':
                    word += chunk.form[j]
                    i += 1
                    k += 1
                else:
                    break

            label.append(word)
            continue

        # 動詞,サ変名詞は終止形にして後続をスキップ
        if chunk.pos1[i] == '動詞' or chunk.pos2[i] == 'サ変接続':
            if len(chunk.pos1) > i+1 and chunk.pos1[i+1] == '助動詞' and chunk.pos4[i+1] == '特殊・ナイ':
                label.append(chunk.form[i])
                label.append(chunk.base[i+1])
            else:
                label.append(chunk.base[i])
            break

        # 動詞,サ変名詞は終止形にして後続をスキップ
        if chunk.pos1[i] == '動詞' or chunk.pos2[i] == 'サ変接続':
            # label.append(chunk.form[i])
            # i += 1
            # continue
            label.append(chunk.base[i])
            break

        # # 格助詞および連体形「の」は除外する
        # if chunk.case[i] != 'ガ格' and \
        #     chunk.case[i] != 'ニ格' and \
        #     chunk.case[i] != 'ヲ格' and \
        #     chunk.case[i] != '連体化「の」':
        #     label.append(chunk.form[i])
        if chunk.case[i] == '':
            label.append(chunk.form[i])
        i += 1

    return '-'.join(label).strip()


# エッジラベルは,格助詞
def merge_edge_label(chunk):
    label = ''.join(chunk.case)
    return ' ' if label == '' else label


def draw_tree(chunks, d=0, s=0, p=0):

    import networkx as nx

    top = len(chunks)

    # グラフ解析する
    G = nx.DiGraph()

    for i in range(0, top):
        fr_node = chunks[i].fr
        to_node = chunks[i].to

        if not G.has_node(fr_node):
            label = merge_node_label(chunks[fr_node - 1])
            G.add_node(fr_node, label=label)

        if not to_node == 0:
            if not G.has_node(to_node):
                label = merge_node_label(chunks[to_node - 1])
                G.add_node(to_node, label=label)

            label = merge_edge_label(chunks[i])
            G.add_edge(fr_node, to_node, weight=label)

    # if logger.isEnabledFor(logging.DEBUG):
    #     logger.debug('--- dfs -------------')
    #     for e in list(nx.dfs_edges(G, top)):
    #         logger.debug(e)
    #
    #     logger.debug('--- bfs -------------')
    #     for e in list(nx.bfs_edges(G, top)):
    #         logger.debug(e)
    #
    #     logger.debug('--- dfs-r -----------')
    #     for e in reversed(list(nx.dfs_edges(G, top))):
    #         logger.debug(e)
    #
    #     logger.debug('--- bfs-r -----------')
    #     for e in reversed(list(nx.bfs_edges(G, top))):
    #         logger.debug(e)
    #
    #     logger.debug('--- iter ------------')
    #     # for e in list(G.edges_iter()):
    #     for e in list(G.edges):
    #         logger.debug(e)

    used_nodes = []
    # for left, right in list(G.edges_iter()):
    for left, right in list(G.edges()):
        edge = G[left][right]['weight']

        if edge == 'ガ格':
            used_nodes.extend([left, right])
        elif edge == 'ニ格':
            used_nodes.extend([left, right])
        elif edge == 'ヲ格':
            used_nodes.extend([left, right])
        elif edge == '連体化「の」':
            used_nodes.extend([left, right])

    # for left, right in list(G.edges_iter()):
    for left, right in list(G.edges()):
        edge = G[left][right]['weight']

        if edge == 'ガ格':
            used_nodes.extend([left, right])
        elif edge == 'ニ格':
            used_nodes.extend([left, right])
        elif edge == 'ヲ格':
            used_nodes.extend([left, right])
        elif edge == '連体化「の」':
            used_nodes.extend([left, right])
        else:
            if left not in used_nodes:
                used_nodes.extend([left])
            if right not in used_nodes or right == top:
                used_nodes.extend([right])

    # dot = nx.drawing.nx_pydot.to_pydot(G, strict='true')
    dot = nx.drawing.nx_pydot.to_pydot(G)
    dot.set_rankdir('LR')
    for e in dot.get_edge_list():
        e.set_label(e.get_attributes()['weight'])
    dot.set_label('{:}'.format(text))
    dot.write_png('graph_d{:03d}_s{:03d}-p{:03d}.png'.format(d+1, s+1, p+1), prog='dot')

    return


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('--input', default='', required=False, type=str, help='input file (.txt)')
    parser.add_argument('--text', default='', required=False, type=str, help='input text')
    args = parser.parse_args()

    documents = []

    if args.input:
        for line in open(args.input, 'r'):
            line = line.strip()
            documents.append(line)
    elif args.text:
            text = args.text.strip()
            documents.append(text)
    else:
        text = '彼は本屋で本を買った。私は太郎のように泳げない。'
        documents.append(text)

    for i, document in enumerate(documents):
        sentences = split(cleans(document))

        for j, text in enumerate(sentences):
            cb = parse_cabocha(text)

            logger.info('=====================')
            logger.info(text)
            logger.info('=====================')

            for k, chunks in enumerate(cb):
                logger.info('----- CaboCha   -----')
                for chunk in chunks:
                    logger.info('{}\t{}\t{}\t{}\t{}'.format(
                        chunk.fr if chunk.fr >= 0 else '',
                        '-'.join(chunk.form) if len(chunk.form) > 0 else '',
                        '-'.join(chunk.case) if len(chunk.case) > 0 else '',
                        '-'.join(chunk.predicate) if len(chunk.predicate) > 0 else '',
                        chunk.to if chunk.to >= 0 else '',
                    ))
                logger.info('---------------------')

                draw_tree(chunks, d=i, s=j, p=k)

    sys.stderr.write('time spent: {}\n'.format(time.time() - start_time))
