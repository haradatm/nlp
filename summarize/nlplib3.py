#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#
# usege: python3.6
#

__version__ = '0.0.1'

import sys, time, logging
import numpy as np
np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def pp(obj):
    import pprint
    pp = pprint.PrettyPrinter(indent=1, width=160)
    print(pp.pformat(obj))


start_time = time.time()


import re, copy
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


def cleanse(text):

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
    # text = re.sub(r'＜[Ｂｂ][Ｒｒ]＞', '　', text)

    # RT 除去する
    # text = re.sub(r'^ＲＴ[　\\S]*＠[^　\\S]+', '', text)

    # RT 文字列を除去する (あれば)
    # text = re.sub(r'ＲＴ[　\\S]*＠[^　\\S]+', '', text)

    # アカウント名を除去する
    # text = re.sub(r'＠[^　\\S]+', '', text)

    # ハッシュ文字列を除去する
    # text = re.sub(r'＃[^　\\S]+', '', text)

    # タグ文字のサニタイズを復元する
    text = re.sub(r'＆ｌｔ；', '＜', text)
    text = re.sub(r'＆ｇｔ；', '＞', text)
    text = re.sub(r'＆ａｍｐ；', '＆', text)

    # 連続する長音を1つにする
    text = re.sub(r'ー+', 'ー', text)

    # トリミングする
    text = text.strip()

    return text


import os, MeCab
CWD = os.getcwd()
tagger = MeCab.Tagger('-r ' + CWD + '/nlplib.conf/mecabrc')
tagger.parse('')


def ___mecab____(sent):
    ret = []
    # encoded = sent.encode('utf-8')
    # node = tagger.parseToNode(encoded)
    node = tagger.parseToNode(sent)
    while node:
        # features = [node.surface.decode('utf-8')]
        # features += re.split('[\s,]', (node.feature.decode('utf-8')).strip())
        features = [node.surface.strip()]
        features += re.split(',', node.feature.strip())
        ret.append(features)
        node = node.next
    return ret


# for CaboCha
import CaboCha
CWD = os.getcwd()
parser = CaboCha.Parser('-f1 -O4 -n0 -r ' + CWD + '/nlplib.conf/cabocharc -b ' + CWD + '/nlplib.conf/mecabrc')

def __cabocha__(sent):
    ret = []
    # encoded = sent.encode('utf-8')
    # tree = parser.parse(encoded)
    tree = parser.parse(sent)
    lines = tree.toString(CaboCha.FORMAT_LATTICE).split('\n')
    for line in lines:
        line = line.strip()
        if line:
            arr = re.split('[\s,]', line)
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


class Morph():
    def __init__(self):
        self.id = None
        self.form = None
        self.read = None
        self.base = None
        self.pos1 = None
        self.pos2 = None
        self.pos3 = None
        self.case = None
        self.pa = None


def parse_mecab(str):
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

        if len(node) < (MCB_READ+1) or node[MCB_READ] == '*':
            morph.read = node[MCB_FORM]
        else:
            morph.read = node[MCB_READ]

        if node[MCB_BASE] == '*':
            morph.base = node[MCB_FORM]
        else:
            morph.base = node[MCB_BASE]

        morph.pos1 = node[MCB_POS1]
        morph.pos2 = node[MCB_POS2]
        morph.pos3 = node[MCB_POS3]
        morph.pos4 = node[MCB_POS4]

        # 補正
        if morph.read == '*':
            morph.read = morph.form

        if morph.base == '*':
            morph.base = morph.form

        morph.case = ''
        morph.pa = ''

        if morph.pos2 == '格助詞' or morph.pos2 == '副助詞':
            if len(morphs) > 0 and morphs[-1].pos1 == '名詞':
                # 主題(ハ格)
                if morph.form == 'は':
                    morphs[-1].case = 'ハ格'
                # ガ格
                elif morph.form == 'が':
                    morphs[-1].case = 'ガ格'
                # ニ格
                elif morph.form == 'に':
                    morphs[-1].case = 'ニ格'
                # ヲ格
                elif morph.form == 'を':
                    morphs[-1].case = 'ヲ格'
                # デ格
                elif morph.form == 'で':
                    morphs[-1].case = 'デ格'
        morphs.append(morph)

    if len(morphs) > 0:
        ret.append(morphs)

    return ret


class Chunk():
    def __init__(self):
        self.fr = -1
        self.to = -1
        self.morphs = []


def parse_cabocha(str):
    lines = __cabocha__(str)
    chunks = []
    ret = []

    chunk = Chunk()
    chunk_id = 1
    morph_id = 1

    for i, arr in enumerate(lines):
        # 文節処理
        if arr[MCB_FORM] == '#' or \
           arr[MCB_FORM] == '*' or \
           arr[MCB_FORM] == 'EOS':

            if i > 0:
                # 文節情報
                chunks.append(chunk)
                chunk = Chunk()
                chunk_id += 1

                if arr[MCB_FORM] == 'EOS':
                    ret.append(chunks)
                    chunks = []
                    morph_id = 1

            if arr[MCB_FORM] == '*':
                # 文節番号
                chunk.fr = chunk_id
                chunk.to = int(arr[2][:-1])+1
        else:
            # 未知語 ('-x 未知語')
            if len(arr) == 2:
                arr.append(u'*')  # (2) MCB_POS2
                arr.append(u'*')  # (3) MCB_POS3
                arr.append(u'*')  # (4)
                arr.append(u'*')  # (5) MCB_POS4
                arr.append(u'*')  # (6)
                arr.append(u'*')  # (7) MCB_BASE
                arr.append(u'*')  # (8) MCB_READ

            morph = Morph()
            morph.id = morph_id
            morph_id += 1

            # 文節 (表層,読み,品詞,その他の素性)
            morph.form = arr[MCB_FORM]

            if len(arr) < (MCB_READ+1) or arr[MCB_READ] == '*':
                morph.read = arr[MCB_FORM]
            else:
                morph.read = arr[MCB_READ]

            if arr[MCB_BASE] == '*':
                morph.base = arr[MCB_FORM]
            else:
                morph.base = arr[MCB_BASE]

            morph.pos1 = arr[MCB_POS1]
            morph.pos2 = arr[MCB_POS2]
            morph.pos3 = arr[MCB_POS3]
            morph.pos4 = arr[MCB_POS4]

            morph.case = ''
            morph.pa = ''

            # 格, 項
            if arr[MCB_POS2] == '格助詞' \
                    or arr[MCB_POS2] == '係助詞' \
                    or arr[MCB_POS2] == '接続助詞' \
                    or arr[MCB_POS2] == '副詞化' \
                    or arr[MCB_POS2] == '並立助詞' \
                    or arr[MCB_POS2] == '連体化':

                # ガ格
                if arr[MCB_FORM] == 'が':
                    morph.pa = 'ガ格'
                # ニ格
                elif arr[MCB_FORM] == 'に':
                    morph.pa = 'ニ格'
                # ヲ格
                elif arr[MCB_FORM] == 'を':
                    morph.pa = 'ヲ格'
                # デ格
                elif arr[MCB_FORM] == 'で':
                    morph.pa = 'デ格'
                # ハ格
                # elif arr[MCB_FORM] == 'は':
                #     morph.case = 'ハ格'

                # 連体化「の」
                elif arr[MCB_FORM] == 'の' and arr[MCB_POS2] == '連体化':
                    if len(chunk.morphs) > 0:
                        prev = chunk.morphs[-1]
                        if prev.pos1 == '名詞' and prev.pos2 != '代名詞':
                            morph.pa = '連体化「の」'
                        else:
                            morph.pa = ''
                    else:
                        morph.pa = ''
                else:
                    morph.pa = ''
            # 述語
            elif arr[MCB_POS1] == '動詞' \
                    or arr[MCB_POS1] == '形容詞' \
                    or arr[MCB_POS1] == '形容動詞' \
                    or arr[MCB_POS2] == 'サ変接続':

                morph.pa = '述語'

            else:
                morph.pa = ''

            chunk.morphs.append(morph)

    if len(chunks) > 0:
        ret.append(chunks)

    return ret


def word_segmenter(text, plus, minus, type='base'):
    sents = parse_mecab(text)

    selected = []
    for morphs in sents:
        for morph in morphs:

            if morph.form == '':
                pass
            elif re.search(r'^(接尾|非自立)', morph.pos2):
                pass
            elif u'サ変・スル' == morph.pos4 or u'ある' == morph.base:
                pass

            else:
                b = False
                if plus == '*':
                    b = True
                else:
                    for flt in plus.split(u'|'):
                        if u'-' in flt:
                            pos1, pos2 = flt.split(u'-')
                            if morph.pos1 == pos1 and morph.pos2 == pos2:
                                b = True
                                break
                        else:
                            pos1 = flt
                            if morph.pos1 == pos1:
                                b = True
                                break

                if minus != '':
                    for flt in minus.split(u'|'):
                        if u'-' in flt:
                            pos1, pos2 = flt.split(u'-')
                            if morph.pos1 == pos1 and morph.pos2 == pos2:
                                b = False
                                break
                        else:
                            pos1 = flt
                            if morph.pos1 == pos1:
                                b = False
                                break
                if b:
                    selected.append(morph)

    words = []
    for morph in selected:
        if type == 'form':
            word = morph.form if morph.form != '' else morph.base
        else:
            word = morph.base if morph.base != '' else morph.form
        words.append(word)

    return words


def sent_splitter(text):
    parenthesis = '（）「」『』【】［］〈〉《》〔〕｛｝””'
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


if __name__ == '__main__':
    text = '望遠鏡で泳ぐ少女の姿を見た。\n望遠鏡で泳ぐ少女の姿を見た。'
    # text = '愛しい人よは歌のタイトルです。'
    # text = '日食きたーーーーーーー！！！！'
    # text = '行動する'

    # print '----- JUMAN   -----'
    # jm = juman(text)
    # mps = morph_juman(jm)
    # for mp in mps:
    #     pp(mp)
    #
    # print '----- KNP     -----'
    # kp = knp(text)
    # phs = phrase_knp(kp)
    # for ph in phs:
    #     pp(ph)

    print('----- MeCab   -----')
    sents = parse_mecab(text)
    for morphs in sents:
        for morph in morphs:
            pp(vars(morph))

    print('----- CaboCha   -----')
    sents = parse_cabocha(text)
    for chunks in sents:
        for chunk in chunks:
            for morph in chunk.morphs:
                pp(vars(morph))
