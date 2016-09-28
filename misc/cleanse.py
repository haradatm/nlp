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

import copy
import unicodedata

han_alpha_lower = u'abcdefghijklmnopqrstuvwxyz'
han_alpha_upper = u'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
han_number      = u'0123456789'
han_kigou       = u'!\"#$%&\'()=~|`{+*}<>?_-‐^\\@[;:],./'
han_blank       = u' '
zen_alpha_lower = u'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
zen_alpha_upper = u'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
zen_number      = u'０１２３４５６７８９'
zen_kigou       = u'！”＃＄％＆’（）＝〜｜｀｛＋＊｝＜＞？＿−−＾￥＠［；：］，．／'
zen_blank       = u'　'

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
    text = re.sub(ur'(ｈｔｔｐｓ?|ｆｔｐ)(：／／[^　\\S]+)', '', text)

    # パス文字列を除去する
    text = re.sub(ur'([ａ-ｚＡ-Ｚ]：)*￥[^　\\S]*', '', text)

    # <br> を除去する
    text = re.sub(ur'＜[Ｂｂ][Ｒｒ]＞', '　', text)

    # RT 除去する
    text = re.sub(ur'^ＲＴ[　\\S]*＠[^　\\S]+', '', text)

    # RT 文字列を除去する (あれば)
    text = re.sub(ur'ＲＴ[　\\S]*＠[^　\\S]+', '', text)

    # アカウント名を除去する
    text = re.sub(ur'＠[^　\\S]+', '', text)

    # ハッシュ文字列を除去する
    text = re.sub(ur'＃[^　\\S]+', '', text)

    # タグ文字のサニタイズを復元する
    text = re.sub(ur'＆ｌｔ；', '＜', text)
    text = re.sub(ur'＆ｇｔ；', '＞', text)
    text = re.sub(ur'＆ａｍｐ；', '＆', text)

    # 連続する長音を1つにする
    text = re.sub(ur'ー+', 'ー', text)

    # トリミングする
    text = text.strip()

    return text


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('--input', type=unicode, default='', help='input file (.txt)')
    parser.add_argument('--text', type=unicode, default='', help='input text')
    args = parser.parse_args()

    tokens = []

    if args.input:
        for line in open(args.input, 'rU'):
            text = unicode(line).strip()
            if text.startswith(u'#'):
                continue
            print(u'{}'.format(cleanse(text)))
            sys.stdout.flush()

    else:
        if args.text:
            text = unicode(args.text).strip()
            print(u'{}'.format(cleanse(text)))
        else:
            text = u'これは漱石が書いた本です。'
            # text = u'望遠鏡で泳ぐ、少女の姿を見た。'
            # text = u'少女の姿を見た。'

            print(u'{}'.format(cleans(text)))

    sys.stderr.write('time spent: {}\n'.format(time.time() - start_time))
