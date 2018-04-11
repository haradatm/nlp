#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 実験: 
    結果: 
"""

__version__ = '0.0.1'

import sys, time, logging
import numpy as np
np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def pp(obj):
    import pprint
    pp = pprint.PrettyPrinter(indent=1, width=160)
    print(pp.pformat(obj))


start_time = time.time()


import re
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
    # text = re.sub(r'(ｈｔｔｐｓ?|ｆｔｐ)(：／／[^　\\S]+)', '', text)

    # パス文字列を除去する
    # text = re.sub(r'([ａ-ｚＡ-Ｚ]：)*￥[^　\\S]*', '', text)

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


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('--input', type=str, default='', help='input file (.txt)')
    parser.add_argument('--text', type=str, default='', help='input text')
    args = parser.parse_args()

    tokens = []

    if args.input:
        for line in open(args.input, 'r'):
            text = line.strip()
            if text.startswith('#'):
                continue
            print('{}'.format(cleanse(text)))
            sys.stdout.flush()

    else:
        if args.text:
            text = args.text.strip()
            print('{}'.format(cleanse(text)))
        else:
            text = 'これは漱石が書いた本です。'
            # text = '望遠鏡で泳ぐ、少女の姿を見た。'

            print('{}'.format(cleanse(text)))

    logger.info('time spent: {}\n'.format(time.time() - start_time))
