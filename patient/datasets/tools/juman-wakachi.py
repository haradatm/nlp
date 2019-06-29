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


from pyknp import Juman
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
    text = re.sub('<tab>', ' ', text)
    text = re.sub('<br>', ' ', text)

    # 半角カナ→全角カナ変換する
    text = normalize(text)

    # 連続する長音を1つにする
    text = re.sub(r'ー+', 'ー', text)

    # トリミングする
    text = text.strip()

    return text


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('file', type=str, help='input file (.txt)')
    args = parser.parse_args()

    juman = Juman(jumanpp=True)

    for line in open(args.file, 'r'):
        line = line.strip()
        if line == '':
            print()
            sys.stdout.flush()
            continue

        text = cleans(line)
        result = juman.analysis(text)
        morphs = [m.midasi for m in result.mrph_list()]

        if len(morphs) > 0:
            print(" ".join(morphs))
        else:
            print()
        sys.stdout.flush()

sys.stderr.write('time spent: {}\n'.format(time.time() - start_time))
