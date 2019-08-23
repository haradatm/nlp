#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, time, logging, re, json
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

# usage:

import requests
from bs4 import BeautifulSoup
import traceback


def replace_string(text):
    text = re.sub(r'\t',   "<tab>", text)
    text = re.sub(r'\r\n', "<br>",  text)
    text = re.sub(r'[\r\n]', "<tab>", text)
    return text


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='%prog url [option]')
    parser.add_argument('--url', type=str, default='http://carinf.mlit.go.jp/jidosha/carinf/opn/search.html?selCarTp=1&lstCarNo=000&txtFrDat=1000/01/01&txtToDat=9999/12/31&txtNamNm=&txtMdlNm=&txtEgmNm=&chkDevCd=&page=', help='url')
    parser.add_argument('--start', '-y', type=int, default=1,    help='start page')
    parser.add_argument('--end',   '-m', type=int, default=5194, help='end page')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    for page in range(args.start, args.end + 1):
        target_url = '{:}{}'.format(args.url, page)
        sys.stderr.write('### {}\n'.format(target_url))
        sys.stderr.flush()

        soup = BeautifulSoup(requests.get(target_url).text, 'html.parser')
        tags = soup.select('.tablecar tbody tr')

        for tag in tags:
            tds = tag.select('td')
            print(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                replace_string(tds[0].get_text().strip()),
                replace_string(tds[1].select('.uctop')[0].get_text().strip()),
                replace_string(tds[1].select('.umidleft')[0].get_text().strip()),
                replace_string(tds[1].select('.umidright')[0].get_text().strip()),
                replace_string(tds[1].select('.ubottom')[0].get_text().strip()),
                replace_string(tds[2].select('.ftopleft')[0].get_text().strip()),
                replace_string(tds[2].select('.ftopright')[0].get_text().strip()),
                replace_string(tds[2].select('.fmidleft')[0].get_text().strip()),
                replace_string(tds[2].select('.fmidright')[0].get_text().strip()),
                replace_string(tds[2].select('.fbottomleft')[0].get_text().strip()),
                replace_string(tds[2].select('.fbottomright')[0].get_text().strip()),
                replace_string(tds[3].select('.ctopleft')[0].get_text().strip()),
                replace_string(tds[3].select('.ctopright')[0].get_text().strip()),
                replace_string(tds[3].select('.cbottom')[0].get_text().strip())
            ))
            sys.stderr.flush()

    sys.stderr.write('time spent: {}\n'.format(time.time() - start_time))
