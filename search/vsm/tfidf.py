#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk

token_pattern = re.compile(r"(?u)\b\w\w+\b")
lemmatizer = nltk.WordNetLemmatizer()


def stopwords():
    symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')', '*', '--', '\\']
    stopwords = nltk.corpus.stopwords.words('english')
    return stopwords + symbols


def analyzer(text, stopwords=[]):
    words = []
    for word in token_pattern.findall(text):
        word = lemmatizer.lemmatize(word.lower())
        if word not in stopwords:
            words.append(word)
    return words


class Vectorizer(TfidfVectorizer):
    def __init__(self, **params):
        super().__init__(
            analyzer=analyzer,
            binary=False,
            stop_words=stopwords(),
            ngram_range=(1, 3),
            min_df=1,
            max_df=1.0,
            smooth_idf=True,
            use_idf=True,
            sublinear_tf=False
        )
