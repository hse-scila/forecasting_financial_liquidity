import nltk

# nltk.download("stopwords")

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from tqdm.notebook import tqdm

from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases
from gensim.models import Word2Vec
from nltk import word_tokenize

import warnings

warnings.filterwarnings("ignore")

import pickle
from copy import deepcopy
from collections import defaultdict

russian_stopwords = stopwords.words("russian")
punct = punctuation + "«»—" + "..." + "--" + "***"

import warnings

warnings.filterwarnings("ignore")

import pymorphy2

morph = pymorphy2.MorphAnalyzer()

russian_stopwords.append("год")
russian_stopwords.append("ещё")
russian_stopwords.append("это")


def initial_prepare(news):
    tokens_news = [nltk.word_tokenize(row["text"]) for i, row in news.iterrows()]

    # приведение к нижнему регистру
    tokens_news = [[token.lower() for token in new] for new in tokens_news]

    # удаление знаков пунктуации
    tokens_news = [
        [token for token in new if token not in punctuation] for new in tokens_news
    ]

    # удаление оставшихся токенов, которые не являются буквенными
    tokens_news = [[token for token in new if token.isalpha()] for new in tokens_news]

    # удаление стоп слов русского языка
    tokens_news = [
        [token for token in new if token not in russian_stopwords]
        for new in tokens_news
    ]

    # лемматизация
    mystem = Mystem()
    tokens_news = [
        [mystem.lemmatize(token)[0] for token in new] for new in tqdm(tokens_news)
    ]

    # повторное удаление стоп слов русского языка
    tokens_news = [
        [
            token
            for token in new
            if (token not in russian_stopwords) and (len(token) > 1)
        ]
        for new in tokens_news
    ]

    return tokens_news


def ngram_prepare(tokens, min_count, threshold):
    ngram = Phrases(tokens, min_count=min_count, threshold=threshold)
    ngram_tokens = ngram[tokens]

    return ngram_tokens


def filter_prepare(tokens, no_below, no_above):
    dct = Dictionary(tokens)
    dct.filter_extremes(no_below=no_below, no_above=no_above)
    cut_filter = set(dct.token2id.keys())
    filter_tokens = [[token for token in new if token in cut_filter] for new in tokens]

    return filter_tokens


def preproc_news(news):
    tokens_news = initial_prepare(news)
    bigram_tokens = ngram_prepare(tokens_news, min_count=100, threshold=50)
    trigram_tokens = ngram_prepare(bigram_tokens, min_count=100, threshold=50)
    filter_tokens = filter_prepare(tokens_news, no_below=10, no_above=0.2)
    filter_ngram = filter_prepare(trigram_tokens, no_below=10, no_above=0.2)

    news["tokens"] = tokens_news
    news["ngram"] = trigram_tokens
    news["filter_tokens"] = filter_tokens
    news["filter_ngram"] = filter_ngram

    return news
