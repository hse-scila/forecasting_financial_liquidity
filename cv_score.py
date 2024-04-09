import numpy as np
import pandas as pd

from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel, Phrases
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_texts
from tqdm import tqdm
import nltk

from src.data.text import TokenizedText
from src.data.moex import load_messages
from src.files import get_project_dir
from src.processing import select_equally_distributed
from gensim.models.phrases import Phraser


def load_content(stop_words, lang="russian"):
    if stop_words is None:
        stop_words_ = []
    else:
        stop_words_ = list(stop_words)
    text = TokenizedText.load(get_project_dir() / "data/tokens/text/", True)
    mapping = text.tok.mapping
    stop_words = set(nltk.corpus.stopwords.words(lang))
    stop_words.update(stop_words_)
    stop_words = list(stop_words.intersection(mapping.index))
    is_stopword = pd.Series(False, index=mapping.index)
    is_stopword.loc[stop_words] = True
    is_ordinary_word = np.zeros(mapping.max() + 1, dtype=bool)
    is_ordinary_word[mapping.values] = ~is_stopword.values

    def to_text_arr(x):
        if x is None:
            return None

        mask = is_ordinary_word[x]
        return text.tok.map_tokens(x[mask])

    messages = load_messages()
    content = text.tok.content
    content = content.reindex(messages.index).dropna()
    corpus = content.apply(to_text_arr)
    index = content.dropna().index
    datetime = messages.datetime.loc[index]
    return corpus, datetime


def model_ngrams(docs, n=3):
    for _ in range(n - 1):
        model = Phrases(
            docs, min_count=5, threshold=100
        )  # higher threshold fewer phrases.
        docs = model[docs]

    return docs


def transform_corpus_to_bow(corpus, min_count, max_df, n=3, tqdm=tqdm):
    # corpus = corpus.dropna()
    if n > 1:
        corpus = model_ngrams(corpus, n=n)

    id2word = Dictionary(corpus)

    id2word.filter_extremes(min_count, max_df)
    x = [id2word.doc2bow(doc) if doc is not None else None for doc in tqdm(corpus)]
    corpus_vec = pd.Series(x, index=corpus.index)
    return corpus_vec, id2word


def split_equally_in_time(data, test_size, train_size):
    in_train = pd.Series(False, data.index)
    train_sample = select_equally_distributed(data, "1w", train_size)
    in_train.loc[train_sample.index] = True
    test_sample = data[~in_train.values]
    if train_size + test_size == 1.0:
        return train_sample, test_sample
    part = test_size / (1 - train_size)
    test_sample = select_equally_distributed(data, "1w", part)
    return train_sample, test_sample


class Corpus:
    def __init__(
        self, stop_words=None, min_count=50, max_df=0.2, lang="russian", tqdm=tqdm
    ):
        corpus, datetime = load_content(stop_words, lang)
        corpus_vec, id2word = transform_corpus_to_bow(corpus, min_count, max_df, tqdm)

        self.dictionary = id2word
        self.data = pd.concat([corpus, corpus_vec, datetime], axis=1)
        self.data.columns = ["tokens", "bow", "datetime"]

    def split(self, train_size=0.8, test_size=0.2):
        return split_equally_in_time(self.data, test_size, train_size)


def get_coherence_score(lda_model, dictionary, data):
    texts = data.tokens.values
    coh_model = CoherenceModel(
        model=lda_model, texts=texts, dictionary=dictionary, coherence="c_v"
    )
    coh_value = coh_model.get_coherence()
    return coh_value
