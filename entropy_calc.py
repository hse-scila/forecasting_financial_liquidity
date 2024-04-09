import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm.notebook import tqdm
from functools import reduce
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp


def get_dates_interfax(tickers, messages, time_window):
    dates_start = []
    l = []
    for date in (
        messages["date"][messages["date"].isin(tickers["date"])].sort_values().unique()
    ):
        l.append(date + pd.Timedelta(days=time_window))
        dates_start.append(date)
    dates = list(zip(dates_start, l))
    return dates


def get_dates(df, time_window):
    dates_start = []
    l = []
    for date in df["date"].unique():
        l.append(date + pd.Timedelta(days=time_window))
        dates_start.append(date)
    dates = list(zip(dates_start, l))
    return dates


def top_changed_words(count_df):
    count_words_df = count_df.copy()
    count_words_df = count_words_df.fillna(0)

    pct_change_words = count_words_df.pct_change(axis=1)
    pct_change_words = pct_change_words.fillna(0)
    pct_change_words = pct_change_words.replace([np.inf, -np.inf], np.nan)

    d_max = {}
    d_max_words = {}
    for col in pct_change_words.columns[1:]:
        d_max[col] = ", ".join(
            pct_change_words.loc[
                pct_change_words[col] == pct_change_words[col].max()
            ].T.columns
        )
        d_max_words[col] = ", ".join(
            pct_change_words.loc[pct_change_words[col] == pct_change_words[col].max()]
            .T.loc[col]
            .values.astype(str)
        )

    d_min = {}
    d_min_words = {}
    for col in pct_change_words.columns[1:]:
        d_min[col] = ", ".join(
            pct_change_words.loc[
                pct_change_words[col] == pct_change_words[col].min()
            ].T.columns
        )
        d_min_words[col] = ", ".join(
            pct_change_words.loc[pct_change_words[col] == pct_change_words[col].min()]
            .T.loc[col]
            .values.astype(str)
        )

    top_words_max = pd.DataFrame.from_dict(d_max, orient="index").rename(
        {0: "words"}, axis=1
    )
    top_words_min = pd.DataFrame.from_dict(d_min, orient="index").rename(
        {0: "words"}, axis=1
    )

    top_values_max = pd.DataFrame.from_dict(d_max_words, orient="index").rename(
        {0: "value"}, axis=1
    )
    top_values_min = pd.DataFrame.from_dict(d_min_words, orient="index").rename(
        {0: "value"}, axis=1
    )

    top_words_max = pd.concat([top_words_max, top_values_max], axis=1)
    top_words_max["words"] = top_words_max["words"].apply(lambda x: x.split(", "))
    top_words_max["value"] = top_words_max["value"].apply(lambda x: x.split(", "))
    top_words_max = top_words_max[["words", "value"]].apply(pd.Series.explode)
    top_words_max["value"] = top_words_max["value"].astype(float)
    top_words_max = top_words_max[top_words_max["value"] != 0]

    top_words_min = pd.concat([top_words_min, top_values_min], axis=1)
    top_words_min["words"] = top_words_min["words"].apply(lambda x: x.split(", "))
    top_words_min["value"] = top_words_min["value"].apply(lambda x: x.split(", "))
    top_words_min = top_words_min[["words", "value"]].apply(pd.Series.explode)
    top_words_min["value"] = top_words_min["value"].astype(float)
    top_words_min = top_words_min[top_words_min["value"] != 0]
    return top_words_max, top_words_min


def ewm_tfidf(tf_df, idf_df, span):
    tf_all_words_ewm = tf_df.T.apply(lambda x: x.ewm(span=span).mean()).T
    idf_all_words_ewm = idf_df.T.apply(lambda x: x.ewm(span=span).mean()).T
    tf_idf_ewm = tf_all_words_ewm * idf_all_words_ewm
    tf_idf_ewm = tf_idf_ewm / tf_idf_ewm.sum()
    tfidf_diversity_ewm = tf_idf_ewm.apply(lambda x: -x * np.log(x)).sum()
    tfidf_diversity_ewm.name = "tfidf_diversity_words_ewm"
    tfidf_diversity_ewm.index = pd.to_datetime(tfidf_diversity_ewm.index)
    return tfidf_diversity_ewm


def source_entropy(tickers, messages, tf_df, time_window=30):
    dates = get_dates(tickers, messages, time_window=time_window)

    for i in tqdm(range(len(dates))):
        tmp = messages[
            (messages["date"] >= dates[i][0]) & (messages["date"] <= dates[i][1])
        ]

        tf_on_date = tf_df[dates[0][1].strftime("%Y-%m-%d")]
        tf_on_date_not_na = tf_on_date[tf_on_date.notna()]
        words = tf_on_date_not_na.index.tolist()

        words_dict = {}
        for word in words:
            sources_present = tmp[
                pd.DataFrame(tmp["filter_ngram"].tolist()).isin([word]).any(1).values
            ]["source"]
            sources_count = sources_present.value_counts(normalize=True)
            word_entropy_source = sources_count.apply(lambda x: -x * np.log(x)).sum()
            words_dict[word] = word_entropy_source

        entropy_sources = pd.Series(entropy_dict)
        entropy_sources.index = pd.to_datetime(entropy_sources.index)
        return entropy_sources


def get_words_diversity(df, dates):
    df_merged = pd.DataFrame(columns=["word"])

    for i in tqdm(range(len(dates))):
        tmp = df[(df["date"] >= dates[i][0]) & (df["date"] <= dates[i][1])]
        if len(tmp) == 0:
            continue
        try:
            vectorizer = CountVectorizer(min_df=0.05, max_df=0.95)
            X = vectorizer.fit_transform(
                tmp["filter_ngram"].apply(lambda x: ", ".join(x))
            )
        except ValueError:
            continue

        count_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
        count_df["message_id"] = tmp.index
        count_df = (
            count_df.drop("message_id", axis=1).sum().sort_values(ascending=False)
        )
        count_df = (
            pd.DataFrame(count_df)
            .reset_index()
            .rename(
                {0: f'{dates[i][1].date().strftime("%Y-%m-%d")}', "index": "word"},
                axis=1,
            )
        )
        df_merged = df_merged.merge(count_df, on="word", how="outer")

    df_merged = df_merged.set_index("word").fillna(0)

    pct_change_words = df_merged.pct_change(axis=1)
    pct_change_words = pct_change_words.fillna(0)
    pct_change_words = pct_change_words.replace([np.inf, -np.inf], np.nan)

    diversity_dict = {}
    for col in pct_change_words.columns:
        pct_change_on_date = pct_change_words[col][pct_change_words[col] > 0]
        pct_change_on_date = pct_change_on_date / pct_change_on_date.sum()
        diversity = pct_change_on_date.apply(lambda x: x * np.log(x)).sum() * (-1)
        diversity_dict[col] = diversity

    diversity_df = pd.DataFrame.from_dict(diversity_dict, orient="index").rename(
        {0: "diversity_words"}, axis=1
    )
    diversity_df.index = pd.to_datetime(diversity_df.index)
    return diversity_df


def get_tf_idf_data(data, dates):
    tf_list = []
    idf_list = []
    for i in tqdm(range(len(dates))):
        tmp = data[(data["date"] >= dates[i][0]) & (data["date"] <= dates[i][1])]
        tmp = tmp[tmp["filter_ngram"].notna()]
        tmp["len"] = tmp["filter_ngram"].apply(lambda x: len(x))

        vectorizer = TfidfVectorizer(norm=None, min_df=0.05, max_df=0.95)
        X = vectorizer.fit_transform(tmp["filter_ngram"].apply(lambda x: ", ".join(x)))
        features = vectorizer.get_feature_names()
        n = len(features)

        inverse_idf = sp.diags(
            1 / vectorizer.idf_, offsets=0, shape=(n, n), format="csr", dtype=np.float64
        ).toarray()
        idf = sp.diags(
            vectorizer.idf_, offsets=0, shape=(n, n), format="csr", dtype=np.float64
        ).toarray()

        tfs = pd.DataFrame(X * inverse_idf, columns=features).div(
            tmp["len"].reset_index(drop=True), axis=0
        )
        tfs_mean = tfs.mean()
        tfs_mean = pd.DataFrame(tfs_mean).rename(
            {0: f'{dates[i][1].date().strftime("%Y-%m-%d")}'}, axis=1
        )
        # idfs = vectorizer.idf_.toarray()
        idfs = pd.DataFrame(idf, columns=features).sum()
        idfs = pd.DataFrame(idfs).rename(
            {0: f'{dates[i][1].date().strftime("%Y-%m-%d")}'}, axis=1
        )

        tf_list.append(tfs_mean)
        idf_list.append(idfs)

    tf_all_words = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="outer"
        ),
        tf_list,
    )
    idf_all_words = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="outer"
        ),
        idf_list,
    )

    tf_all_words = tf_all_words.fillna(0)
    idf_all_words = idf_all_words.fillna(0)
    return tf_all_words, idf_all_words


def get_topics_diversity(ldamodel, corpus, df, source):
    get_document_topics = [ldamodel.get_document_topics(item) for item in corpus]
    docs_dict = {}
    for i in tqdm(range(len(get_document_topics))):
        tmp = pd.DataFrame([e[1] for e in get_document_topics[i]]).T
        tmp.columns = ["topic_" + str(e[0]) for e in get_document_topics[i]]
        tmp = tmp.reset_index()
        if source == "interfax":
            tmp["index"] = df.index[i]
        else:
            tmp["index"] = i
        docs_dict[i] = tmp

    topics_df = pd.concat(docs_dict)
    topics_df["date"] = messages["date"].values
    topics_df = topics_df.fillna(0)

    cols = ["date"]
    cols += [f"topic_" + str(i) for i in range(0, 10)]

    topics_df = topics_df[cols].set_index("date")

    res = topics_df.resample("1D")
    day_topics = res.apply(lambda rows: rows.sum(axis=0) / max(rows.shape[0], 1))
    sel = day_topics.sum(axis=1) > 0
    X = day_topics[sel]
    diversity = -np.sum(X.values * np.log(X.values), axis=1)
    diversity = pd.Series(diversity, index=sel[sel.values].index)
    diversity.name = "diversity"
    div_df = diversity.to_frame()
    div_df["nmess"] = res.size()
    div_df.index.name = "date"
    return div_df
