import warnings
from typing import Union, Callable

import numpy as np
import pandas as pd
import sklearn.base
from sklearn import clone
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import clone

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils.multiclass import type_of_target


def get_metrics(y_true, y_pred, **kwargs):
    metrics = get_clf_metrics(y_true, y_pred, clf_threshold, **kwargs)
    return metrics


def get_metrics_by(data, true_col, pred_col, by, **kwargs):
    records = []
    data = data.dropna(subset=[true_col, pred_col])
    for keys, rows in data.groupby(by):
        y_true = rows[true_col]
        y_pred = rows[pred_col]
        metrics = get_metrics(y_true, y_pred, **kwargs)
        if isinstance(by, str):
            metrics[by] = keys
        else:
            metrics.update(dict(zip(by, keys)))

        records.append(metrics)

    return pd.DataFrame(records)


def get_clf_metrics(y_true, y_pred_proba, threshold=0.5):
    return {
        "AUC ROC": roc_auc_score(y_true, y_pred_proba)}


def get_cv_results(
    model: sklearn.base.BaseEstimator,
    data: Union[pd.DataFrame, np.ndarray],
    feature_names,
    target: Union[str, np.array],
    n_folds: int = 5,
):
    y = target
    if isinstance(target, str):
        y = data[target]
    X = data
    if feature_names:
        X = data[feature_names]

    k_fold = StratifiedKFold(n_folds)
    results = []
    try:
        index = data.index
    except AttributeError:
        index = np.arange(len(data))

    for (
        train_index,
        test_index,
    ) in k_fold.split(data, target):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        y_pred = pd.Series(y_pred, index=index[test_index])
        results.append(y_pred)

    y_pred = pd.concat(results)
    y_pred = y_pred.set_index()

    return y_pred


def _select_items(data, index_series):
    if len(index_series) == 0:
        return None
    val_index = np.concatenate(index_series.values)
    return data.loc[val_index]


def time_series_split(
    data,
    split_size,
    datetime_col="datetime",
    train_size=4,
    val_size=1,
    test_size=0,
    expanding=True,
    **kwargs,
):
    assert data.index.is_unique

    groups = pd.Series(
        {
            i: rows.index
            for i, (date, rows) in enumerate(
                data.resample(on=datetime_col, rule=split_size, **kwargs)
            )
        }
    )

    group_ids = np.arange(len(groups))
    size = train_size + val_size + test_size
    if size > len(group_ids):
        mess = "Not enough data to perform cross-validation"
        if "ticker" in data:
            ticker = data.ticker.iloc[0]
            mess += f" for {ticker}"
        warnings.warn(mess)
        return
    view = sliding_window_view(group_ids, size)
    for ids in view:
        train_idx = ids[:train_size]
        val_ids = ids[train_size : train_size + val_size]
        test_ids = ids[train_size + val_size :]
        if expanding:
            train_index = groups[: train_idx[-1] + 1]
        else:
            train_index = groups.iloc[train_idx]

        # train_index = np.concatenate(train_index.values)
        test_data = _select_items(data, groups.iloc[test_ids])
        train_data = _select_items(data, train_index)
        val_data = _select_items(data, groups.iloc[val_ids])
        if test_data is None:
            yield train_data, val_data
        else:
            yield train_data, val_data, test_data


def get_cv_index(
    data,
    split_size,
    datetime_col="datetime",
    train_size=4,
    val_size=1,
    test_size=0,
    expanding=True,
    **kwargs,
):
    assert data.index.is_unique

    groups = pd.Series(
        {
            i: rows.index
            for i, (date, rows) in enumerate(
                data.resample(on=datetime_col, rule=split_size, **kwargs)
            )
        }
    )

    group_ids = np.arange(len(groups))
    size = train_size + val_size + test_size
    if size > len(group_ids):
        mess = "Not enough data to perform cross-validation"
        if "ticker" in data:
            ticker = data.ticker.iloc[0]
            mess += f" for {ticker}"
        warnings.warn(mess)
        return
    view = sliding_window_view(group_ids, size)
    train_ind = []
    val_ind = []

    for ids in view:
        train_idx = ids[:train_size]
        train_ind.append(train_idx)
        val_ids = ids[train_size : train_size + val_size]
        val_ind.append(val_ids)
        test_ids = ids[train_size + val_size :]
        if expanding:
            train_index = groups[: train_idx[-1] + 1]
        else:
            train_index = groups.iloc[train_idx]

    cv = list(zip(train_ind, val_ind))
    train_inds = [np.hstack(groups.iloc[cv[i][0]]) for i in range(len(cv))]
    val_inds = [np.hstack(groups.iloc[cv[i][1]]) for i in range(len(cv))]
    cv = list(zip(train_inds, val_inds))

    return cv


def _add_predicts(train_data, model, feature_names):
    X = train_data[feature_names]
    try:
        y_pred = model.predict_proba(X)
    except AttributeError:
        if isinstance(model, SGDClassifier):
            y_pred = model.decision_function(X)
        else:
            y_pred = model.predict(X)

    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]

    train_data["y_pred"] = y_pred
    return train_data


def cross_validate_by_ticker(
    features, model, feature_names, target_name, return_model=False, **kwargs
):
    for ticker, rows in features.groupby("ticker"):
        yield from cross_validate_by_time(
            rows, model, feature_names, target_name, return_model, **kwargs
        )


def cross_validate_by_time(
    rows,
    model,
    feature_names,
    target_name,
    return_model=False,
    rule="1y",
    test_size=1,
    train_size=3,
    train_filter_f: Callable[[pd.DataFrame], pd.DataFrame] = None,
    **kwargs,
):
    for train_data, val_data, test_data in time_series_split(
        rows, split_size=rule, test_size=test_size, train_size=train_size, **kwargs
    ):
        incomplete = False
        for x in (train_data, val_data, test_data):
            if len(x) == 0:
                incomplete = True
                break

        if incomplete:
            continue

        if train_filter_f:
            train_data = train_filter_f(train_data)

        X = train_data[feature_names]
        y = train_data[target_name]
        estimator = clone(model).fit(X, y)
        _add_predicts(train_data, estimator, feature_names)
        _add_predicts(val_data, estimator, feature_names)
        _add_predicts(test_data, estimator, feature_names)

        if return_model:
            yield model, train_data, val_data, test_data
        else:
            yield train_data, val_data, test_data
