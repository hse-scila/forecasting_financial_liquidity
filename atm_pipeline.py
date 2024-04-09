import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler as sc


def calc_all_models(ts, entropy_ts, art_count_ts, model_list, start_date, 
                    end_date, all_models, model_params, media_name, 
                    entropy_name, entropy_lvl):
    """
    Function to evaluate all models for passed cash demand, entropy and articles count time series 
    """
    return_vals = []
    for model_name in model_list:
        print(f'model: {model_name}')
        resampled_ts = ts.T.resample('1W').sum().T
        resampled_ts = resampled_ts.apply(get_yoy_ts, axis=1)

        train_dates = [pd.to_datetime(resampled_ts.columns[0]),
                       np.min((entropy_ts.index[-1], art_count_ts.index[-1],
                               pd.to_datetime(end_date)))-timedelta(days=365)]
        test_dates = [np.min((entropy_ts.index[-1], art_count_ts.index[-1],
                              pd.to_datetime(end_date)))-timedelta(days=364),
                     np.min((entropy_ts.index[-1], art_count_ts.index[-1],
                             pd.to_datetime(end_date)))]

        entr_flag=True
        res = resampled_ts.apply(calc_score, axis=1, model_name=model_name, entr_flag=entr_flag,
                           test_dates=test_dates, entropy_ts=entropy_ts, art_count_ts=art_count_ts,
                           all_models=all_models, train_dates=train_dates, model_params=model_params)
        maes_entropy = res.apply(lambda x: x[0])
        preds_entropy = np.stack(res.apply(lambda x: x[1]))

        entr_flag = False
        res = resampled_ts.apply(calc_score, axis=1, model_name=model_name, entr_flag=entr_flag,
                           test_dates=test_dates, entropy_ts=entropy_ts, art_count_ts=art_count_ts, 
                           all_models=all_models, train_dates=train_dates, model_params=model_params)
        maes_endogen = res.apply(lambda x: x[0])
        preds_endogen = np.stack(res.apply(lambda x: x[1]))
        print(f'mae exogen: {np.round(np.mean(maes_entropy.values), 4)}, mae endogen: {np.round(np.mean(maes_endogen), 4)}')
        print(f'MAE improvement in %', np.round(100 * (np.mean(maes_endogen.values) - np.mean(maes_entropy))/np.mean(maes_endogen), 4), '%')
        print(f'% of ts with mae improvement: {np.round(np.nonzero(maes_endogen.values - maes_entropy.values > 0)[0].shape[0]/maes_entropy.shape[0]*100, 2)}')

        score = np.round(100 * (np.mean(maes_endogen.values) - np.mean(maes_entropy))/np.mean(maes_endogen), 4)
        perc_ts = np.round(np.nonzero(maes_endogen.values - maes_entropy.values > 0)[0].shape[0]\
                                                                   /maes_entropy.shape[0]*100, 2)

        return_vals.append([media_name, entropy_name, entropy_lvl, model_name,
                    score, perc_ts])
    return return_vals


def calc_score(ts, model_name, entr_flag, test_dates, train_dates, 
               entropy_ts, art_count_ts, all_models, model_params, 
               lags=5):
    """
    Evaluate specified model on passed time series
    """
    ts = ts.to_frame().sort_index()
    ts.columns = ['ts']
    ts = ts[:test_dates[1]]
    endogen_df = pd.DataFrame(ts['ts'])
    for lag in range(1, lags):
        endogen_df['lag: {}'.format(str(lag))] = endogen_df['ts'].shift(int(lag))
    endogen_df = endogen_df[endogen_df.columns[1:]]
    if entr_flag:
        data = pd.concat([endogen_df, entropy_ts.loc[endogen_df.index]], axis=1)
        data = pd.concat([data, art_count_ts.loc[endogen_df.index]], axis=1)
    else:
        data = endogen_df.copy()
    target_train = ts['ts'][train_dates[0]:train_dates[-1]].values[lags:]
    target_test = ts['ts'][test_dates[0]:test_dates[-1]].values[lags:]
    X_train = data[train_dates[0]:train_dates[-1]].values[lags:]
    X_test = data[test_dates[0]:test_dates[-1]].values[lags:]

    scaler_x = sc()
    X_train_sc = scaler_x.fit_transform(X_train)
    X_test_sc = scaler_x.transform(X_test)
    scaler_y = sc()
    y_train_sc = np.ravel(scaler_y.fit_transform(target_train.reshape((-1, 1))))
    y_test_sc = np.ravel(scaler_y.transform(target_test.reshape((-1, 1))))
    model = all_models[model_name](**model_params[model_name])
    model.fit(X_train_sc, y_train_sc)
    preds = np.ravel(scaler_y.inverse_transform(model.predict(\
                                  X_test_sc).reshape((-1, 1))))
    mae = np.mean(np.abs(target_test - preds))
    return mae, preds


def yoy(arr):
    """
    Yoy calculation function
    """
    res = arr.diff() / arr.shift()
    return res


def get_yoy_ts(yoy_ts):
    """
    Calculate Yoy for time series with daily granularity
    """
    yoy_ts = yoy_ts.to_frame()
    yoy_ts.columns = ['ts']
    yoy_ts['number']= yoy_ts.groupby(yoy_ts.index.year)\
                                    .apply(lambda x: np.cumsum(x==x)).values.ravel()
    yoy_ts = yoy_ts.groupby('number')['ts'].apply(yoy).dropna()
    yoy_ts.index = [i[1] for i in yoy_ts.index]
    return yoy_ts
