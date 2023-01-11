import math
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

import sqlconnect


def build_sarimax(model_path, df, model_id=10):
    print('build_sarimax started at {0}'.format(datetime.now()))
    start = datetime.now()
    path = '{0}SARIMAX'.format(model_path)
    df['DT'] = pd.to_datetime(df['DT'], infer_datetime_format=True)
    df = df.set_index(['DT'])
    df['Troloff2'] = df['Troloff'] ** 2
    drivers = ['c', 'wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t', 'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7',
               'cy']
    y = df.loc[:, drivers].iloc[:, 0]
    x = df.loc[:, drivers].iloc[:, 1:]
    model = auto_arima(y=y,
                       X=x,
                       m=7)
    os.makedirs(os.path.dirname(path+r'/sarimax.pickle'), exist_ok=True)
    with open(path+r'/sarimax.pickle', 'wb') as f:
        pickle.dump(model, f)
    y_pred = model.predict(n_periods=len(x), X=x)
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)*100
    loss = np.power(mean_squared_error(y, y_pred), 0.5)
    y_pred = model.predict(n_periods=7, X=x[-7:])
    val_mape = mean_absolute_percentage_error(y[-7:], y_pred)*100
    val_loss = np.power(mean_squared_error(y[-7:], y_pred), 0.5)
    elapsed = (datetime.now() - start).total_seconds() / 60
    with open(path+r'/sarimax.pickle', 'rb') as f:
        model_file = f.read()
    sqlconnect.insert_model(model_id=model_id, model=model_file, json_data='', r2=r2, loss=loss, val_loss=val_loss,
                            mape=mape, val_mape=val_mape, elapsed=elapsed)


def predict_sarimax(model_path, df,df1, model_id=10):
    path = '{0}SARIMAX'.format(model_path)
    with open(path+r'/sarimax.pickle', 'rb') as handle:
        model = pickle.load(handle)
    df ['DT'] = pd.to_datetime(df ['DT'], infer_datetime_format=True)
    df = df.set_index(['DT'])
    df ['Troloff2'] = df ['Troloff'] ** 2
    df1 ['Troloff2'] = df1 ['Troloff'] ** 2
    drivers = ['c', 'wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t', 'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7',
               'cy']
    test_data = df.loc [:, drivers]
    forecast_data = pd.concat([test_data [len(test_data) - 3:], df1.loc [:,
                                                              ['c', 'wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t',
                                                               'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7',
                                                               'cy']]])

    for r in range(3, len(forecast_data)):
        if math.isnan(float(forecast_data ['c7'] [r])): forecast_data ['c7'] [r] = forecast_data ['c'] [r - 7]
        if math.isnan(float(forecast_data ['cy'] [r])): forecast_data ['cy'] [r] = forecast_data ['c'] [r - 1]
        # dcy = forecast_data['cy'][r] / forecast_data['cy'][r - 1]
        forecast_data ['dcy'] [r] = 0
        if math.isnan(float(forecast_data ['c'] [r])):
            test_x = forecast_data.iloc [r - 1:r, 1:]
            y_p = model.predict(n_periods=len(test_x), X=test_x)
            forecast_data ['c'] [r] = y_p [len(y_p) - 1]
    print(forecast_data ['c'])
    results = forecast_data ['c']
    return model_id, results
