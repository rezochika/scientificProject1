import math
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

import sqlconnect


def build_ols(model_path, model_id=1):
    start = datetime.now()
    print('build_ols started at {0}'.format(datetime.now()))
    df, df1 = sqlconnect.getdatafromsqlols()
    path = '{0}OLS'.format(model_path)
    df['Troloff2'] = df['Troloff'] ** 2
    drivers = df.columns.tolist()
    drivers.remove('hdd')
    drivers.remove('Date')

    test_data = df.loc[:, drivers]
    columns = test_data.columns.tolist()

    model = LinearRegression()
    x = test_data[[c for c in columns if c not in ['c']]]
    y = test_data['c']

    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)*100
    loss = np.power(mean_squared_error(y, y_pred), 0.5)
    y_pred = model.predict(x[-7:])
    val_mape = mean_absolute_percentage_error(y[-7:], y_pred)*100
    val_loss = np.power(mean_squared_error(y[-7:], y_pred), 0.5)
    os.makedirs(os.path.dirname(path + r'/ols.pickle'), exist_ok=True)
    with open(path + r'/ols.pickle', 'wb') as f:
        pickle.dump(model, f)

    with open(path + r'/ols.pickle', 'rb') as f:
        model_file = f.read()
    elapsed = (datetime.now() - start).total_seconds() / 60
    sqlconnect.insert_model(model_id=model_id, model=model_file, json_data='', r2=r2, loss=loss, val_loss=val_loss,
                            mape=mape, val_mape=val_mape, elapsed=elapsed)


def predict_ols(model_path, model_id=1):
    ws = 7
    df, df1 = sqlconnect.getdatafromsqlols()
    df['Troloff2'] = df['Troloff'] ** 2
    df1 ['Troloff2'] = df1 ['Troloff'] ** 2
    drivers = df.columns.tolist()
    drivers.remove('hdd')
    drivers.remove('Date')

    test_data = df.loc[:, drivers]
    columns = test_data.columns.tolist()
    forecast_data = pd.concat([test_data [len(test_data) - ws:], df1.loc [:, drivers]])
    path = '{0}OLS'.format(model_path)
    with open(path+r'/ols.pickle', 'rb') as handle:
        model = pickle.load(handle)

    for r in range(ws, len(forecast_data)):
        if math.isnan(float(forecast_data ['c7'] [r])): forecast_data ['c7'] [r] = forecast_data ['c'] [r - 7]
        if math.isnan(float(forecast_data ['cy'] [r])): forecast_data ['cy'] [r] = forecast_data ['c'] [r - 1]
        dcy = forecast_data ['cy'] [r] / forecast_data ['cy'] [r - 1]
        forecast_data ['dcy'] [r] = 0 if dcy < 1.3 else dcy
        # if math.isnan(float(forecast_data['c'][r])):
        x1 = forecast_data [[c for c in columns if c not in ['c']]] [r - 1:r]
        y_p = model.predict(x1)
        forecast_data ['c'] [r] = y_p [len(y_p) - 1]

    # print(forecast_data)
    results = forecast_data ['c']  # * means['c'][1] + means['c'][0]
    return model_id, results
