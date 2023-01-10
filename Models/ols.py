import pickle
from statistics import LinearRegression

import pandas as pd
from pmdarima import auto_arima


def build_ols(model_path, df):
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
    with open(path+r'/ols.pickle', 'wb') as f:
        pickle.dump(model, f)
