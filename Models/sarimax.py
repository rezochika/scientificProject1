import pickle

import pandas as pd
from pmdarima import auto_arima


def build_sarimax(model_path, df):
    path = '{0}SARIMAX'.format(model_path)
    df['DT'] = pd.to_datetime(df['DT'], infer_datetime_format=True)
    df = df.set_index(['DT'])
    df['Troloff2'] = df['Troloff'] ** 2
    drivers = ['c', 'wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t', 'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7',
               'cy']
    training_y = df.loc[:, drivers].iloc[:, 0]
    training_x = df.loc[:, drivers].iloc[:, 1:]
    model = auto_arima(y=training_y,
                       X=training_x,
                       m=7)
    with open(path+r'/sarimax.pickle', 'wb') as f:
        pickle.dump(model, f)
