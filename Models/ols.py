import os
import pickle
from sklearn.linear_model import LinearRegression

import sqlconnect


def build_ols(model_path):
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
    os.makedirs(os.path.dirname(path+r'/ols.pickle'), exist_ok=True)
    with open(path+r'/ols.pickle', 'wb') as f:
        pickle.dump(model, f)
