import json
import math

import numpy as np
from datetime import datetime

import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.engine.input_layer import InputLayer
from keras.layers import LSTM, Dense, Bidirectional
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsolutePercentageError
from keras.models import load_model
from keras.optimizer_v2.adam import Adam
from sklearn.metrics import r2_score

import US
import sqlconnect


def build_dnn(ws, lr, ep, bi, dnnlayers, model_path, df, verbose, model_id):
    start = datetime.now()
    path = '{0}LSTM_layers_{1}'.format(model_path, dnnlayers)
    print('LSTM_layers_{0}_bi={1} started at {2}'.format(dnnlayers, bi, datetime.now()))
    if bi: path = path + '_Bidirectional'
    path += '/'
    df ['Troloff2'] = df ['Troloff'] ** 2
    drivers = ['wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t', 'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7', 'cy',
               'c']
    test_data = df.loc [:, drivers]
    mean = np.mean(test_data ['c'])
    std = np.std(test_data ['c'])
    test_data ['c'] = (test_data ['c'] - mean) / std
    test_data ['cy'] = (test_data ['cy'] - mean) / std
    test_data ['c7'] = (test_data ['c7'] - mean) / std

    means = {'c': [mean, std], 'cy': [mean, std], 'c7': [mean, std]}

    for col in test_data.columns:
        if col not in ['c', 'cy', 'c7']:
            test_data [col], m, s = US.normalizeArray(test_data [col])
            means [col] = [m, s]
    x1, y1 = US.dftoXy1(test_data, ws)

    data_l = len(y1)

    x_train1, y_train1 = x1 [0:data_l - ws], y1 [0:data_l - ws]
    x_val1, y_val1 = x1 [data_l - ws:data_l], y1 [data_l - ws:data_l]
    neurs = len(x_train1 [0] [0])

    model = Sequential()
    model.add(InputLayer((ws, neurs)))
    if bi:
        model.add(Bidirectional(LSTM(dnnlayers, return_sequences=True)))
        model.add(Bidirectional(LSTM(dnnlayers * 2)))
    else:
        model.add((LSTM(dnnlayers, return_sequences=True)))
        model.add((LSTM(dnnlayers * 2)))
    model.add(Dense(16, 'relu'))
    model.add(Dense(8, 'linear'))
    model.add(Dense(1, 'linear'))
    model.summary()

    # cp = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)
    es = EarlyStopping(monitor='val_loss', verbose=verbose, patience=1000, restore_best_weights=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=lr),
                  metrics=[MeanAbsolutePercentageError()])

    history = model.fit(x_train1, y_train1, validation_data=(x_val1, y_val1), epochs=ep, callbacks=[es],
                        verbose=verbose)
    model.save(path + 'model.h5')
    with open(path + "means.json", "w") as write_file:
        json.dump(means, write_file, indent=4)

    loss_hist = history.history ['val_loss']
    best_epoch = np.argmin(loss_hist)
    print(f'best epoch: {best_epoch + 1}')
    his = {'epoch': (best_epoch * 1.0 + 1)}
    for key in history.history.keys():
        print(f'{key:<40}={history.history [key] [best_epoch]:.4f}')
        his [key] = history.history [key] [best_epoch]

    with open(path + "history.json", "w") as write_file:
        json.dump(his, write_file, indent=4)

    y_pred = model.predict(x_train1)
    r2 = r2_score(y_train1, y_pred)
    elapsed = (datetime.now() - start).total_seconds() / 60
    with open(path + 'model.h5', 'rb') as f:
        model_file = f.read()
    sqlconnect.insert_model(model_id=model_id, model=model_file, r2=r2,
                            mape=history.history ['mean_absolute_percentage_error'] [best_epoch],
                            loss=np.sqrt(history.history ['loss'] [best_epoch] * std + mean),
                            val_mape=history.history ['val_mean_absolute_percentage_error'] [best_epoch],
                            val_loss=np.sqrt(history.history ['val_loss'] [best_epoch] * std + mean),
                            json_data=json.dumps(means), elapsed=elapsed)


def predict_dnn(ws, lr, ep, bi, dnnlayers, model_path, df, df1, verbose, model_id):
    path = '{0}LSTM_layers_{1}'.format(model_path, dnnlayers)
    print('LSTM_layers_{0}_bi={1} prediction started at {2}'.format(dnnlayers, bi, datetime.now()))
    if bi: path = path + '_Bidirectional'
    path += '/'
    model = load_model(path + 'model.h5')
    means = json.load(open(path + "means.json", "r"))

    df ['Troloff2'] = df ['Troloff'] ** 2
    df1 ['Troloff2'] = df1 ['Troloff'] ** 2
    drivers = ['wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t', 'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7', 'cy',
               'c']
    test_data = df.loc [:, drivers]

    forecast_data = pd.concat([test_data [len(test_data) - 3 * ws:], df1.loc [:,
                                                                  ['wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t',
                                                                   'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7',
                                                                   'cy',
                                                                   'c']]])

    # print(test_data)
    mean = means ['c'] [0]
    std = means ['c'] [1]
    test_data ['c'] = (test_data ['c'] - mean) / std
    test_data ['cy'] = (test_data ['cy'] - mean) / std
    test_data ['c7'] = (test_data ['c7'] - mean) / std

    means = {'c': [mean, std], 'cy': [mean, std], 'c7': [mean, std]}

    for col in test_data.columns:
        if col not in ['c', 'cy', 'c7']:
            test_data [col], m, s = US.normalizeArray(test_data [col])
            means [col] = [m, s]

    for col in forecast_data.columns:
        forecast_data [col] = (forecast_data [col] - means [col] [0]) / means [col] [1]

    # for r in range(2*ws-1, 3*ws):
    #     # forecast_data['c7'][r] = math.nan
    #     # forecast_data['cy'][r] = math.nan
    #     forecast_data['c'][r] = math.nan

    for r in range(2 * ws - 2, len(forecast_data)):
        if math.isnan(float(forecast_data ['c7'] [r])): forecast_data ['c7'] [r] = forecast_data ['c'] [r - 7]
        if math.isnan(float(forecast_data ['cy'] [r])): forecast_data ['cy'] [r] = forecast_data ['c'] [r - 1]
        # dcy = forecast_data['cy'][r] / forecast_data['cy'][r - 1]
        forecast_data ['dcy'] [r] = - means ['dcy'] [0] / means ['dcy'] [1]
        if math.isnan(float(forecast_data ['c'] [r])):
            x1, y1 = US.dftoXy1(forecast_data [:r + 1], ws)
            y_p = model.predict(x1)
            forecast_data ['c'] [r] = y_p [len(y_p) - 1]

    # print(forecast_data)
    results = forecast_data ['c'] * means ['c'] [1] + means ['c'] [0]
    # print(results)
    return model_id, results
