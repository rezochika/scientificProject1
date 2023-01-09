import json

import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.engine.input_layer import InputLayer
from keras.layers import LSTM, Dense, Bidirectional
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from keras.optimizer_v2.adam import Adam
import US


def build_dnn(ws, lr, ep, bi, dnnlayers, model_path, df):
    path = model_path + 'LSTM_layers_'+str(dnnlayers)
    if bi: path = path+'_Bidirectional'
    path += '/'
    df['Troloff2'] = df['Troloff'] ** 2
    drivers = ['wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t', 'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7', 'cy', 'c']
    test_data = df.loc[:, drivers]
    mean = np.mean(test_data['c'])
    std = np.std(test_data['c'])
    test_data['c'] = (test_data['c'] - mean) / std
    test_data['cy'] = (test_data['cy'] - mean) / std
    test_data['c7'] = (test_data['c7'] - mean) / std

    means = {'c': [mean, std], 'cy': [mean, std], 'c7': [mean, std]}

    for col in test_data.columns:
        if col not in ['c', 'cy', 'c7']:
            test_data[col], m, s = US.normalizeArray(test_data[col])
            means[col] = [m, s]
    x1, y1 = US.dftoXy1(test_data, ws)

    data_l = len(y1)

    x_train1, y_train1 = x1[0:data_l - ws], y1[0:data_l - ws]
    x_val1, y_val1 = x1[data_l - ws:data_l], y1[data_l - ws:data_l]
    neurs = len(x_train1[0][0])

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
    es = EarlyStopping(monitor='val_loss', verbose=0, patience=1000, restore_best_weights=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=lr),
                  metrics=[MeanAbsolutePercentageError()])

    history = model.fit(x_train1, y_train1, validation_data=(x_val1, y_val1), epochs=ep, callbacks=[es],
                        verbose=2)
    model.save(path + 'model.h5')
    with open(path + "means.json", "w") as write_file:
        json.dump(means, write_file, indent=4)

    loss_hist = history.history['val_loss']
    best_epoch = np.argmin(loss_hist)
    print(f'best epoch: {best_epoch + 1}')
    his = {'epoch': (best_epoch * 1.0 + 1)}
    for key in history.history.keys():
        print(f'{key:<40}={history.history[key][best_epoch]:.4f}')
        his[key] = history.history[key][best_epoch]

    with open(path + "history.json", "w") as write_file:
        json.dump(his, write_file, indent=4)
