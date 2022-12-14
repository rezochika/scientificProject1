import json
import math
import sys

import absl.logging
import keras.losses
import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.input_layer import InputLayer
from keras.layers import LSTM, Dense, Bidirectional
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from keras.optimizer_v2.adam import Adam
from keras.saving.save import load_model
from matplotlib import pyplot as plt

import US
from ir_finder import LRFinder

absl.logging.set_verbosity(absl.logging.ERROR)
modelPath = 'modelAp/'
rebuild = False

lr = 0.0033330107  # # 0.011353
ep = 5000
plot = False
ws = 72
df = pd.read_csv(r'Data\ApConsTemp.csv')
df.index = pd.to_datetime(df['DT'])
df['Temp'] = df['Temp'] ** 2
df['H'] = pd.to_datetime(df['DT']).dt.hour
df['M'] = pd.to_datetime(df['DT']).dt.month
df['W'] = pd.to_datetime(df['DT']).dt.day_of_week
df['Msin'] = np.sin(2 * np.pi * df['M'] / 12)
df['Mcos'] = np.cos(2 * np.pi * df['M'] / 12)

df['Wsin'] = np.sin(2 * np.pi * df['W'] / 7)
df['Wcos'] = np.cos(2 * np.pi * df['W'] / 7)

df['Hsin'] = np.sin(2 * np.pi * df['H'] / 24)
df['Hcos'] = np.cos(2 * np.pi * df['H'] / 24)

df['c'] = df['Value']
df.drop('DT', inplace=True, axis=1)
df.drop('H', inplace=True, axis=1)
df.drop('M', inplace=True, axis=1)
df.drop('W', inplace=True, axis=1)
df.drop('Value', inplace=True, axis=1)
# df.to_csv(r'Data\ForOls.csv')
TestData = df
ForecastData = TestData[len(TestData) - 3 * ws:]
mean = np.mean(TestData['c'])
std = np.std(TestData['c'])
TestData['c'] = (TestData['c'] - mean) / std
TestData['cy'] = (TestData['cy'] - mean) / std

means = {'c': [mean, std], 'cy': [mean, std], 'c7': [mean, std]}

for col in TestData.columns:
    if col not in ['c', 'cy', 'c7']:
        TestData[col], m, s = US.normalizeArray(TestData[col])
        means[col] = [m, s]

X1, y1 = US.dftoXy1(TestData, ws)

dataL = len(y1)

X_train1, y_train1 = X1[0:dataL - ws], y1[0:dataL - ws]
X_val1, y_val1 = X1[dataL - 2 * ws:dataL - ws], y1[dataL - 2 * ws:dataL - ws]
X_test1, y_test1 = X1[dataL - 2 * ws:], y1[dataL - 2 * ws:]
neurs = len(X_train1[0][0])

if rebuild:
    modelDNN = Sequential()
    modelDNN.add(InputLayer((ws, neurs)))
    modelDNN.add((LSTM(126, return_sequences=True)))
    modelDNN.add((LSTM(256)))
    modelDNN.add(Dense(16, 'relu'))
    modelDNN.add(Dense(8, 'linear'))
    modelDNN.add(Dense(1, 'linear'))
    modelDNN.summary()

    cp = ModelCheckpoint(modelPath, save_best_only=True, monitor='val_loss', verbose=1)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=1000, restore_best_weights=True)
    modelDNN.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=lr),
                     metrics=[MeanAbsolutePercentageError()])

    # lr_finder = LRFinder(modelDNN)
    # lr_finder.find(X_train1, y_train1, 0.0005, 0.1, 32, ep)
    # lr_finder.plot_loss()
    # # lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.02, 0.01))
    # print(float(modelDNN.optimizer.lr))
    # print(lr_finder.best_lr, lr_finder.best_loss)
    # exit()

    history = modelDNN.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=ep, callbacks=[es],
                           verbose=2)
    modelDNN.save(modelPath)
    with open(modelPath + "means.json", "w") as write_file:
        json.dump(means, write_file, indent=4)

    loss_hist = history.history['val_loss']
    bestEpoch = np.argmin(loss_hist)
    print(f'best epoch: {bestEpoch + 1}')
    his = {'epoch': (bestEpoch * 1.0 + 1)}
    for key in history.history.keys():
        print(f'{key:<40}={history.history[key][bestEpoch]:.4f}')
        his[key] = history.history[key][bestEpoch]

    with open(modelPath + "history.json", "w") as write_file:
        json.dump(his, write_file, indent=4)

    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='validation')
    # plt.legend()
    # plt.show()

modelDNN = load_model(modelPath)
means = json.load(open(modelPath + "means.json", "r"))

# US.plot_predictions(modelDNN, X_test1, y_test1, 'Test Predictions', 'Actuals', 'modelAP',
#                     means['c'][0], means['c'][1],
#                     float(modelDNN.optimizer.lr), ep, ws, start=0, end=100)

for col in ForecastData.columns:
    ForecastData[col] = (ForecastData[col] - means[col][0]) / means[col][1]

for r in range(2*ws, len(ForecastData)):
    # ForecastData['c7'][r] = math.nan
    ForecastData['cy'][r] = math.nan
    ForecastData['c'][r] = math.nan

for r in range(2*ws, len(ForecastData)):
    if math.isnan(float(ForecastData['cy'][r])): ForecastData['cy'][r] = ForecastData['c'][r - 1]
    if math.isnan(float(ForecastData['c'][r])):
        X_1, y_1 = US.dftoXy1(ForecastData[:r + 1], ws)
        yP = modelDNN.predict(X_1)
        ForecastData['c'][r] = yP[len(yP) - 1]

# print(ForecastData)
results = ForecastData['c'] * means['c'][1] + means['c'][0]
actual = TestData['c'][len(TestData) - 3 * ws:]
actual = actual * means['c'][1] + means['c'][0]
if plot:
    plt.plot(actual[ws:2*ws+36], color='#1f76b4')
    plt.plot(results[2*ws:2*ws+36], color='darkgreen')
    plt.grid(visible=True, which='both')
    plt.show()

X1, y2 = US.dftoXy1(ForecastData, ws)
print(X1.shape)
US.plot_predictions(modelDNN, X1, y1[len(y1)-2*ws:], 'Predictions', 'Actuals', 'model??????',
                    means['c'][0], means['c'][1],
                    float(modelDNN.optimizer.lr), ep, ws, start=84, end=108)

# TestData['Temp'][len(TestData)-ws:].plot()
# plt.show()
# TestData['c'][len(TestData)-ws:].plot()
# plt.show()