import json
import sys

import absl.logging
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

absl.logging.set_verbosity(absl.logging.ERROR)
modelPath = 'modelTbilisi1/'
rebuild = False
ws = 7
lr = 0.008886502
ep = 2000
for arg in sys.argv:
    if arg == 'rebuild=1':
        rebuild = True
    elif arg.startswith("modelPath"):
        modelPath = arg.replace("modelPath=", "")
    elif arg.startswith("epochs"):
        ep = int(arg.replace("epochs=", ""))

# rebuild = True

# 0.011353
df = pd.read_csv(r'Data\TbilisiData.csv')
df1 = pd.read_csv(r'Data\TbilisiFore.csv')
df['Troloff2'] = df['Troloff'] ** 2
# df['Troloff3'] = df['Troloff'] ** 3
df1['Troloff2'] = df1['Troloff'] ** 2
# df1['Troloff3'] = df1['Troloff'] ** 3
df.index = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df1.index = pd.to_datetime(df1['Date'], format='%m/%d/%Y')
drivers = ['wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t', 'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7', 'cy',
           'c']
TestData = df.loc[:, drivers]

ForecastData = pd.concat([TestData[len(TestData) - 2 * ws:], df1.loc[:,
                                                             ['wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t',
                                                              'Troloff', 'Troloff2', 'mx2', 'mn4']]])
mean = np.mean(TestData['c'])
std = np.std(TestData['c'])
TestData['c'] = (TestData['c'] - mean) / std
TestData['cy'] = (TestData['cy'] - mean) / std
TestData['c7'] = (TestData['c7'] - mean) / std

means = {'c': [mean, std], 'cy': [mean, std], 'c7': [mean, std]}

for col in TestData.columns:
    if col not in ['c', 'cy', 'c7']:
        TestData[col], m, s = US.normalizeArray(TestData[col])
        means[col] = [m, s]

# with open(modelPath + "means.json", "w") as write_file:
#     json.dump(means, write_file, indent=4)
X1, y1 = US.dftoXy1(TestData, ws)

dataL = len(y1)
testL = ws
X_train1, y_train1 = X1[0:dataL], y1[0:dataL]
X_val1, y_val1 = X1[dataL - testL:dataL], y1[dataL - testL:dataL]
X_test1, y_test1 = X1[dataL - 3:], y1[dataL - 3:]
neurs = len(X_train1[0][0])

if rebuild:
    modelDNN = Sequential()
    modelDNN.add(InputLayer((ws, neurs)))
    modelDNN.add(Bidirectional(LSTM(64, return_sequences=True)))
    modelDNN.add(Bidirectional(LSTM(128)))
    modelDNN.add(Dense(16, 'relu'))
    modelDNN.add(Dense(8, 'linear'))
    modelDNN.add(Dense(1, 'linear'))
    modelDNN.summary()

    cp = ModelCheckpoint(modelPath, save_best_only=True, monitor='val_loss', verbose=1)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=500, restore_best_weights=True)
    modelDNN.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=lr),
                     metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()])

    # lr_finder = LRFinder(model7)
    # lr_finder.find(X_train1, y_train1, 0.0005, 0.1, 32, ep)
    # lr_finder.plot_loss()
    # # lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.02, 0.01))
    # print(float(model7.optimizer.lr))
    # print(lr_finder.best_lr, lr_finder.best_loss)
    # exit()

    history = modelDNN.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=ep, callbacks=[es],
                           verbose=2)
    modelDNN.save(modelPath)
    with open(modelPath + "means.json", "w") as write_file:
        json.dump(means, write_file, indent=4)

    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='validation')
    # plt.legend()
    # plt.show()

modelDNN = load_model(modelPath)
means = json.load(open(modelPath + "means.json", "r"))

# US.plot_predictions(modelDNN, X_test1, y_test1, 'Test Predictions', 'Actuals', 'model7',
#                     means['c'][0], means['c'][1],
#                     float(modelDNN.optimizer.lr), ep, ws, start=0, end=100)

for col in ForecastData.columns:
    ForecastData[col] = (ForecastData[col] - means[col][0]) / means[col][1]

for r in range(2 * ws, len(ForecastData)):
    ForecastData['c7'][r] = ForecastData['c'][r - 7]
    ForecastData['cy'][r] = ForecastData['c'][r - 1]
    dcy = ForecastData['cy'][r] / ForecastData['cy'][r - 1]
    ForecastData['dcy'][r] = - means['dcy'][0] / means['dcy'][1]
    X1, y1 = US.dftoXy1(ForecastData[:r + 1], ws)
    yP = modelDNN.predict(X1)
    ForecastData['c'][r] = yP[len(yP) - 1]

results = ForecastData['c'] * means['c'][1] + means['c'][0]
print(results)
plt.plot(results)
plt.show()

exit(0)
