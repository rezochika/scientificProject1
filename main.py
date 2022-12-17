import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.input_layer import InputLayer
from keras.layers import LSTM, Dense
from keras.losses import MeanSquaredError, mean_absolute_percentage_error
from keras.metrics import RootMeanSquaredError
from keras.optimizer_v2.adam import Adam
from keras.saving.save import load_model
import absl.logging
from matplotlib import pyplot as plt

from ir_finder import LRFinder

absl.logging.set_verbosity(absl.logging.ERROR)

import US
import US as us

ws = 7
lr = 0.008886502

# 0.011353
ep = 2000
df1 = pd.read_csv(r'C:\Users\rezoc\OneDrive\Documents\TbilisiData.csv')
df1.index = pd.to_datetime(df1['Date'], format='%m/%d/%Y')
cons = df1[500:].loc[:,
       ['wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t', 'Troloff', 'mx2', 'mn4', 'dcy', 'c7', 'cy', 'c']]

meann = np.mean(cons['c'])
stdd = np.std(cons['c'])
print(meann, stdd)
cons['c'] = (cons['c'] - meann) / stdd
cons['cy'] = (cons['cy'] - meann) / stdd
cons['c7'] = (cons['c7'] - meann) / stdd
means = {'c': [meann, stdd]}
for col in cons.columns:
    if col not in ['c', 'cy', 'c7']:
        cons[col], m, s = US.normalizeArray(cons[col])
        means[col] = [m, s]
print(means)

X1, y1 = us.dftoXy1(cons, ws)

print(X1.shape)
print(y1.shape)

dataL = len(y1)
testL = 4
X_train1, y_train1 = X1[0:dataL-testL], y1[0:dataL-testL]
X_val1, y_val1 = X1[dataL - testL - testL:dataL - testL], y1[dataL - testL - testL:dataL - testL]
X_test1, y_test1 = X1[dataL - testL:], y1[dataL - testL:]
print(X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape)
neurs = len(X_train1[0][0])
print(X_test1[0])

model7 = Sequential()
model7.add(InputLayer((ws, neurs)))
model7.add(LSTM(64, return_sequences=True))
model7.add(LSTM(128))
model7.add(Dense(16, 'relu'))
# model7.add(Dense(8, 'relu'))
model7.add(Dense(1, 'linear'))
model7.summary()

cp7 = ModelCheckpoint('model7/', save_best_only=True, monitor='val_loss', verbose=1)
es = EarlyStopping(monitor='val_loss', verbose=1, patience=300)
model7.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=lr), metrics=[RootMeanSquaredError()])

# lr_finder = LRFinder(model7)
# lr_finder.find(X_train1, y_train1, 0.0005, 0.1, 32, ep)
# lr_finder.plot_loss()
# # lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.02, 0.01))
# print(float(model7.optimizer.lr))
# print(lr_finder.best_lr, lr_finder.best_loss)
# exit()

history = model7.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=ep, callbacks=[cp7, es], verbose=2)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

model7 = load_model('model7/')
xP = np.array([X_test1[2], X_test1[3]])
print(xP)
print(xP.shape)
yP = model7.predict(xP) * stdd + meann
print(yP)
print(us.plot_predictions(model7, X_test1, y_test1, 'Test Predictions', 'Actuals', 'model7', meann, stdd,
                          float(model7.optimizer.lr), ep, ws,
                          start=0,
                          end=100))
exit(0)

# model1 = Sequential()
# model1.add(InputLayer((WINDOW_SIZE, neurs)))
# model1.add(LSTM(64))
# model1.add(Dense(8, 'relu'))
# model1.add(Dense(1, 'linear'))
#
# model1.summary()
#
# cp1 = ModelCheckpoint('modelTbilisiMulti/', monitor='val_loss', save_best_only=True)
# model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01), metrics=[RootMeanSquaredError()])
#
# model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=1000, callbacks=[cp1])
# model1 = load_model('modelTbilisiMulti/')
# # print(us.plot_predictions(model1, X_train1, y_train1, 'Train Predictions', 'Actuals', 'model1', meann, stdd, start=1300, end=1310))
# # print(us.plot_predictions1(model1, X_val1, y_val1, 'Val Predictions', 'Actuals', 'model1', start=0, end=100))
# print(us.plot_predictions(model1, X_test1, y_test1, 'Test Predictions', 'Actuals', 'model1', meann, stdd, start=0, end=135))
