import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.engine.input_layer import InputLayer
from keras.layers import LSTM, Dense
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.saving.save import load_model

import US
import US as us

WINDOW_SIZE = 5
df1 = pd.read_csv(r'/Users/rezochikashua/Data/TbilisiData.csv')
df1.index = pd.to_datetime(df1['Date'], format='%m/%d/%Y')
cons = df1[2000:].loc[:, ['wf', 'IHT', 'wfy', 'dhdd1', 'dhdd13', 'hdd1t', 'Troloff', 'mx2', 'mn4', 'c7', 'cy', 'c']]

meann = np.mean(cons['c'])
stdd = np.std(cons['c'])
print(meann, stdd)
cons['c'] = (cons['c'] - meann) / stdd
cons['cy'] = (cons['cy'] - meann) / stdd
cons['c7'] = (cons['c7'] - meann) / stdd

for col in cons.columns:
    if col not in ['c', 'cy', 'c7']: cons[col] = US.normalizeArray(cons[col])

print(cons.head(10))


# plt.plot(cons)
# plt.show()

X1, y1 = us.dftoXy1(cons, WINDOW_SIZE)

print(X1.shape)
print(y1.shape)

X_train1, y_train1 = X1[:1800], y1[0:1800]
X_val1, y_val1 = X1[2000:], y1[2000:]
X_test1, y_test1 = X1[2000:], y1[2000:]
print(X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape)

model1 = Sequential()
model1.add(InputLayer((WINDOW_SIZE, 12)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()

cp1 = ModelCheckpoint('modelTbilisiMulti/', monitor='val_loss', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01), metrics=[RootMeanSquaredError()])

model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=10, callbacks=[cp1])
model1 = load_model('modelTbilisiMulti/')
# print(us.plot_predictions(model1, X_train1, y_train1, 'Train Predictions', 'Actuals', 'model1', meann, stdd, start=1300, end=1310))
# print(us.plot_predictions1(model1, X_val1, y_val1, 'Val Predictions', 'Actuals', 'model1', start=0, end=100))
print(us.plot_predictions(model1, X_test1, y_test1, 'Test Predictions', 'Actuals', 'model1', meann, stdd, start=0, end=135))
exit(0)
