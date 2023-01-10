import math
import pickle
import sys
from datetime import datetime

import absl.logging
import matplotlib.pylab as plt
import pandas as pd
from pmdarima import auto_arima

import sqlconnect

absl.logging.set_verbosity(absl.logging.ERROR)
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
modelPath = 'modelLSTM6420230105/'
rebuild = False
ep = 5000
plot = False
Bi = False
dnnlayers = 64
for arg in sys.argv:
    if arg == 'rebuild=1':
        rebuild = True
    elif arg.startswith("modelPath"):
        modelPath = arg.replace("modelPath=", "")
    elif arg.startswith("epochs"):
        ep = int(arg.replace("epochs=", ""))
    elif arg.startswith("layers"):
        dnnlayers = int(arg.replace("layers=", ""))
    elif arg == 'plot=1':
        plot = True
    elif arg == 'bi=1':
        Bi = True
print(datetime.now())


ws = 3
df, df1 = sqlconnect.getdatafromsql()
df['DT'] = pd.to_datetime(df['DT'], infer_datetime_format=True)
df = df.set_index(['DT'])
df['Troloff2'] = df['Troloff'] ** 2
df1['Troloff2'] = df1['Troloff'] ** 2
drivers = ['c', 'wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t', 'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7',
           'cy']
TestData = df.loc[:, drivers]
ForecastData = pd.concat([TestData[len(TestData) - ws:], df1.loc[:,
                                                             ['c', 'wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t',
                                                              'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7', 'cy']]])

# decomposed = seasonal_decompose(TestData['c'], model='additive', extrapolate_trend='freq', period=365)
# decomposed.plot()
# plt.show()

training_y = TestData.iloc[:, 0]
test_y = TestData.iloc[-3:, 0]
training_X = TestData.iloc[:, 1:]
test_X = TestData.iloc[-3:, 1:]
if rebuild:
    model = auto_arima(y=training_y,
                       X=training_X,
                       m=7)
    with open(r'modelSARIMAX/sarimax.pickle', 'wb') as f:
        pickle.dump(model, f)

with open(r'modelSARIMAX/sarimax.pickle', 'rb') as handle:
    model = pickle.load(handle)
print(datetime.now())
predictions = pd.Series(model.predict(n_periods=3, X=test_X))
predictions.index = test_y.index
# test_y['preds'] = predictions
# print(predictions)
# print(test_y)
# Visualize
# training_y['2023-01-01':].plot(figsize=(16, 6), legend=True)
# test_y.plot(legend=True)
# predictions.plot()
# plt.show()
for r in range(ws, len(ForecastData)):
    if math.isnan(float(ForecastData['c7'][r])): ForecastData['c7'][r] = ForecastData['c'][r - 7]
    if math.isnan(float(ForecastData['cy'][r])): ForecastData['cy'][r] = ForecastData['c'][r - 1]
    # dcy = ForecastData['cy'][r] / ForecastData['cy'][r - 1]
    ForecastData['dcy'][r] = 0
    if math.isnan(float(ForecastData['c'][r])):
        test_X = ForecastData.iloc[r-1:r, 1:]
        yP = model.predict(n_periods=len(test_X), X=test_X)
        ForecastData['c'][r] = yP[len(yP) - 1]
print(ForecastData['c'])
results = ForecastData['c']
plt.plot(results[:ws + 1], color='#1f76b4')
plt.plot(results[ws:], color='darkgreen')
plt.grid(visible=True, which='both')
plt.show()
exit()


def testStat(timeseries):
    timeseries.dropna(inplace=True)
    rolmean = timeseries.rolling(window=356).mean()
    rolstd = timeseries.rolling(window=365).std()
    orig = plt.plot(timeseries, label='Original')
    mean = plt.plot(rolmean, label='Rolling mean')
    std = plt.plot(rolstd, label='Rolling std')
    plt.legend(loc='best')
    plt.title('Timeseries data with rolling mean and std. dev.')
    plt.show()
    from statsmodels.tsa.stattools import adfuller
    dftest = adfuller(timeseries)
    dfoutput = pd.Series(dftest[0:4],
                         index=['The test statistic', 'Mackinnon\'s approximate p - value', '# usedLags', 'NOBS'])
    print(dfoutput)


testStat(TestData['c'])

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

lag_acf = acf(TestData['c'], nlags=140)
lag_pacf = pacf(TestData['c'], nlags=70)
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
plot_acf(lag_acf, ax=ax[0])
plot_pacf(lag_pacf, lags=20, ax=ax[1])
plt.show()
exit()
