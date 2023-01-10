import math
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import sqlconnect

ws = 7

df, df1 = sqlconnect.getdatafromsqlols()
# df['Troloff2'] = df['Troloff'] ** 2
# df1['Troloff2'] = df1['Troloff'] ** 2
drivers = df.columns.tolist()
drivers.remove('hdd')
drivers.remove('Date')

TestData = df.loc[:, drivers]
ForecastData = pd.concat([TestData[len(TestData) - ws:], df1.loc[:, drivers]])

columns = TestData.columns.tolist()

model = LinearRegression()
X = TestData[[c for c in columns if c not in ['c']]]
y = TestData['c']

model.fit(X, y)
print(model.score(X, y))
print(model.coef_)
print(model.intercept_)
print(model.score(X, y))
for r in range(ws, len(ForecastData)):
    if math.isnan(float(ForecastData['c7'][r])): ForecastData['c7'][r] = ForecastData['c'][r - 7]
    if math.isnan(float(ForecastData['cy'][r])): ForecastData['cy'][r] = ForecastData['c'][r - 1]
    dcy = ForecastData['cy'][r] / ForecastData['cy'][r - 1]
    ForecastData['dcy'][r] = 0 if dcy < 1.3 else dcy
    # if math.isnan(float(ForecastData['c'][r])):
    X1 = ForecastData[[c for c in columns if c not in ['c']]][r - 1:r]
    yP = model.predict(X1)
    ForecastData['c'][r] = yP[len(yP) - 1]

# print(ForecastData)
results = ForecastData['c']  # * means['c'][1] + means['c'][0]
print(results)
plt.plot(results[:ws + 1], color='#1f76b4')
plt.plot(results[ws:], color='darkgreen')
plt.grid(visible=True, which='both')
plt.show()

exit(0)
