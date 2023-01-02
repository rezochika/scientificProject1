import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'Data\Apkhazia2021.csv')
df['DT'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
df.index = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
df.columns.drop('Date')
# df['Value'].plot()
values = df['Value']
timestamps = df['DT']

ts = pd.Series(values, index=timestamps)
lenT = len(ts)
ts = ts.resample('H').mean()
tsinter = ts.interpolate(method='spline', order=5)
print(tsinter)
tsinter.plot()
# print(ts.interpolate(method='time'))
# ts.interpolate(method='time', order=len(df)-1).plot()
lines, labels = plt.gca().get_legend_handles_labels()
labels = ['spline', 'time']
plt.legend(lines, labels, loc='best')
plt.show()
tsinter.to_csv(r'Data\SplineAp.csv')
