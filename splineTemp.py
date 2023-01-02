import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'Data\TimeTemp.csv.csv')
df.index = pd.to_datetime(df['Date'], format='%m.%d.%Y %H:%m')
print(df)
values = [271238, 329285, -1, 260260, 263711]
timestamps = pd.to_datetime(['2015-01-04 08:29:05',
                             '2015-01-04 08:34:05',
                             '2015-01-04 08:39:05',
                             '2015-01-04 08:44:05',
                             '2015-01-04 08:49:05'])

ts = pd.Series(values, index=timestamps)
ts[ts == -1] = np.nan
ts = ts.resample('T').mean()

ts.interpolate(method='spline', order=3).plot()
ts.interpolate(method='time').plot()
lines, labels = plt.gca().get_legend_handles_labels()
labels = ['spline', 'time']
plt.legend(lines, labels, loc='best')
plt.show()
