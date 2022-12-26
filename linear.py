import numpy as np
from sklearn.linear_model import LinearRegression

import sqlconnect

df, df1 = sqlconnect.getdatafromsql()
df['Troloff2'] = df['Troloff'] ** 2
df1['Troloff2'] = df1['Troloff'] ** 2

# Get all the columns from the dataframe.
columns = df.columns.tolist()

# Store the variable well be predicting on.
target = "c"

# Initialize the model class.
lin_model = LinearRegression()
X = df[[c for c in columns if c not in [target, 'DT']]]
y = df[target]
# Fit the model to the training data.
lin_model.fit(X, y)
print(lin_model.score(X, y))
print(lin_model.coef_)
print(lin_model.intercept_)