import sys

import absl.logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import US
import sqlconnect

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
absl.logging.set_verbosity(absl.logging.ERROR)
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
print(tf.__version__)

df, df1 = sqlconnect.getdatafromsql()
df.drop('DT', axis=1)
df.columns.drop('DT')
drivers = ['wf', 'mf', 'sf', 'IHT', 'wfy', 'dhdd1', 'hdd1t', 'Troloff', 'Troloff2', 'mx2', 'mn4', 'dcy', 'c7', 'cy',
           'c']
df['Troloff2'] = df['Troloff'] ** 2

TestData = df.loc[:, drivers].copy()
print(TestData)
TestData = TestData.dropna()
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
print(TestData.tail())

train_dataset = TestData.sample(frac=0.9, random_state=0)
test_dataset = TestData.drop(train_dataset.index)

print(train_dataset.describe().transpose())

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('c')
test_labels = test_features.pop('c')

## Normalization

normalizer = tf.keras.layers.Normalization(axis=-1)

"""Then, fit the state of the preprocessing layer to the data by calling `Normalization.adapt`:"""
normalizer.adapt(np.array(train_features))

normalizer.mean.numpy()


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [C]')
    plt.legend()
    plt.grid(True)
    plt.show()


### Linear regression with multiple inputs


linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

print(linear_model.predict(train_features[:10]))

print(linear_model.layers[1].kernel)

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error')

# Commented out IPython magic to ensure Python compatibility.
# %%time
history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)

plot_loss(history)
test_results = {
    'linear_model': linear_model.evaluate(test_features, test_labels, verbose=0) * means['c'][1] + means['c'][0]}


## Regression with a deep neural network (DNN)

def plot_horsepower(x, y):
    plt.scatter(train_features['Troloff'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('T')
    plt.ylabel('C')
    plt.legend()
    plt.show()


def build_and_compile_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dnn_horsepower_model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

dnn_horsepower_model.compile(loss='mean_squared_error',
                             optimizer=tf.keras.optimizers.Adam(0.001))

# Commented out IPython magic to ensure Python compatibility.
# %%time
history = dnn_horsepower_model.fit(
    train_features['Troloff'],
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

x = tf.linspace(0.0, 1, 1)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x, y)
test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Troloff'], test_labels,
    verbose=0) * means['c'][1] + means['c'][0]

dnn_model = build_and_compile_model()

# Commented out IPython magic to ensure Python compatibility.
# %%time
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0) * means['c'][1] + means['c'][0]

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [C]')
plt.ylabel('Predictions [C]')
# lims = [0, 50]
# plt.xlim(lims)
# plt.ylim(lims)
# _ = plt.plot(lims, lims)
plt.show()
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [C]')
_ = plt.ylabel('Count')
plt.show()
print(pd.DataFrame(test_results, index=['Mean absolute error [C]']).T)
