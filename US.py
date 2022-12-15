import numpy as np
import pandas as pd
from keras.losses import mse
from matplotlib import pyplot as plt


def dftoXy(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)


def plot_predictions1(model, X, y, label1, label2, title, start=0, end=100):
    predictions = model.predict(X).flatten()
    err = mse(y, predictions)
    df = pd.DataFrame(data={'Predictions': predictions, 'Actuals': y})
    plt.plot(df['Predictions'][start:end], label=label1)
    plt.plot(df['Actuals'][start:end], label=label2)
    plt.legend()
    plt.title(f'{title}, mse={err}')
    plt.show()
    return df, err


def plot_predictions(model, X, y, label1, label2, title, mean, std, start=0, end=100):
    predictions = model.predict(X).flatten()
    predictions = predictions * std + mean
    y = y * std + mean
    err = mse(y, predictions)
    mape = MAPE(y, predictions)
    df = pd.DataFrame(data={'Predictions': predictions, 'Actuals': y})
    plt.plot(df['Predictions'][start:end], label=label1)
    plt.plot(df['Actuals'][start:end], label=label2)
    plt.legend()
    plt.title(f'{title}, MAPE={mape}')
    plt.show()
    return df, err


def preprocess(X, mean, std):
    X[:, :, 0] = (X[:, :, 0] - mean) / std
    return X


def MAPE(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted) / Y_actual)) * 100
    return mape


def preprocessOne(X, mean, std):
    X = (X - mean) / std
    return X
