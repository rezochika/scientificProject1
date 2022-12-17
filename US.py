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


def dftoXy1(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    ln = len(df_as_np[0])
    for i in range(len(df_as_np) - window_size):
        row = [r for r in df_as_np[i + 1:i + window_size + 1]]
        rr = []
        for r in row:
            rr.append(np.delete(r, ln - 1))
        X.append(rr)
        label = df_as_np[i + window_size][ln - 1]
        y.append(label)
    return np.array(X), np.array(y)


def plot_predictions1(model, X, y, label1, label2, title, start=0, end=100):
    predictions = model.predict(X).flatten()
    err = mse(y, predictions)
    df = pd.DataFrame(data={'Predictions': predictions, 'Actuals': y})
    # plt.plot(df['Predictions'][start:end], label=label1)
    # plt.plot(df['Actuals'][start:end], label=label2)
    # plt.legend()
    # plt.title(f'{title}, mse={err}')
    # plt.show()
    return df, err


def normalizeArray(X):
    m = np.mean(X)
    s = np.std(X)
    X = (X - m) / s
    return X, m, s


def plot_predictions(model, X, y, label1, label2, title, mean, std,lr, ep, ws, start=0, end=100):
    predictions = model.predict(X).flatten()
    predictions = predictions * std + mean
    y = y * std + mean
    err = float(mse(y, predictions))
    mape = MAPE(y, predictions)
    df = pd.DataFrame(data={'Predictions': predictions, 'Actuals': y})
    plt.plot(df['Predictions'][start:end], label=label1)
    plt.plot(df['Actuals'][start:end], label=label2)
    plt.legend()
    plt.title(f'{title}, mape={mape:.2%}, lr: {lr:.5f}, ep: {ep}, ws: {ws}')
    plt.show()
    return df, err, mape


def preprocess(X, mean, std):
    X[:, :, 0] = (X[:, :, 0] - mean) / std
    return X


def MAPE(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted) / Y_actual))
    return mape


def preprocessOne(X, mean, std):
    X = (X - mean) / std
    return X
