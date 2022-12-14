from ast import Str

import pandas as pd
from keras.losses import mse
from matplotlib import pyplot as plt


def plot_predictions1(model, X, y, label1, label2, title, start=0, end=100):
    predictions = model.predict(X).flatten()
    err = mse(y, predictions)
    df = pd.DataFrame(data={'Predictions': predictions, 'Actuals': y})
    print(df)
    plt.plot(df['Predictions'][start:end], label=label1)
    plt.plot(df['Actuals'][start:end], label=label2)
    plt.legend()
    plt.title(f'{title}, mse={err}')
    plt.show()
    return df, err
