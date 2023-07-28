import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

from technical_analysis import add_trend_indicators, add_volatility_indicators, add_momentum_indicators, add_volume_indicators


def normalization(data):
    return (data - data.mean()) / data.std()


def get_technical_indicators(df: pd.DataFrame,
                      window: int = 20,
                      fillna: bool = False,
                      normalize: bool = False) -> pd.DataFrame:
    """
    Get the results from technical indicators.

    :param df: The stock prices high, low, prices
    :param window: The lookback period
    :return: A dataframe concatenated the results of technical indicators.
    """

    df = df.copy()
    df = add_trend_indicators(df, window, fillna)
    df = add_volatility_indicators(df, window, fillna)
    df = add_momentum_indicators(df, window, fillna)
    df = add_volume_indicators(df, window, fillna)

    if normalize:
        df = normalization(df)

    return df


def split_train_test_data(X, y):
    train_size = int(X.shape[0] * 0.8)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    return X_train, X_test, y_train, y_test


def build_time_series_data(data, n_future:int=60, n_past:int=90):
    """
    Build supervised timeseries data. For each timestep, we use n_past days as feature and use n_past + i + n_future day as the corresponding label, which is the open price.
    The dimension should be (num_samples - window size, window size, num_features).
    Window size is the number of time-steps.


    Args:
        data: Training data
        n_future: Number of days we want to predict into the future
        n_past: Number of past days we want to use to predict the future

    Returns: Time-series data

    """

    # Creating a data structure with 90 timestamps and 1 output
    X = []
    y = []

    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, :data.shape[1]])
        y.append(data[i + n_future - 1:i + n_future, 0])

    X, y = np.array(X), np.array(y)

    return X, y


def datetime_to_timestamp(x):
    '''
        x : a given datetime value (datetime.date)
    '''
    return dt.datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


def plot_result(data, title, xlabel, ylabel, fname):
    plt.figure()
    plt.plot(data)
    predict_start_date = data['Predicted Stock Price'].first_valid_index()
    plt.axvline(x=predict_start_date, color='red', linewidth=1, linestyle='--')
    plt.legend(data.columns.tolist(), loc='best')
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(fname)
    print(f'\n===> Save prediction result image to {fname}\n')


