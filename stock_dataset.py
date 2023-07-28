import pandas as pd
import yfinance as yf
from pathlib import Path
import datetime as dt
from sklearn.preprocessing import StandardScaler
from utils import get_technical_indicators, build_time_series_data

class StockDataset:

    def __init__(self,
                 symbol,
                 start_date,
                 end_date,
                 future,
                 lookback,
                 use_technical_indicator,
                 save_data_path,
                 load_data_path,
                 verbose
                 ):
        self.end_date = end_date if end_date else dt.date.today().strftime('%Y-%m-%d')
        self.verbose = verbose

        self.data = self.get_data(symbol, start_date, self.end_date, use_technical_indicator, save_data_path, load_data_path)
        self.features, self.target = self.preprocessing(self.data, future, lookback)


    def get_data(self, symbol, start_date, end_date, use_technical_indicator, save_data_path, load_data_path) -> pd.DataFrame:
        if load_data_path:
            if Path(load_data_path).exists():
                data = pd.read_csv(str(load_data_path)).loc[start_date:].dropna()

        else:
            print('Downloading data...')
            data = yf.download(tickers=symbol,
                               start=start_date,
                               end=end_date).dropna().sort_index().astype(float)

            # Save downloaded data as csv
            save_dir = Path(save_data_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            data.to_csv(save_data_path)
            print(f'\n===> Save dataset to {save_data_path}\n')

        data.rename(columns=str.lower, inplace=True)
        data = data[list(data.columns[:5])]

        # Get the technical analysis features
        if use_technical_indicator:
            data = get_technical_indicators(data, fillna=True)
            data = data.drop(columns=['high', 'low', 'close', 'adj close', 'volume'])

        if self.verbose:
            print(f'Shape of dataset: {data.shape}')

        return data


    def preprocessing(self, data, n_future, n_past):
        # Feature Scaling
        sc = StandardScaler()
        scaled_train = sc.fit_transform(data.values)

        self.sc_target = StandardScaler()
        self.sc_target.fit_transform(data.values[:, 0:1])

        # Convert to the LSTM data shape
        X, y = build_time_series_data(scaled_train, n_future, n_past)

        if self.verbose:
            print(f'Shape of x: {X.shape}')
            print(f'Shape of y: {y.shape}')

        return X, y
