import argparse
import sys
from pathlib import Path
import pandas as pd
from stock_dataset import StockDataset
from lstm_model import LSTMModel
from utils import plot_result

ROOT = Path().absolute()  # Current working directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

def parse_opt():
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--sd', type=str, default=None, help='Start date')
    parser.add_argument('--ed', type=str, default=None, help='End date')
    parser.add_argument('--data', type=str, default=None, help='Path of csv data')
    parser.add_argument('--lookback', type=float, default=90, help='The number of past days used for predicting the future price')
    parser.add_argument('--future', type=int, default=60, help='The future days to predict')
    parser.add_argument('--ta', action='store_true', help='Enable to use technical indicators as features')
    parser.add_argument('--save-dir', type=str, default=ROOT / 'runs', help='The directory to save the results')

    parser.add_argument('--train', action='store_true', help='Train new weights')
    parser.add_argument('--predict', action='store_true', help='Load weights from path for prediction')
    parser.add_argument('--weights', type=str, default=ROOT / 'runs/weights.h5', help='The path of weights')

    parser.add_argument('--epochs', type=int, default=70, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=64, help='The batch size')
    parser.add_argument('--neurons', type=tuple, default=(256, 128), help='(256, 128)')
    parser.add_argument('--dropout', type=float, default=0.25, help='The dropout rate for LSTM runs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--verbose', action='store_true', help='For debugging')

    return parser.parse_args()

def train(model, data, epochs, batch_size, save_path):
    history = model.train(data.features, data.target, epochs, batch_size, save_path)
    print(f"Training loss: {history.history['loss'][0]}")
    print(f"Validation loss: {history.history['val_loss'][0]}")


def prediction(model, weights, data, future, lookback, save_dir):
    model.load_model(weights)

    # Make predictions for future dates
    predictions_future = model.model.predict(data.features[-future:])
    predictions_train = model.model.predict(data.features[lookback:])

    # Unscaling
    y_pred_future = data.sc_target.inverse_transform(predictions_future)
    y_pred_train = data.sc_target.inverse_transform(predictions_train)

    # Plot result
    date_list = data.data.index
    date_list_future = pd.date_range(data.end_date, periods=future, freq='1d')

    plot_from_date = '2012-06-01'
    predictions_future_df = pd.DataFrame(y_pred_future, columns=['Predicted Stock Price'], index=date_list_future)
    predictions_train_df = pd.DataFrame(y_pred_train, columns=['Training predictions'],
                                        index=date_list[2 * lookback + future - 1:])

    df = pd.concat([predictions_train_df, predictions_future_df])
    df['Actual Stock Price'] = data.data['open']
    df = df[df.columns[::-1]]  # just reverse
    df = df.loc[plot_from_date:]

    title = 'Predictions and Actual Stock Prices - OHLC'
    xlabel = 'Timeline'
    ylabel = 'Stock Price Value'
    fname = save_dir / 'ta_result.jpg'
    plot_result(df, title, xlabel, ylabel, fname)


def main(opt):
    # Configuration
    symbol = opt.symbol.upper()
    start_date = opt.sd
    end_date = opt.ed
    load_data_path = opt.data
    future = opt.future
    lookback = opt.lookback
    weights = str(opt.weights)
    use_ta = opt.ta if opt.ta else None
    save_dir = opt.save_dir
    verbose = 1 if opt.verbose else 0

    # Hyperparameters
    epochs = opt.epochs
    batch_size = opt.batch_size
    neurons = opt.neurons
    dropout_rate = opt.dropout
    learning_rate = opt.lr
    is_train = opt.train
    is_predict = opt.predict

    if verbose:
        print(f'\nArguments:\n{vars(opt)}')

    # Get stock data
    save_data_path = Path(save_dir).parent / f'data/{symbol}.csv'
    stock_data = StockDataset(symbol, start_date, end_date, future, lookback, use_ta, save_data_path, load_data_path, verbose)

    # Build model
    lstm = LSTMModel(lookback, neurons[0], neurons[1], stock_data.data.shape[1], dropout_rate, learning_rate)

    # Train and/or test the model
    if is_train:
        save_weights_path = save_dir / 'ta_weights.h5'  # Path to save the trained weights
        train(lstm, stock_data, epochs, batch_size, save_weights_path)
        weights = save_weights_path
        is_train = False

    if is_predict or not is_train:
        prediction(lstm, weights, stock_data, future, lookback, save_dir)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
