# Stock Price Prediction in Tensorflow
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) designed to capture long-range dependencies in sequential data. Its ability to retain and forget information over time makes it ideal for time-series forecasting.
In this project, we use LSTM to build an Artificial Neural Network (ANN) for predicting stock prices by considering multiple 
forecasting factors (features). The final stock price prediction can be evaluating the 
significance of historical low prices, high prices, trading volumes, adjusted prices, or technical indicators.

# How to use

## Install packages
Ensure all dependencies are installed using:

`pip install -r requirements.txt`


## Dataset
To download the historical price for a specific stock, we'll be using`yfinance` library, which provide access to Yahoo Finance API.

## Train weights

You can train your model by running:

```python
python main.py --train \
               --symbol \     # stock symbol
               --sd \         # start date
               --ed \         # end date
               --lookback \   # The number of days to used for prediction.
               --future \     # the number of days to predict in the future
               --save-dir \   # The path of dir to save the weights and plots
               --epochs \     
               --batch-size \
               --neurons \    # tuple i.e. (256, 128)
               --dropout \    # dropout rate
               --lr           # learning rate
```
You can use `--ta` to use technical indicators as your features. The normalized high, low, close, adjust close are used as the 
features by default.

The downloaded historical price data will be saved in the `data/` folder.

If you already have the data in CSV format, you can provide the path using the `--data` flag.

## Predict price

After training, it will automatically execute predictions.
If you wish to make price predictions manually, you can use the following command:

`python main.py --predict --weights [path of weights]`

Default path of weights is `runs/weights.h5`. The plot of prediction result will be saved in folder `runs/` by default. 
