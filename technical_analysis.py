import pandas as pd

from ta.trend import (SMAIndicator, EMAIndicator, MACD, VortexIndicator,
                      TRIXIndicator, MassIndex, DPOIndicator, KSTIndicator,
                      IchimokuIndicator, ADXIndicator, CCIIndicator)
from ta.volatility import (BollingerBands, KeltnerChannel, DonchianChannel,
                           AverageTrueRange)
from ta.momentum import (RSIIndicator, TSIIndicator, StochasticOscillator,
                         WilliamsRIndicator, ROCIndicator,
                         PercentagePriceOscillator, PercentageVolumeOscillator,
                         KAMAIndicator)
from ta.volume import (AccDistIndexIndicator, OnBalanceVolumeIndicator,
                       ChaikinMoneyFlowIndicator, ForceIndexIndicator,
                       EaseOfMovementIndicator, VolumePriceTrendIndicator,
                       VolumeWeightedAveragePrice, MFIIndicator)
from ta.others import (NDaysReturnIndicator)


def add_trend_indicators(data: pd.DataFrame,
                         window: int = 14,
                         fillna: bool = False) -> pd.DataFrame:
    # SMAs
    data["sma"] = SMAIndicator(close=data['close'],
                               window=window,
                               fillna=fillna).sma_indicator()

    # MACD
    indicator_macd = MACD(close=data['close'], fillna=fillna)
    data["macd"] = indicator_macd.macd()
    data["macd_signal"] = indicator_macd.macd_signal()
    data["macd_diff"] = indicator_macd.macd_diff()

    # EMAs
    data["ema"] = EMAIndicator(close=data['close'],
                               window=window,
                               fillna=fillna).ema_indicator()

    # Vortex Indicator
    indicator_vortex = VortexIndicator(high=data['high'],
                                       low=data['low'],
                                       close=data['close'],
                                       window=window,
                                       fillna=fillna)
    data["vortex_pos"] = indicator_vortex.vortex_indicator_pos()
    data["vortex_neg"] = indicator_vortex.vortex_indicator_neg()
    data["vortex_diff"] = indicator_vortex.vortex_indicator_diff()

    # TRIX Indicator
    data["trix"] = TRIXIndicator(close=data['close'],
                                 window=window,
                                 fillna=fillna).trix()

    # Mass Index
    data["mass_index"] = MassIndex(high=data['high'],
                                   low=data['low'],
                                   fillna=fillna).mass_index()

    # DPO Indicator
    data["dpo"] = DPOIndicator(close=data['close'],
                               window=window,
                               fillna=fillna).dpo()

    # KST Indicator
    indicator_kst = KSTIndicator(close=data['close'], fillna=fillna)
    data["kst"] = indicator_kst.kst()
    data["kst_sig"] = indicator_kst.kst_sig()
    data["kst_diff"] = indicator_kst.kst_diff()

    # Ichimoku Indicator
    indicator_ichi = IchimokuIndicator(high=data['high'],
                                       low=data['low'],
                                       fillna=fillna)
    data["ichimoku_a"] = indicator_ichi.ichimoku_a()
    data["ichimoku_b"] = indicator_ichi.ichimoku_b()

    # Average Directional Movement Index (ADX)
    indicator_adx = ADXIndicator(high=data['high'],
                                 low=data['low'],
                                 close=data['close'],
                                 window=window,
                                 fillna=fillna)
    data["adx"] = indicator_adx.adx()
    data["adx_pos"] = indicator_adx.adx_pos()
    data["adx_neg"] = indicator_adx.adx_neg()

    # CCI Indicator
    data["cci"] = CCIIndicator(high=data['high'],
                               low=data['low'],
                               close=data['close'],
                               window=window,
                               fillna=fillna).cci()

    return data


# Volatility
def add_volatility_indicators(data: pd.DataFrame,
                              window: int = 14,
                              fillna: bool = False) -> pd.DataFrame:
    # Bollinger Bands
    indicator_bb = BollingerBands(close=data['close'],
                                  window=window,
                                  window_dev=2,
                                  fillna=fillna)
    data["bbh"] = indicator_bb.bollinger_hband()
    data["bbl"] = indicator_bb.bollinger_lband()
    data["bbp"] = indicator_bb.bollinger_pband()

    # Keltner Channel
    indicator_kc = KeltnerChannel(close=data['close'],
                                  high=data['high'],
                                  low=data['low'],
                                  window=window,
                                  fillna=fillna)

    data["kch"] = indicator_kc.keltner_channel_hband()
    data["kcl"] = indicator_kc.keltner_channel_lband()
    data["kcp"] = indicator_kc.keltner_channel_pband()

    # Donchian Channel
    indicator_dc = DonchianChannel(high=data['high'],
                                   low=data['low'],
                                   close=data['close'],
                                   window=window,
                                   fillna=fillna)
    data["dcl"] = indicator_dc.donchian_channel_lband()
    data["dch"] = indicator_dc.donchian_channel_hband()
    data["dcp"] = indicator_dc.donchian_channel_pband()

    # Average True Range
    data["atr"] = AverageTrueRange(close=data['close'],
                                   high=data['high'],
                                   low=data['low'],
                                   window=window,
                                   fillna=fillna).average_true_range()

    return data


def add_momentum_indicators(data: pd.DataFrame,
                            window: int = 14,
                            fillna: bool = False) -> pd.DataFrame:
    # Relative Strength Index (RSI)
    data["rsi"] = RSIIndicator(close=data['close'],
                               window=window,
                               fillna=fillna).rsi()

    # TSI Indicator
    data["tsi"] = TSIIndicator(close=data['close'], fillna=fillna).tsi()

    # Stoch Indicator
    indicator_so = StochasticOscillator(
        high=data['high'],
        low=data['low'],
        close=data['close'],
        window=window,
        fillna=fillna,
    )
    data["stoch"] = indicator_so.stoch()
    data["stoch_signal"] = indicator_so.stoch_signal()

    # Williams R Indicator
    data["wr"] = WilliamsRIndicator(high=data['high'],
                                    low=data['low'],
                                    close=data['close'],
                                    lbp=window,
                                    fillna=fillna).williams_r()

    # Rate Of Change
    data["roc"] = ROCIndicator(close=data['close'],
                               window=window,
                               fillna=fillna).roc()

    # Percentage Price Oscillator
    indicator_ppo = PercentagePriceOscillator(close=data['close'],
                                              fillna=fillna)
    data["ppo"] = indicator_ppo.ppo()
    data["ppo_signal"] = indicator_ppo.ppo_signal()
    data["ppo_hist"] = indicator_ppo.ppo_hist()

    # Percentage Volume Oscillator
    indicator_pvo = PercentageVolumeOscillator(volume=data['volume'],
                                               fillna=fillna)
    data["pvo"] = indicator_pvo.pvo()
    data["pvo_signal"] = indicator_pvo.pvo_signal()
    data["pvo_hist"] = indicator_pvo.pvo_hist()

    # KAMA
    data["kama"] = KAMAIndicator(close=data['close'],
                                 window=10,
                                 pow1=2,
                                 pow2=30,
                                 fillna=fillna).kama()

    return data


def add_volume_indicators(data: pd.DataFrame,
                          window: int = 14,
                          fillna: bool = False) -> pd.DataFrame:
    # Accumulation Distribution Index
    data["adi"] = AccDistIndexIndicator(high=data['high'],
                                        low=data['low'],
                                        close=data['close'],
                                        volume=data['volume'],
                                        fillna=fillna).acc_dist_index()

    # On Balance Volume
    data["obv"] = OnBalanceVolumeIndicator(close=data['close'],
                                           volume=data['volume'],
                                           fillna=fillna).on_balance_volume()

    # Chaikin Money Flow
    data["cmf"] = ChaikinMoneyFlowIndicator(high=data['high'],
                                            low=data['low'],
                                            close=data['close'],
                                            volume=data['volume'],
                                            fillna=fillna).chaikin_money_flow()

    # Force Index
    data["fi"] = ForceIndexIndicator(close=data['close'],
                                     volume=data['volume'],
                                     window=window,
                                     fillna=fillna).force_index()

    # Ease of Movement
    indicator_eom = EaseOfMovementIndicator(high=data['high'],
                                            low=data['low'],
                                            volume=data['volume'],
                                            window=window,
                                            fillna=fillna)
    data["em"] = indicator_eom.ease_of_movement()
    data["sma_em"] = indicator_eom.sma_ease_of_movement()

    # Volume Price Trend
    data["vpt"] = VolumePriceTrendIndicator(close=data['close'],
                                            volume=data['volume'],
                                            fillna=fillna).volume_price_trend()

    # Volume Weighted Average Price
    data["vwap"] = VolumeWeightedAveragePrice(
        high=data['high'],
        low=data['low'],
        close=data['close'],
        volume=data['volume'],
        window=window,
        fillna=fillna).volume_weighted_average_price()

    # Money Flow Indicator
    data["mfi"] = MFIIndicator(high=data['high'],
                               low=data['low'],
                               close=data['close'],
                               volume=data['volume'],
                               window=window,
                               fillna=fillna).money_flow_index()

    return data


# def add_other_indicators(data: pd.DataFrame,
#                          n_days: int = 1,
#                          fillna: bool = False) -> pd.DataFrame:
#
#     data["n_days_returns"] = NDaysReturnIndicator(
#         close=data['close'], n_days=n_days, fillna=fillna).n_days_return()
#
#     return data


def get_ta_indicators(data: pd.DataFrame,
                      window: int = 14,
                      fillna: bool = False,
                      normalize: bool = False) -> pd.DataFrame:
    """
    Get the results from technical indicators.

    :param data: The stock prices high, low, prices
    :param n_days: n days return
    :return: A dataframe concatenated the results of technical indicators.
    """

    df = data.copy()
    df = add_trend_indicators(df, window, fillna)
    df = add_volatility_indicators(df, window, fillna)

    df = add_volume_indicators(df, window, fillna)
    # df = add_other_indicators(df, n_days, fillna)

    if normalize:
        df = (df - df.mean()) / df.std()

    return df


if __name__ == "__main__":
    from stock_dataset import StockDataset

    # Stock raw_data config
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    period = 14

    stock_config = {
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
    }
    source = StockDataset(**stock_config)

    data = get_ta_indicators(source.data, period)
    print(data)
