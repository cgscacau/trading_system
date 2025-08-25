import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, DonchianChannel

def sma_crossover_strategy(data: pd.DataFrame, short_window: int = 20, long_window: int = 50, **kwargs):
    close_prices = data['Close']
    data['sma_short'] = SMAIndicator(close=close_prices, window=short_window).sma_indicator()
    data['sma_long'] = SMAIndicator(close=close_prices, window=long_window).sma_indicator()
    data['signal'] = 0
    data.loc[data['sma_short'] > data['sma_long'], 'signal'] = 1
    data.loc[data['sma_short'] < data['sma_long'], 'signal'] = -1
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10
    return data

def ema_crossover_strategy(data: pd.DataFrame, short_window: int = 12, long_window: int = 26, **kwargs):
    close_prices = data['Close']
    data['ema_short'] = EMAIndicator(close=close_prices, window=short_window).ema_indicator()
    data['ema_long'] = EMAIndicator(close=close_prices, window=long_window).ema_indicator()
    data['signal'] = 0
    data.loc[data['ema_short'] > data['ema_long'], 'signal'] = 1
    data.loc[data['ema_short'] < data['ema_long'], 'signal'] = -1
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10
    return data

def rsi_strategy(data: pd.DataFrame, window: int = 14, buy_level: int = 30, sell_level: int = 70, **kwargs):
    close_prices = data['Close']
    data['rsi'] = RSIIndicator(close=close_prices, window=window).rsi()
    data['signal'] = 0
    data.loc[data['rsi'] < buy_level, 'signal'] = 1
    data.loc[data['rsi'] > sell_level, 'signal'] = -1
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10
    return data

def macd_strategy(data: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9, **kwargs):
    close_prices = data['Close']
    macd_indicator = MACD(close=close_prices, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
    data['macd_line'] = macd_indicator.macd()
    data['signal_line'] = macd_indicator.macd_signal()
    data['signal'] = 0
    data.loc[data['macd_line'] > data['signal_line'], 'signal'] = 1
    data.loc[data['macd_line'] < data['signal_line'], 'signal'] = -1
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10
    return data

def bollinger_mean_reversion_strategy(data: pd.DataFrame, window: int = 20, window_dev: int = 2, **kwargs):
    close_prices = data['Close']
    bb_indicator = BollingerBands(close=close_prices, window=window, window_dev=window_dev)
    data['bb_low'] = bb_indicator.bollinger_lband()
    data['bb_high'] = bb_indicator.bollinger_hband()
    data['signal'] = 0
    data.loc[data['Close'] < data['bb_low'], 'signal'] = 1
    data.loc[data['Close'] > data['bb_high'], 'signal'] = -1
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10
    return data

def bollinger_breakout_strategy(data: pd.DataFrame, window: int = 20, window_dev: int = 2, **kwargs):
    close_prices = data['Close']
    bb_indicator = BollingerBands(close=close_prices, window=window, window_dev=window_dev)
    data['bb_high'] = bb_indicator.bollinger_hband()
    data['bb_low'] = bb_indicator.bollinger_lband()
    data['signal'] = 0
    data.loc[data['Close'] > data['bb_high'].shift(1), 'signal'] = 1
    data.loc[data['Close'] < data['bb_low'].shift(1), 'signal'] = -1
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10
    return data

def adx_dmi_strategy(data: pd.DataFrame, window: int = 14, adx_threshold: int = 25, **kwargs):
    adx_indicator = ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=window)
    data['adx'] = adx_indicator.adx()
    data['dmi_pos'] = adx_indicator.adx_pos()
    data['dmi_neg'] = adx_indicator.adx_neg()
    data['signal'] = 0
    buy_condition = (data['adx'] > adx_threshold) & (data['dmi_pos'] > data['dmi_neg'])
    data.loc[buy_condition, 'signal'] = 1
    sell_condition = (data['adx'] > adx_threshold) & (data['dmi_neg'] > data['dmi_pos'])
    data.loc[sell_condition, 'signal'] = -1
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10
    return data

def donchian_breakout_strategy(data: pd.DataFrame, window: int = 20, **kwargs):
    donchian_indicator = DonchianChannel(high=data['High'], low=data['Low'], close=data['Close'], window=window)
    data['donchian_high'] = donchian_indicator.donchian_channel_hband()
    data['donchian_low'] = donchian_indicator.donchian_channel_lband()
    data['signal'] = 0
    data.loc[data['Close'] > data['donchian_high'].shift(1), 'signal'] = 1
    data.loc[data['Close'] < data['donchian_low'].shift(1), 'signal'] = -1
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10
    return data
