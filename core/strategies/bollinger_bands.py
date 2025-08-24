"""
Estratégias baseadas em Bollinger Bands
"""
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands, average_true_range

def bollinger_breakout_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia de breakout das Bollinger Bands
    """
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    volume_filter = params.get('volume_filter', True)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 1.5)
    target_r_mult = params.get('target_r_mult', 2.0)
    
    df_signals = df.copy()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=bb_period, window_dev=bb_std)
    df_signals['BB_Upper'] = bb.bollinger_hband()
    df_signals['BB_Lower'] = bb.bollinger_lband()
    df_signals['BB_Middle'] = bb.bollinger_mavg()
    df_signals['BB_Width'] = (df_signals['BB_Upper'] - df_signals['BB_Lower']) / df_signals['BB_Middle']
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Volume confirmation
    if volume_filter and 'Volume' in df.columns:
        df_signals['Volume_MA'] = df['Volume'].rolling(20).mean()
        volume_above_avg = df_signals['Volume'] > df_signals['Volume_MA']
    else:
        volume_above_avg = True
    
    # Breakouts
    upper_breakout = (df['Close'] > df_signals['BB_Upper']) & \
                     (df['Close'].shift(1) <= df_signals['BB_Upper'].shift(1))
    
    lower_breakout = (df['Close'] < df_signals['BB_Lower']) & \
                     (df['Close'].shift(1) >= df_signals['BB_Lower'].shift(1))
    
    # Filtro de volatilidade (bandas expandindo)
    expanding_bands = df_signals['BB_Width'] > df_signals['BB_Width'].shift(1)
    
    # Sinais
    df_signals['signal'] = 0
    df_signals.loc[upper_breakout & volume_above_avg & expanding_bands, 'signal'] = 1
    df_signals.loc[lower_breakout & volume_above_avg & expanding_bands, 'signal'] = -1
    
    # Stops e targets
    df_signals['stop'] = np.nan
    df_signals['target'] = np.nan
    
    buy_mask = df_signals['signal'] == 1
    df_signals.loc[buy_mask, 'stop'] = df_signals.loc[buy_mask, 'BB_Middle']
    df_signals.loc[buy_mask, 'target'] = df_signals.loc[buy_mask, 'Close'] + \
                                         (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    sell_mask = df_signals['signal'] == -1
    df_signals.loc[sell_mask, 'stop'] = df_signals.loc[sell_mask, 'BB_Middle']
    df_signals.loc[sell_mask, 'target'] = df_signals.loc[sell_mask, 'Close'] - \
                                          (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    return df_signals[['signal', 'stop', 'target']].fillna(0)

def bollinger_mean_reversion_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia de mean reversion das Bollinger Bands
    """
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 2.0)
    target_r_mult = params.get('target_r_mult', 1.5)
    
    df_signals = df.copy()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=bb_period, window_dev=bb_std)
    df_signals['BB_Upper'] = bb.bollinger_hband()
    df_signals['BB_Lower'] = bb.bollinger_lband()
    df_signals['BB_Middle'] = bb.bollinger_mavg()
    
    # RSI para confirmação
    from ta.momentum import rsi
    df_signals['RSI'] = rsi(df['Close'], window=rsi_period)
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Toques nas bandas
    touch_lower = df['Close'] <= df_signals['BB_Lower']
    touch_upper = df['Close'] >= df_signals['BB_Upper']
    
    # Confirmação RSI
    rsi_confirms_buy = df_signals['RSI'] < rsi_oversold
    rsi_confirms_sell = df_signals['RSI'] > rsi_overbought
    
    # Sinais (mean reversion)
    df_signals['signal'] = 0
    df_signals.loc[touch_lower & rsi_confirms_buy, 'signal'] = 1
    df_signals.loc[touch_upper & rsi_confirms_sell, 'signal'] = -1
    
    # Stops e targets
    df_signals['stop'] = np.nan
    df_signals['target'] = np.nan
    
    buy_mask = df_signals['signal'] == 1
    df_signals.loc[buy_mask, 'stop'] = df_signals.loc[buy_mask, 'Close'] - \
                                       (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult)
    df_signals.loc[buy_mask, 'target'] = df_signals.loc[buy_mask, 'BB_Middle']
    
    sell_mask = df_signals['signal'] == -1
    df_signals.loc[sell_mask, 'stop'] = df_signals.loc[sell_mask, 'Close'] + \
                                        (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult)
    df_signals.loc[sell_mask, 'target'] = df_signals.loc[sell_mask, 'BB_Middle']
    
    return df_signals[['signal', 'stop', 'target']].fillna(0)

BOLLINGER_BREAKOUT_PARAMS = {
    'bb_period': 20,
    'bb_std': 2.0,
    'volume_filter': True,
    'atr_period': 14,
    'atr_stop_mult': 1.5,
    'target_r_mult': 2.0
}

BOLLINGER_MEAN_REVERSION_PARAMS = {
    'bb_period': 20,
    'bb_std': 2.0,
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'atr_period': 14,
    'atr_stop_mult': 2.0,
    'target_r_mult': 1.5
}

