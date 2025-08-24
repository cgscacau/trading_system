"""
Estratégia baseada no MACD com confirmação por histograma
"""
import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.volatility import average_true_range

def macd_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia MACD 12/26/9 com confirmação por inclinação do histograma
    
    Parâmetros:
        fast_period (int): Período EMA rápida (default: 12)
        slow_period (int): Período EMA lenta (default: 26)
        signal_period (int): Período sinal (default: 9)
        histogram_confirmation (bool): Usar confirmação por histograma (default: True)
        atr_period (int): Período ATR (default: 14)
        atr_stop_mult (float): Multiplicador ATR para stop (default: 2.0)
        target_r_mult (float): Múltiplo R para target (default: 2.0)
    """
    fast_period = params.get('fast_period', 12)
    slow_period = params.get('slow_period', 26)
    signal_period = params.get('signal_period', 9)
    histogram_confirmation = params.get('histogram_confirmation', True)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 2.0)
    target_r_mult = params.get('target_r_mult', 2.0)
    
    df_signals = df.copy()
    
    # Calcular MACD
    macd_indicator = MACD(close=df['Close'], 
                         window_fast=fast_period,
                         window_slow=slow_period, 
                         window_sign=signal_period)
    
    df_signals['MACD'] = macd_indicator.macd()
    df_signals['MACD_Signal'] = macd_indicator.macd_signal()
    df_signals['MACD_Histogram'] = macd_indicator.macd_diff()
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Cruzamentos MACD
    cross_above = (df_signals['MACD'] > df_signals['MACD_Signal']) & \
                  (df_signals['MACD'].shift(1) <= df_signals['MACD_Signal'].shift(1))
    
    cross_below = (df_signals['MACD'] < df_signals['MACD_Signal']) & \
                  (df_signals['MACD'].shift(1) >= df_signals['MACD_Signal'].shift(1))
    
    # Confirmação por inclinação do histograma
    if histogram_confirmation:
        hist_rising = df_signals['MACD_Histogram'] > df_signals['MACD_Histogram'].shift(1)
        hist_falling = df_signals['MACD_Histogram'] < df_signals['MACD_Histogram'].shift(1)
        
        buy_condition = cross_above & hist_rising
        sell_condition = cross_below & hist_falling
    else:
        buy_condition = cross_above
        sell_condition = cross_below
    
    # Gerar sinais
    df_signals['signal'] = 0
    df_signals.loc[buy_condition, 'signal'] = 1
    df_signals.loc[sell_condition, 'signal'] = -1
    
    # Stops e targets
    df_signals['stop'] = np.nan
    df_signals['target'] = np.nan
    
    buy_mask = df_signals['signal'] == 1
    df_signals.loc[buy_mask, 'stop'] = df_signals.loc[buy_mask, 'Close'] - \
                                       (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult)
    df_signals.loc[buy_mask, 'target'] = df_signals.loc[buy_mask, 'Close'] + \
                                         (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    sell_mask = df_signals['signal'] == -1
    df_signals.loc[sell_mask, 'stop'] = df_signals.loc[sell_mask, 'Close'] + \
                                        (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult)
    df_signals.loc[sell_mask, 'target'] = df_signals.loc[sell_mask, 'Close'] - \
                                          (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    return df_signals[['signal', 'stop', 'target']].fillna(0)

MACD_PARAMS = {
    'fast_period': 12,
    'slow_period': 26,
    'signal_period': 9,
    'histogram_confirmation': True,
    'atr_period': 14,
    'atr_stop_mult': 2.0,
    'target_r_mult': 2.0
}

