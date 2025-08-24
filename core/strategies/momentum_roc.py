"""
Estratégias baseadas em Momentum e Rate of Change
"""
import pandas as pd
import numpy as np
from ta.volatility import average_true_range
from ta.momentum import roc

def momentum_roc_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia Momentum/Rate of Change com cruzamento do zero
    """
    roc_period = params.get('roc_period', 12)
    threshold = params.get('threshold', 0.0)
    trend_filter_period = params.get('trend_filter_period', 50)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 2.0)
    target_r_mult = params.get('target_r_mult', 2.0)
    
    df_signals = df.copy()
    
    # Rate of Change
    df_signals['ROC'] = roc(df['Close'], window=roc_period)
    
    # Filtro de tendência
    from ta.trend import sma_indicator
    df_signals['SMA_Trend'] = sma_indicator(df['Close'], window=trend_filter_period)
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Condições de tendência
    uptrend = df['Close'] > df_signals['SMA_Trend']
    downtrend = df['Close'] < df_signals['SMA_Trend']
    
    # Cruzamentos do threshold
    cross_above = (df_signals['ROC'] > threshold) & (df_signals['ROC'].shift(1) <= threshold)
    cross_below = (df_signals['ROC'] < threshold) & (df_signals['ROC'].shift(1) >= threshold)
    
    # Sinais
    df_signals['signal'] = 0
    df_signals.loc[cross_above & uptrend, 'signal'] = 1
    df_signals.loc[cross_below & downtrend, 'signal'] = -1
    
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

MOMENTUM_ROC_PARAMS = {
    'roc_period': 12,
    'threshold': 0.0,
    'trend_filter_period': 50,
    'atr_period': 14,
    'atr_stop_mult': 2.0,
    'target_r_mult': 2.0
}

