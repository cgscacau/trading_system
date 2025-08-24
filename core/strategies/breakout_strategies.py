"""
Estratégias de breakout de máximas/mínimas
"""
import pandas as pd
import numpy as np
from ta.volatility import average_true_range
from ta.trend import sma_indicator

def high_low_breakout_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia de breakout de máximas/mínimas com pullback opcional
    """
    lookback_period = params.get('lookback_period', 20)
    pullback_confirmation = params.get('pullback_confirmation', False)
    pullback_pct = params.get('pullback_pct', 0.02)
    volume_filter = params.get('volume_filter', True)
    trend_filter = params.get('trend_filter', True)
    trend_period = params.get('trend_period', 200)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 2.0)
    target_r_mult = params.get('target_r_mult', 2.0)
    
    df_signals = df.copy()
    
    # Máximas e mínimas do período
    df_signals['Period_High'] = df['High'].rolling(lookback_period).max()
    df_signals['Period_Low'] = df['Low'].rolling(lookback_period).min()
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Filtro de tendência
    if trend_filter:
        df_signals['SMA_Trend'] = sma_indicator(df['Close'], window=trend_period)
        uptrend = df['Close'] > df_signals['SMA_Trend']
        downtrend = df['Close'] < df_signals['SMA_Trend']
    else:
        uptrend = True
        downtrend = True
    
    # Volume filter
    if volume_filter and 'Volume' in df.columns:
        df_signals['Volume_MA'] = df['Volume'].rolling(20).mean()
        volume_condition = df_signals['Volume'] > df_signals['Volume_MA']
    else:
        volume_condition = True
    
    # Breakouts básicos
    high_breakout = (df['High'] > df_signals['Period_High'].shift(1)) & volume_condition & uptrend
    low_breakout = (df['Low'] < df_signals['Period_Low'].shift(1)) & volume_condition & downtrend
    
    if pullback_confirmation:
        # Aguardar pullback após breakout
        recent_high = df['High'].rolling(3).max()
        recent_low = df['Low'].rolling(3).min()
        
        pullback_after_high = (df['Close'] < recent_high.shift(1) * (1 - pullback_pct))
        pullback_after_low = (df['Close'] > recent_low.shift(1) * (1 + pullback_pct))
        
        buy_condition = high_breakout & pullback_after_high
        sell_condition = low_breakout & pullback_after_low
    else:
        buy_condition = high_breakout
        sell_condition = low_breakout
    
    # Sinais
    df_signals['signal'] = 0
    df_signals.loc[buy_condition, 'signal'] = 1
    df_signals.loc[sell_condition, 'signal'] = -1
    
    # Stops e targets
    df_signals['stop'] = np.nan
    df_signals['target'] = np.nan
    
    buy_mask = df_signals['signal'] == 1
    df_signals.loc[buy_mask, 'stop'] = np.minimum(
        df_signals.loc[buy_mask, 'Period_Low'].shift(1),
        df_signals.loc[buy_mask, 'Close'] - (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult)
    )
    df_signals.loc[buy_mask, 'target'] = df_signals.loc[buy_mask, 'Close'] + \
                                         (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    sell_mask = df_signals['signal'] == -1
    df_signals.loc[sell_mask, 'stop'] = np.maximum(
        df_signals.loc[sell_mask, 'Period_High'].shift(1),
        df_signals.loc[sell_mask, 'Close'] + (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult)
    )
    df_signals.loc[sell_mask, 'target'] = df_signals.loc[sell_mask, 'Close'] - \
                                          (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    return df_signals[['signal', 'stop', 'target']].fillna(0)

HIGH_LOW_BREAKOUT_PARAMS = {
    'lookback_period': 20,
    'pullback_confirmation': False,
    'pullback_pct': 0.02,
    'volume_filter': True,
    'trend_filter': True,
    'trend_period': 200,
    'atr_period': 14,
    'atr_stop_mult': 2.0,
    'target_r_mult': 2.0
}

