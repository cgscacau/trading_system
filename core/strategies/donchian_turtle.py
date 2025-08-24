"""
Estratégias Donchian/Turtle Trading
"""
import pandas as pd
import numpy as np
from ta.volatility import average_true_range

def donchian_breakout_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia de breakout Donchian (Turtle Trading)
    """
    entry_period = params.get('entry_period', 20)
    exit_period = params.get('exit_period', 10)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 2.0)
    target_r_mult = params.get('target_r_mult', 2.0)
    volume_filter = params.get('volume_filter', False)
    
    df_signals = df.copy()
    
    # Canais Donchian
    df_signals['Donchian_High'] = df['High'].rolling(entry_period).max()
    df_signals['Donchian_Low'] = df['Low'].rolling(entry_period).min()
    df_signals['Exit_High'] = df['High'].rolling(exit_period).max()
    df_signals['Exit_Low'] = df['Low'].rolling(exit_period).min()
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Volume filter (opcional)
    if volume_filter and 'Volume' in df.columns:
        df_signals['Volume_MA'] = df['Volume'].rolling(20).mean()
        volume_condition = df_signals['Volume'] > df_signals['Volume_MA']
    else:
        volume_condition = True
    
    # Breakouts
    upper_breakout = (df['High'] > df_signals['Donchian_High'].shift(1)) & volume_condition
    lower_breakout = (df['Low'] < df_signals['Donchian_Low'].shift(1)) & volume_condition
    
    # Sinais
    df_signals['signal'] = 0
    df_signals.loc[upper_breakout, 'signal'] = 1
    df_signals.loc[lower_breakout, 'signal'] = -1
    
    # Stops e targets
    df_signals['stop'] = np.nan
    df_signals['target'] = np.nan
    
    buy_mask = df_signals['signal'] == 1
    # Stop no canal de saída oposto ou ATR
    df_signals.loc[buy_mask, 'stop'] = np.minimum(
        df_signals.loc[buy_mask, 'Exit_Low'].shift(1),
        df_signals.loc[buy_mask, 'Close'] - (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult)
    )
    df_signals.loc[buy_mask, 'target'] = df_signals.loc[buy_mask, 'Close'] + \
                                         (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    sell_mask = df_signals['signal'] == -1
    df_signals.loc[sell_mask, 'stop'] = np.maximum(
        df_signals.loc[sell_mask, 'Exit_High'].shift(1),
        df_signals.loc[sell_mask, 'Close'] + (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult)
    )
    df_signals.loc[sell_mask, 'target'] = df_signals.loc[sell_mask, 'Close'] - \
                                          (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    return df_signals[['signal', 'stop', 'target']].fillna(0)

DONCHIAN_BREAKOUT_PARAMS = {
    'entry_period': 20,
    'exit_period': 10,
    'atr_period': 14,
    'atr_stop_mult': 2.0,
    'target_r_mult': 2.0,
    'volume_filter': False
}

