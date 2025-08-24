"""
Estratégias baseadas no RSI (IFR2 e IFR padrão)
"""
import pandas as pd
import numpy as np
from ta.momentum import rsi
from ta.volatility import average_true_range
from ta.trend import sma_indicator

def rsi_ifr2_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia IFR2 (Larry Connors) - Mean Reversion
    
    Parâmetros:
        rsi_period (int): Período do RSI (default: 2)
        oversold_level (float): Nível de sobrevenda (default: 10)
        overbought_level (float): Nível de sobrecompra (default: 90)
        trend_sma_period (int): SMA para filtro de tendência (default: 200)
        atr_period (int): Período ATR (default: 14)
        atr_stop_mult (float): Multiplicador ATR para stop (default: 1.5)
        target_r_mult (float): Múltiplo R para target (default: 1.5)
    """
    rsi_period = params.get('rsi_period', 2)
    oversold_level = params.get('oversold_level', 10)
    overbought_level = params.get('overbought_level', 90)
    trend_sma_period = params.get('trend_sma_period', 200)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 1.5)
    target_r_mult = params.get('target_r_mult', 1.5)
    
    df_signals = df.copy()
    
    # Indicadores
    df_signals['RSI2'] = rsi(df['Close'], window=rsi_period)
    df_signals['SMA_Trend'] = sma_indicator(df['Close'], window=trend_sma_period)
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Filtro de tendência
    trend_up = df_signals['Close'] > df_signals['SMA_Trend']
    trend_down = df_signals['Close'] < df_signals['SMA_Trend']
    
    # Condições de entrada
    buy_condition = (df_signals['RSI2'] < oversold_level) & trend_up
    sell_condition = (df_signals['RSI2'] > overbought_level) & trend_down
    
    # Sinais
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

def rsi_standard_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia RSI padrão (30/70)
    
    Parâmetros:
        rsi_period (int): Período do RSI (default: 14)
        oversold_level (float): Nível de sobrevenda (default: 30)
        overbought_level (float): Nível de sobrecompra (default: 70)
        atr_period (int): Período ATR (default: 14)
        atr_stop_mult (float): Multiplicador ATR para stop (default: 2.0)
        target_r_mult (float): Múltiplo R para target (default: 2.0)
    """
    rsi_period = params.get('rsi_period', 14)
    oversold_level = params.get('oversold_level', 30)
    overbought_level = params.get('overbought_level', 70)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 2.0)
    target_r_mult = params.get('target_r_mult', 2.0)
    
    df_signals = df.copy()
    
    # Indicadores
    df_signals['RSI'] = rsi(df['Close'], window=rsi_period)
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Cruzamentos dos níveis
    cross_above_oversold = (df_signals['RSI'] > oversold_level) & \
                          (df_signals['RSI'].shift(1) <= oversold_level)
    
    cross_below_overbought = (df_signals['RSI'] < overbought_level) & \
                            (df_signals['RSI'].shift(1) >= overbought_level)
    
    # Sinais
    df_signals['signal'] = 0
    df_signals.loc[cross_above_oversold, 'signal'] = 1
    df_signals.loc[cross_below_overbought, 'signal'] = -1
    
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

# Parâmetros padrão
RSI_IFR2_PARAMS = {
    'rsi_period': 2,
    'oversold_level': 10,
    'overbought_level': 90,
    'trend_sma_period': 200,
    'atr_period': 14,
    'atr_stop_mult': 1.5,
    'target_r_mult': 1.5
}

RSI_STANDARD_PARAMS = {
    'rsi_period': 14,
    'oversold_level': 30,
    'overbought_level': 70,
    'atr_period': 14,
    'atr_stop_mult': 2.0,
    'target_r_mult': 2.0
}

