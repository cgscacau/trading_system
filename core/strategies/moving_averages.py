"""
Estratégias baseadas em médias móveis
"""
import pandas as pd
import numpy as np
from ta.trend import ema_indicator, sma_indicator
from ta.volatility import average_true_range

def ema_crossover_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia de cruzamento de EMAs com filtro de tendência
    
    Parâmetros:
        fast_period (int): Período da EMA rápida (default: 9)
        slow_period (int): Período da EMA lenta (default: 21) 
        trend_period (int): Período da SMA de tendência (default: 200)
        atr_period (int): Período do ATR (default: 14)
        atr_stop_mult (float): Multiplicador ATR para stop (default: 2.0)
        target_r_mult (float): Múltiplo R para target (default: 2.0)
    """
    # Parâmetros padrão
    fast_period = params.get('fast_period', 9)
    slow_period = params.get('slow_period', 21)
    trend_period = params.get('trend_period', 200)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 2.0)
    target_r_mult = params.get('target_r_mult', 2.0)
    
    df_signals = df.copy()
    
    # Calcular indicadores
    df_signals['EMA_Fast'] = ema_indicator(df['Close'], window=fast_period)
    df_signals['EMA_Slow'] = ema_indicator(df['Close'], window=slow_period)
    df_signals['SMA_Trend'] = sma_indicator(df['Close'], window=trend_period)
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Sinais de cruzamento
    cross_up = (df_signals['EMA_Fast'] > df_signals['EMA_Slow']) & \
               (df_signals['EMA_Fast'].shift(1) <= df_signals['EMA_Slow'].shift(1))
    
    cross_down = (df_signals['EMA_Fast'] < df_signals['EMA_Slow']) & \
                 (df_signals['EMA_Fast'].shift(1) >= df_signals['EMA_Slow'].shift(1))
    
    # Filtro de tendência
    trend_up = df_signals['Close'] > df_signals['SMA_Trend']
    trend_down = df_signals['Close'] < df_signals['SMA_Trend']
    
    # Gerar sinais
    df_signals['signal'] = 0
    df_signals.loc[cross_up & trend_up, 'signal'] = 1  # Buy
    df_signals.loc[cross_down & trend_down, 'signal'] = -1  # Sell
    
    # Calcular stops e targets
    df_signals['stop'] = np.nan
    df_signals['target'] = np.nan
    
    # Para sinais de compra
    buy_mask = df_signals['signal'] == 1
    df_signals.loc[buy_mask, 'stop'] = df_signals.loc[buy_mask, 'Close'] - \
                                       (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult)
    df_signals.loc[buy_mask, 'target'] = df_signals.loc[buy_mask, 'Close'] + \
                                         (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    # Para sinais de venda
    sell_mask = df_signals['signal'] == -1
    df_signals.loc[sell_mask, 'stop'] = df_signals.loc[sell_mask, 'Close'] + \
                                        (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult)
    df_signals.loc[sell_mask, 'target'] = df_signals.loc[sell_mask, 'Close'] - \
                                          (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    return df_signals[['signal', 'stop', 'target']].fillna(0)

def sma_crossover_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia de cruzamento de SMAs
    
    Parâmetros similares à EMA crossover
    """
    fast_period = params.get('fast_period', 50)
    slow_period = params.get('slow_period', 200)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 2.0)
    target_r_mult = params.get('target_r_mult', 2.0)
    
    df_signals = df.copy()
    
    # Calcular SMAs
    df_signals['SMA_Fast'] = sma_indicator(df['Close'], window=fast_period)
    df_signals['SMA_Slow'] = sma_indicator(df['Close'], window=slow_period)
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Cruzamentos
    cross_up = (df_signals['SMA_Fast'] > df_signals['SMA_Slow']) & \
               (df_signals['SMA_Fast'].shift(1) <= df_signals['SMA_Slow'].shift(1))
    
    cross_down = (df_signals['SMA_Fast'] < df_signals['SMA_Slow']) & \
                 (df_signals['SMA_Fast'].shift(1) >= df_signals['SMA_Slow'].shift(1))
    
    # Sinais
    df_signals['signal'] = 0
    df_signals.loc[cross_up, 'signal'] = 1
    df_signals.loc[cross_down, 'signal'] = -1
    
    # Stops e targets (mesma lógica da EMA)
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

# Parâmetros padrão para as estratégias
EMA_CROSSOVER_PARAMS = {
    'fast_period': 9,
    'slow_period': 21,
    'trend_period': 200,
    'atr_period': 14,
    'atr_stop_mult': 2.0,
    'target_r_mult': 2.0
}

SMA_CROSSOVER_PARAMS = {
    'fast_period': 50,
    'slow_period': 200,
    'atr_period': 14,
    'atr_stop_mult': 2.0,
    'target_r_mult': 2.0
}

