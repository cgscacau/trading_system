"""
Estratégias baseadas em padrões de candlesticks
"""
import pandas as pd
import numpy as np
from ta.volatility import average_true_range

def candle_patterns_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia baseada em padrões de candlesticks com validação estatística
    """
    patterns = params.get('patterns', ['hammer', 'engulfing', 'doji'])
    trend_filter = params.get('trend_filter', True)
    trend_period = params.get('trend_period', 50)
    volume_confirmation = params.get('volume_confirmation', True)
    min_pattern_frequency = params.get('min_pattern_frequency', 0.6)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 1.5)
    target_r_mult = params.get('target_r_mult', 2.0)
    
    df_signals = df.copy()
    
    # Calcular componentes do candle
    df_signals['Body'] = abs(df['Close'] - df['Open'])
    df_signals['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
    df_signals['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
    df_signals['Range'] = df['High'] - df['Low']
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Filtro de tendência
    if trend_filter:
        from ta.trend import sma_indicator
        df_signals['SMA_Trend'] = sma_indicator(df['Close'], window=trend_period)
        uptrend = df['Close'] > df_signals['SMA_Trend']
        downtrend = df['Close'] < df_signals['SMA_Trend']
    else:
        uptrend = True
        downtrend = True
    
    # Volume confirmation
    if volume_confirmation and 'Volume' in df.columns:
        df_signals['Volume_MA'] = df['Volume'].rolling(10).mean()
        volume_high = df['Volume'] > df_signals['Volume_MA']
    else:
        volume_high = True
    
    # Detectar padrões
    bullish_patterns = pd.Series(False, index=df.index)
    bearish_patterns = pd.Series(False, index=df.index)
    
    if 'hammer' in patterns:
        # Hammer: corpo pequeno, sombra inferior longa
        hammer = (df_signals['Body'] < df_signals['Range'] * 0.3) & \
                 (df_signals['Lower_Shadow'] > df_signals['Body'] * 2) & \
                 (df_signals['Upper_Shadow'] < df_signals['Body'] * 0.5)
        bullish_patterns |= hammer & downtrend
    
    if 'engulfing' in patterns:
        # Bullish Engulfing
        bullish_engulfing = (df['Close'] > df['Open']) & \
                           (df['Close'].shift(1) < df['Open'].shift(1)) & \
                           (df['Open'] < df['Close'].shift(1)) & \
                           (df['Close'] > df['Open'].shift(1))
        bullish_patterns |= bullish_engulfing & downtrend
        
        # Bearish Engulfing
        bearish_engulfing = (df['Close'] < df['Open']) & \
                           (df['Close'].shift(1) > df['Open'].shift(1)) & \
                           (df['Open'] > df['Close'].shift(1)) & \
                           (df['Close'] < df['Open'].shift(1))
        bearish_patterns |= bearish_engulfing & uptrend
    
    if 'doji' in patterns:
        # Doji: corpo muito pequeno
        doji = df_signals['Body'] < df_signals['Range'] * 0.1
        bullish_patterns |= doji & downtrend
        bearish_patterns |= doji & uptrend
    
    # Validação estatística simplificada
    # Calcular expectância histórica dos padrões
    future_return = df['Close'].pct_change(5).shift(-5)
    
    bullish_expectancy = future_return[bullish_patterns].mean()
    bearish_expectancy = future_return[bearish_patterns].mean()
    
    # Aplicar filtro de frequência
    bullish_valid = bullish_expectancy > 0 and len(future_return[bullish_patterns]) > 10
    bearish_valid = bearish_expectancy < 0 and len(future_return[bearish_patterns]) > 10
    
    # Sinais
    df_signals['signal'] = 0
    
    if bullish_valid:
        df_signals.loc[bullish_patterns & volume_high, 'signal'] = 1
    if bearish_valid:
        df_signals.loc[bearish_patterns & volume_high, 'signal'] = -1
    
    # Stops e targets
    df_signals['stop'] = np.nan
    df_signals['target'] = np.nan
    
    buy_mask = df_signals['signal'] == 1
    df_signals.loc[buy_mask, 'stop'] = df_signals.loc[buy_mask, 'Low'] - \
                                       (df_signals.loc[buy_mask, 'ATR'] * 0.5)
    df_signals.loc[buy_mask, 'target'] = df_signals.loc[buy_mask, 'Close'] + \
                                         (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    sell_mask = df_signals['signal'] == -1
    df_signals.loc[sell_mask, 'stop'] = df_signals.loc[sell_mask, 'High'] + \
                                        (df_signals.loc[sell_mask, 'ATR'] * 0.5)
    df_signals.loc[sell_mask, 'target'] = df_signals.loc[sell_mask, 'Close'] - \
                                          (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    return df_signals[['signal', 'stop', 'target']].fillna(0)

CANDLE_PATTERNS_PARAMS = {
    'patterns': ['hammer', 'engulfing', 'doji'],
    'trend_filter': True,
    'trend_period': 50,
    'volume_confirmation': True,
    'min_pattern_frequency': 0.6,
    'atr_period': 14,
    'atr_stop_mult': 1.5,
    'target_r_mult': 2.0
}

