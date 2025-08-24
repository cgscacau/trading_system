"""
Estratégias baseadas em ADX e DMI
"""
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from ta.volatility import average_true_range

def adx_dmi_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia ADX + DMI com filtro de tendência
    """
    adx_period = params.get('adx_period', 14)
    adx_threshold = params.get('adx_threshold', 25)
    di_crossover = params.get('di_crossover', True)
    price_confirmation = params.get('price_confirmation', True)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 2.0)
    target_r_mult = params.get('target_r_mult', 2.0)
    
    df_signals = df.copy()
    
    # ADX e DMI
    adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=adx_period)
    df_signals['ADX'] = adx_indicator.adx()
    df_signals['DI_Plus'] = adx_indicator.adx_pos()
    df_signals['DI_Minus'] = adx_indicator.adx_neg()
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Filtro de tendência forte
    strong_trend = df_signals['ADX'] > adx_threshold
    
    if di_crossover:
        # Cruzamentos DI
        di_bullish_cross = (df_signals['DI_Plus'] > df_signals['DI_Minus']) & \
                          (df_signals['DI_Plus'].shift(1) <= df_signals['DI_Minus'].shift(1))
        di_bearish_cross = (df_signals['DI_Plus'] < df_signals['DI_Minus']) & \
                          (df_signals['DI_Plus'].shift(1) >= df_signals['DI_Minus'].shift(1))
        
        buy_condition = di_bullish_cross & strong_trend
        sell_condition = di_bearish_cross & strong_trend
    else:
        # Apenas direção DI
        buy_condition = (df_signals['DI_Plus'] > df_signals['DI_Minus']) & strong_trend
        sell_condition = (df_signals['DI_Plus'] < df_signals['DI_Minus']) & strong_trend
    
    # Confirmação de preço (opcional)
    if price_confirmation:
        from ta.trend import sma_indicator
        sma_20 = sma_indicator(df['Close'], window=20)
        price_up = df['Close'] > sma_20
        price_down = df['Close'] < sma_20
        
        buy_condition = buy_condition & price_up
        sell_condition = sell_condition & price_down
    
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

ADX_DMI_PARAMS = {
    'adx_period': 14,
    'adx_threshold': 25,
    'di_crossover': True,
    'price_confirmation': True,
    'atr_period': 14,
    'atr_stop_mult': 2.0,
    'target_r_mult': 2.0
}

