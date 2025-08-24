"""
Estratégias inventadas usando métodos matemáticos e combinações inovadoras
"""
import pandas as pd
import numpy as np
from ta.volatility import average_true_range, BollingerBands
from ta.trend import sma_indicator, ema_indicator, ADXIndicator
from ta.momentum import rsi

def vol_regime_switch_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Alterna entre mean-reversion e breakout conforme regime de volatilidade
    """
    atr_period = params.get('atr_period', 14)
    vol_threshold_mult = params.get('vol_threshold_mult', 1.2)
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 2.0)
    target_r_mult = params.get('target_r_mult', 2.0)
    
    df_signals = df.copy()
    
    # Calcular volatilidade
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    df_signals['ATR_MA'] = df_signals['ATR'].rolling(50).mean()
    df_signals['Vol_Ratio'] = df_signals['ATR'] / df_signals['ATR_MA']
    
    # Bollinger Bands para mean reversion
    bb = BollingerBands(close=df['Close'], window=bb_period, window_dev=bb_std)
    df_signals['BB_Upper'] = bb.bollinger_hband()
    df_signals['BB_Lower'] = bb.bollinger_lband()
    df_signals['BB_Middle'] = bb.bollinger_mavg()
    
    # RSI
    df_signals['RSI'] = rsi(df['Close'], window=rsi_period)
    
    # Regime de volatilidade
    high_vol_regime = df_signals['Vol_Ratio'] > vol_threshold_mult
    low_vol_regime = df_signals['Vol_Ratio'] <= vol_threshold_mult
    
    # Sinais baseados no regime
    df_signals['signal'] = 0
    
    # Regime baixa volatilidade: Mean Reversion
    mean_reversion_buy = (df['Close'] <= df_signals['BB_Lower']) & \
                        (df_signals['RSI'] < 30) & low_vol_regime
    mean_reversion_sell = (df['Close'] >= df_signals['BB_Upper']) & \
                         (df_signals['RSI'] > 70) & low_vol_regime
    
    # Regime alta volatilidade: Breakout
    breakout_buy = (df['Close'] > df_signals['BB_Upper']) & high_vol_regime
    breakout_sell = (df['Close'] < df_signals['BB_Lower']) & high_vol_regime
    
    # Aplicar sinais
    df_signals.loc[mean_reversion_buy | breakout_buy, 'signal'] = 1
    df_signals.loc[mean_reversion_sell | breakout_sell, 'signal'] = -1
    
    # Stops e targets adaptativos
    df_signals['stop'] = np.nan
    df_signals['target'] = np.nan
    
    buy_mask = df_signals['signal'] == 1
    sell_mask = df_signals['signal'] == -1
    
    # Para mean reversion: stops mais apertados
    mean_rev_buy_mask = buy_mask & low_vol_regime
    mean_rev_sell_mask = sell_mask & low_vol_regime
    
    df_signals.loc[mean_rev_buy_mask, 'stop'] = df_signals.loc[mean_rev_buy_mask, 'Close'] - \
                                               (df_signals.loc[mean_rev_buy_mask, 'ATR'] * atr_stop_mult * 0.7)
    df_signals.loc[mean_rev_buy_mask, 'target'] = df_signals.loc[mean_rev_buy_mask, 'BB_Middle']
    
    df_signals.loc[mean_rev_sell_mask, 'stop'] = df_signals.loc[mean_rev_sell_mask, 'Close'] + \
                                                (df_signals.loc[mean_rev_sell_mask, 'ATR'] * atr_stop_mult * 0.7)
    df_signals.loc[mean_rev_sell_mask, 'target'] = df_signals.loc[mean_rev_sell_mask, 'BB_Middle']
    
    # Para breakout: stops normais
    breakout_buy_mask = buy_mask & high_vol_regime
    breakout_sell_mask = sell_mask & high_vol_regime
    
    df_signals.loc[breakout_buy_mask, 'stop'] = df_signals.loc[breakout_buy_mask, 'Close'] - \
                                               (df_signals.loc[breakout_buy_mask, 'ATR'] * atr_stop_mult)
    df_signals.loc[breakout_buy_mask, 'target'] = df_signals.loc[breakout_buy_mask, 'Close'] + \
                                                 (df_signals.loc[breakout_buy_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    df_signals.loc[breakout_sell_mask, 'stop'] = df_signals.loc[breakout_sell_mask, 'Close'] + \
                                                (df_signals.loc[breakout_sell_mask, 'ATR'] * atr_stop_mult)
    df_signals.loc[breakout_sell_mask, 'target'] = df_signals.loc[breakout_sell_mask, 'Close'] - \
                                                  (df_signals.loc[breakout_sell_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    return df_signals[['signal', 'stop', 'target']].fillna(0)

def meta_ensemble_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Meta-estratégia que combina voto de múltiplas estratégias clássicas
    """
    ema_fast = params.get('ema_fast', 9)
    ema_slow = params.get('ema_slow', 21)
    rsi_period = params.get('rsi_period', 14)
    macd_fast = params.get('macd_fast', 12)
    macd_slow = params.get('macd_slow', 26)
    macd_signal = params.get('macd_signal', 9)
    ml_weight = params.get('ml_weight', 1.5)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 2.0)
    target_r_mult = params.get('target_r_mult', 2.0)
    
    df_signals = df.copy()
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Estratégia 1: EMA Crossover
    df_signals['EMA_Fast'] = ema_indicator(df['Close'], window=ema_fast)
    df_signals['EMA_Slow'] = ema_indicator(df['Close'], window=ema_slow)
    
    ema_cross_up = (df_signals['EMA_Fast'] > df_signals['EMA_Slow']) & \
                   (df_signals['EMA_Fast'].shift(1) <= df_signals['EMA_Slow'].shift(1))
    ema_cross_down = (df_signals['EMA_Fast'] < df_signals['EMA_Slow']) & \
                     (df_signals['EMA_Fast'].shift(1) >= df_signals['EMA_Slow'].shift(1))
    
    df_signals['EMA_Vote'] = 0
    df_signals.loc[ema_cross_up, 'EMA_Vote'] = 1
    df_signals.loc[ema_cross_down, 'EMA_Vote'] = -1
    
    # Estratégia 2: RSI
    df_signals['RSI'] = rsi(df['Close'], window=rsi_period)
    df_signals['RSI_Vote'] = 0
    
    rsi_buy = (df_signals['RSI'] > 30) & (df_signals['RSI'].shift(1) <= 30)
    rsi_sell = (df_signals['RSI'] < 70) & (df_signals['RSI'].shift(1) >= 70)
    
    df_signals.loc[rsi_buy, 'RSI_Vote'] = 1
    df_signals.loc[rsi_sell, 'RSI_Vote'] = -1
    
    # Estratégia 3: MACD
    from ta.trend import MACD
    macd_indicator = MACD(close=df['Close'], window_fast=macd_fast, 
                         window_slow=macd_slow, window_sign=macd_signal)
    df_signals['MACD'] = macd_indicator.macd()
    df_signals['MACD_Signal'] = macd_indicator.macd_signal()
    
    df_signals['MACD_Vote'] = 0
    macd_cross_up = (df_signals['MACD'] > df_signals['MACD_Signal']) & \
                    (df_signals['MACD'].shift(1) <= df_signals['MACD_Signal'].shift(1))
    macd_cross_down = (df_signals['MACD'] < df_signals['MACD_Signal']) & \
                      (df_signals['MACD'].shift(1) >= df_signals['MACD_Signal'].shift(1))
    
    df_signals.loc[macd_cross_up, 'MACD_Vote'] = 1
    df_signals.loc[macd_cross_down, 'MACD_Vote'] = -1
    
    # Estratégia 4: ML simplificado (baseado em momentum multi-timeframe)
    df_signals['Return_3'] = df['Close'].pct_change(3)
    df_signals['Return_7'] = df['Close'].pct_change(7)
    df_signals['Return_14'] = df['Close'].pct_change(14)
    
    momentum_score = (df_signals['Return_3'] > 0).astype(int) + \
                    (df_signals['Return_7'] > 0).astype(int) + \
                    (df_signals['Return_14'] > 0).astype(int)
    
    df_signals['ML_Vote'] = 0
    df_signals.loc[momentum_score >= 3, 'ML_Vote'] = 1
    df_signals.loc[momentum_score == 0, 'ML_Vote'] = -1
    
    # Voto combinado
    df_signals['Total_Score'] = (df_signals['EMA_Vote'] + 
                                df_signals['RSI_Vote'] + 
                                df_signals['MACD_Vote'] + 
                                df_signals['ML_Vote'] * ml_weight)
    
    # Sinais baseados em consenso
    df_signals['signal'] = 0
    df_signals.loc[df_signals['Total_Score'] > 1.5, 'signal'] = 1
    df_signals.loc[df_signals['Total_Score'] < -1.5, 'signal'] = -1
    
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

def pullback_trend_bias_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Compra no reteste da EMA21 dentro de tendência confirmada por ADX
    """
    ema_period = params.get('ema_period', 21)
    adx_period = params.get('adx_period', 14)
    adx_threshold = params.get('adx_threshold', 25)
    pullback_tolerance = params.get('pullback_tolerance', 0.005)
    volume_confirmation = params.get('volume_confirmation', True)
    atr_period = params.get('atr_period', 14)
    atr_stop_mult = params.get('atr_stop_mult', 1.5)
    target_r_mult = params.get('target_r_mult', 2.5)
    
    df_signals = df.copy()
    
    # EMA para trend bias
    df_signals['EMA'] = ema_indicator(df['Close'], window=ema_period)
    
    # ADX para força da tendência
    adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=adx_period)
    df_signals['ADX'] = adx_indicator.adx()
    df_signals['DI_Plus'] = adx_indicator.adx_pos()
    df_signals['DI_Minus'] = adx_indicator.adx_neg()
    
    # ATR
    df_signals['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Volume confirmation
    if volume_confirmation and 'Volume' in df.columns:
        df_signals['Volume_MA'] = df['Volume'].rolling(10).mean()
        volume_condition = df_signals['Volume'] > df_signals['Volume_MA']
    else:
        volume_condition = True
    
    # Condições de tendência forte
    strong_uptrend = (df_signals['ADX'] > adx_threshold) & \
                    (df_signals['DI_Plus'] > df_signals['DI_Minus']) & \
                    (df['Close'] > df_signals['EMA'])
    
    strong_downtrend = (df_signals['ADX'] > adx_threshold) & \
                      (df_signals['DI_Minus'] > df_signals['DI_Plus']) & \
                      (df['Close'] < df_signals['EMA'])
    
    # Pullback para EMA
    distance_from_ema = abs(df['Close'] - df_signals['EMA']) / df['Close']
    near_ema = distance_from_ema <= pullback_tolerance
    
    # Rejeição na EMA
    bullish_rejection = (df['Low'] <= df_signals['EMA'] * (1 + pullback_tolerance)) & \
                       (df['Close'] > df_signals['EMA']) & \
                       (df['Close'] > df['Open'])
    
    bearish_rejection = (df['High'] >= df_signals['EMA'] * (1 - pullback_tolerance)) & \
                       (df['Close'] < df_signals['EMA']) & \
                       (df['Close'] < df['Open'])
    
    # Sinais
    df_signals['signal'] = 0
    
    buy_condition = strong_uptrend & bullish_rejection & volume_condition
    sell_condition = strong_downtrend & bearish_rejection & volume_condition
    
    df_signals.loc[buy_condition, 'signal'] = 1
    df_signals.loc[sell_condition, 'signal'] = -1
    
    # Stops e targets
    df_signals['stop'] = np.nan
    df_signals['target'] = np.nan
    
    buy_mask = df_signals['signal'] == 1
    df_signals.loc[buy_mask, 'stop'] = df_signals.loc[buy_mask, 'EMA'] - \
                                       (df_signals.loc[buy_mask, 'ATR'] * 0.5)
    df_signals.loc[buy_mask, 'target'] = df_signals.loc[buy_mask, 'Close'] + \
                                         (df_signals.loc[buy_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    sell_mask = df_signals['signal'] == -1
    df_signals.loc[sell_mask, 'stop'] = df_signals.loc[sell_mask, 'EMA'] + \
                                        (df_signals.loc[sell_mask, 'ATR'] * 0.5)
    df_signals.loc[sell_mask, 'target'] = df_signals.loc[sell_mask, 'Close'] - \
                                          (df_signals.loc[sell_mask, 'ATR'] * atr_stop_mult * target_r_mult)
    
    return df_signals[['signal', 'stop', 'target']].fillna(0)

# Parâmetros padrão
VOL_REGIME_SWITCH_PARAMS = {
    'atr_period': 14,
    'vol_threshold_mult': 1.2,
    'bb_period': 20,
    'bb_std': 2.0,
    'rsi_period': 14,
    'atr_stop_mult': 2.0,
    'target_r_mult': 2.0
}

META_ENSEMBLE_PARAMS = {
    'ema_fast': 9,
    'ema_slow': 21,
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'ml_weight': 1.5,
    'atr_period': 14,
    'atr_stop_mult': 2.0,
    'target_r_mult': 2.0
}

PULLBACK_TREND_BIAS_PARAMS = {
    'ema_period': 21,
    'adx_period': 14,
    'adx_threshold': 25,
    'pullback_tolerance': 0.005,
    'volume_confirmation': True,
    'atr_period': 14,
    'atr_stop_mult': 1.5,
    'target_r_mult': 2.5
}

