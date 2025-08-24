import pandas as pd
import ta

# Importa outras estratégias para serem usadas no Meta-Ensemble
from .moving_averages import ema_crossover_strategy, EMA_CROSSOVER_PARAMS
from .rsi_strategies import rsi_standard_strategy, RSI_STANDARD_PARAMS
from .bollinger_bands import bollinger_mean_reversion_strategy, BOLLINGER_MEAN_REVERSION_PARAMS

# --- PARÂMETROS PARA AS ESTRATÉGIAS DESTE ARQUIVO ---
# Garanta que estes nomes são EXATAMENTE os mesmos que você importa em app.py
VOL_REGIME_SWITCH_PARAMS = {'vol_period': 14, 'vol_threshold_pct': 1.5, 'trend_period': 50}
META_ENSEMBLE_PARAMS = {}
PULLBACK_TREND_BIAS_PARAMS = {'trend_period': 50, 'pullback_period': 5}

# --- IMPLEMENTAÇÃO DAS ESTRATÉGIAS ---
# Garanta que os nomes das funções são EXATAMENTE os mesmos

def vol_regime_switch_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estratégia que alterna entre seguidora de tendência e reversão à média
    baseado no regime de volatilidade (usando ATR).
    """
    df_s = df.copy()
    vol_period = params.get('vol_period', 14)
    vol_threshold_pct = params.get('vol_threshold_pct', 1.5)
    trend_period = params.get('trend_period', 50)

    df_s['atr'] = ta.volatility.average_true_range(df_s['High'], df_s['Low'], df_s['Close'], window=vol_period)
    df_s['atr_ma'] = df_s['atr'].rolling(window=trend_period).mean()
    df_s['trend_ma'] = ta.trend.sma_indicator(df_s['Close'], window=trend_period)
    
    signals = pd.DataFrame(index=df_s.index)
    signals['signal'] = 0
    
    high_vol_condition = df_s['atr'] > (df_s['atr_ma'] * vol_threshold_pct)
    low_vol_condition = ~high_vol_condition
    
    signals.loc[high_vol_condition & (df_s['Close'] < df_s['trend_ma']), 'signal'] = 1
    signals.loc[high_vol_condition & (df_s['Close'] > df_s['trend_ma']), 'signal'] = -1
    signals.loc[low_vol_condition & (df_s['Close'] > df_s['trend_ma']), 'signal'] = 1
    signals.loc[low_vol_condition & (df_s['Close'] < df_s['trend_ma']), 'signal'] = -1

    signals['stop'] = df_s.apply(lambda row: row['Low'] * 0.98 if signals.loc[row.name, 'signal'] == 1 else row['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda row: row['High'] * 1.05 if signals.loc[row.name, 'signal'] == 1 else row['Low'] * 0.95, axis=1)

    return signals[['signal', 'stop', 'target']]

def meta_ensemble_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Combina os sinais de múltiplas estratégias para um sinal mais robusto.
    """
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)
    
    ema_signals = ema_crossover_strategy(df_s, EMA_CROSSOVER_PARAMS)
    rsi_signals = rsi_standard_strategy(df_s, RSI_STANDARD_PARAMS)
    bb_signals = bollinger_mean_reversion_strategy(df_s, BOLLINGER_MEAN_REVERSION_PARAMS)
    
    # Lógica de Votação/Ensemble
    vote_sum = ema_signals['signal'] + rsi_signals['signal'] + bb_signals['signal']
    signals['signal'] = 0
    signals.loc[vote_sum >= 2, 'signal'] = 1   # Compra se 2 ou mais estratégias concordarem
    signals.loc[vote_sum <= -2, 'signal'] = -1 # Vende se 2 ou mais estratégias concordarem
    
    signals['stop'] = df_s.apply(lambda row: row['Low'] * 0.98 if signals.loc[row.name, 'signal'] == 1 else row['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda row: row['High'] * 1.05 if signals.loc[row.name, 'signal'] == 1 else row['Low'] * 0.95, axis=1)

    return signals[['signal', 'stop', 'target']]

def pullback_trend_bias_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Busca por pullbacks (pequenas correções) dentro de uma tendência de alta estabelecida.
    """
    df_s = df.copy()
    trend_period = params.get('trend_period', 50)
    pullback_period = params.get('pullback_period', 5)

    df_s['trend_ma'] = ta.trend.sma_indicator(df_s['Close'], window=trend_period)
    is_uptrend = df_s['Close'] > df_s['trend_ma']
    
    df_s['pullback_low'] = df_s['Low'].rolling(window=pullback_period).min()
    is_pullback = df_s['Low'] <= df_s['pullback_low']

    signals = pd.DataFrame(index=df_s.index)
    signals['signal'] = 0
    signals.loc[is_uptrend & is_pullback, 'signal'] = 1
    
    signals['stop'] = df_s.apply(lambda row: row['pullback_low'] * 0.99 if signals.loc[row.name, 'signal'] == 1 else pd.NA, axis=1)
    signals['target'] = df_s.apply(lambda row: row['High'] * 1.07 if signals.loc[row.name, 'signal'] == 1 else pd.NA, axis=1)

    return signals[['signal', 'stop', 'target']]
