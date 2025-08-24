import pandas as pd
import ta

# ATENÇÃO: Não estamos mais importando de outros arquivos de estratégia locais
# para evitar erros em cascata.

# --- PARÂMETROS PARA AS ESTRATÉGIAS DESTE ARQUIVO ---
VOL_REGIME_SWITCH_PARAMS = {'vol_period': 14, 'vol_threshold_pct': 1.5, 'trend_period': 50}
META_ENSEMBLE_PARAMS = {}
PULLBACK_TREND_BIAS_PARAMS = {'trend_period': 50, 'pullback_period': 5}


# --- IMPLEMENTAÇÃO DAS ESTRATÉGIAS ---

def vol_regime_switch_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Estratégia que alterna com base no regime de volatilidade (ATR)."""
    df_s = df.copy()
    vol_period = params.get('vol_period', 14)
    vol_threshold_pct = params.get('vol_threshold_pct', 1.5)
    trend_period = params.get('trend_period', 50)

    # Garante que os dados passados para os indicadores são 1-dimensionais
    df_s['atr'] = ta.volatility.average_true_range(df_s['High'], df_s['Low'], df_s['Close'], window=vol_period)
    df_s['atr_ma'] = ta.trend.sma_indicator(df_s['atr'], window=trend_period)
    df_s['trend_ma'] = ta.trend.sma_indicator(df_s['Close'], window=trend_period)
    
    signals = pd.DataFrame(index=df_s.index)
    signals['signal'] = 0
    
    high_vol_condition = df_s['atr'] > (df_s['atr_ma'] * vol_threshold_pct)
    low_vol_condition = ~high_vol_condition
    
    # Lógica de sinais
    signals.loc[high_vol_condition & (df_s['Close'] < df_s['trend_ma']), 'signal'] = 1
    signals.loc[high_vol_condition & (df_s['Close'] > df_s['trend_ma']), 'signal'] = -1
    signals.loc[low_vol_condition & (df_s['Close'] > df_s['trend_ma']), 'signal'] = 1
    signals.loc[low_vol_condition & (df_s['Close'] < df_s['trend_ma']), 'signal'] = -1

    signals['stop'] = df_s.apply(lambda row: row['Low'] * 0.98 if signals.loc[row.name, 'signal'] == 1 else row['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda row: row['High'] * 1.05 if signals.loc[row.name, 'signal'] == 1 else row['Low'] * 0.95, axis=1)

    return signals[['signal', 'stop', 'target']]


def meta_ensemble_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Versão autossuficiente que combina sinais de múltiplas lógicas."""
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)

    # --- Lógicas re-implementadas aqui para remover dependências externas ---
    # 1. Lógica de Cruzamento de Médias Móveis (EMA)
    ema_short = ta.trend.ema_indicator(df_s['Close'], window=12)
    ema_long = ta.trend.ema_indicator(df_s['Close'], window=26)
    ema_signal = pd.Series(0, index=df_s.index, dtype=int)
    ema_signal[ema_short > ema_long] = 1
    ema_signal[ema_short < ema_long] = -1

    # 2. Lógica de IFR (RSI) Padrão
    rsi = ta.momentum.rsi(df_s['Close'], window=14)
    rsi_signal = pd.Series(0, index=df_s.index, dtype=int)
    rsi_signal[rsi < 30] = 1
    rsi_signal[rsi > 70] = -1
    
    # 3. Lógica de Reversão à Média com Bandas de Bollinger
    indicator_bb = ta.volatility.BollingerBands(close=df_s['Close'], window=20, window_dev=2)
    bb_upper = indicator_bb.bollinger_hband()
    bb_lower = indicator_bb.bollinger_lband()
    bb_signal = pd.Series(0, index=df_s.index, dtype=int)
    bb_signal[df_s['Close'] < bb_lower] = 1
    bb_signal[df_s['Close'] > bb_upper] = -1

    # --- Lógica de Votação ---
    vote_sum = ema_signal + rsi_signal + bb_signal
    signals['signal'] = 0
    signals.loc[vote_sum >= 2, 'signal'] = 1   # Compra se 2 ou mais concordarem
    signals.loc[vote_sum <= -2, 'signal'] = -1 # Vende se 2 ou mais concordarem
    
    signals['stop'] = df_s.apply(lambda row: row['Low'] * 0.98 if signals.loc[row.name, 'signal'] == 1 else row['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda row: row['High'] * 1.05 if signals.loc[row.name, 'signal'] == 1 else row['Low'] * 0.95, axis=1)
    
    return signals[['signal', 'stop', 'target']]


def pullback_trend_bias_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Busca por pullbacks dentro de uma tendência de alta."""
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
