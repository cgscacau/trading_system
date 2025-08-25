import pandas as pd
import ta

VOL_REGIME_SWITCH_PARAMS = {'vol_period': 14, 'vol_threshold_pct': 1.5, 'trend_period': 50}
META_ENSEMBLE_PARAMS = {}
PULLBACK_TREND_BIAS_PARAMS = {'trend_period': 50, 'pullback_period': 5}

def vol_regime_switch_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)

    vol_period = params.get('vol_period', 14)
    vol_threshold_pct = params.get('vol_threshold_pct', 1.5)
    trend_period = params.get('trend_period', 50)

    df_s['atr'] = ta.volatility.average_true_range(df_s['High'], df_s['Low'], df_s['Close'], window=vol_period)
    df_s['atr_ma'] = ta.trend.sma_indicator(df_s['atr'], window=trend_period)
    df_s['trend_ma'] = ta.trend.sma_indicator(df_s['Close'], window=trend_period)
    
    high_vol = df_s['atr'] > (df_s['atr_ma'] * vol_threshold_pct)
    low_vol = ~high_vol
    
    signals['signal'] = 0
    signals.loc[high_vol & (df_s['Close'] < df_s['trend_ma']), 'signal'] = 1
    signals.loc[high_vol & (df_s['Close'] > df_s['trend_ma']), 'signal'] = -1
    signals.loc[low_vol & (df_s['Close'] > df_s['trend_ma']), 'signal'] = 1
    signals.loc[low_vol & (df_s['Close'] < df_s['trend_ma']), 'signal'] = -1

    signals['stop'] = df_s.apply(lambda r: r['Low'] * 0.98 if signals.loc[r.name]['signal'] == 1 else r['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda r: r['High'] * 1.05 if signals.loc[r.name]['signal'] == 1 else r['Low'] * 0.95, axis=1)

    return signals[['signal', 'stop', 'target']]

def meta_ensemble_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)

    ema_short = ta.trend.ema_indicator(df_s['Close'], window=12)
    ema_long = ta.trend.ema_indicator(df_s['Close'], window=26)
    ema_signal = pd.Series(0, index=df_s.index, dtype=int)
    ema_signal[ema_short > ema_long] = 1
    ema_signal[ema_short < ema_long] = -1

    rsi = ta.momentum.rsi(df_s['Close'], window=14)
    rsi_signal = pd.Series(0, index=df_s.index, dtype=int)
    rsi_signal[rsi < 30] = 1
    rsi_signal[rsi > 70] = -1
    
    vote_sum = ema_signal + rsi_signal
    signals['signal'] = 0
    signals.loc[vote_sum >= 2, 'signal'] = 1
    signals.loc[vote_sum <= -2, 'signal'] = -1
    
    signals['stop'] = df_s.apply(lambda r: r['Low'] * 0.98 if signals.loc[r.name]['signal'] == 1 else r['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda r: r['High'] * 1.05 if signals.loc[r.name]['signal'] == 1 else r['Low'] * 0.95, axis=1)
    
    return signals[['signal', 'stop', 'target']]

def pullback_trend_bias_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)
    trend_period = params.get('trend_period', 50)
    pullback_period = params.get('pullback_period', 5)

    df_s['trend_ma'] = ta.trend.sma_indicator(df_s['Close'], window=trend_period)
    is_uptrend = df_s['Close'] > df_s['trend_ma']
    
    df_s['pullback_low'] = df_s['Low'].rolling(window=pullback_period).min()
    is_pullback = df_s['Low'] <= df_s['pullback_low']

    signals['signal'] = 0
    signals.loc[is_uptrend & is_pullback, 'signal'] = 1
    
    signals['stop'] = df_s.apply(lambda r: r['pullback_low'] * 0.99 if signals.loc[r.name]['signal'] == 1 else pd.NA, axis=1)
    signals['target'] = df_s.apply(lambda r: r['High'] * 1.07 if signals.loc[r.name]['signal'] == 1 else pd.NA, axis=1)

    return signals[['signal', 'stop', 'target']]
