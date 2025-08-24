import pandas as pd
import ta

EMA_CROSSOVER_PARAMS = {'ema_short': 12, 'ema_long': 26}
SMA_CROSSOVER_PARAMS = {'sma_short': 12, 'sma_long': 26}

def ema_crossover_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)
    
    ema_short = ta.trend.ema_indicator(df_s['Close'], window=params.get('ema_short', 12))
    ema_long = ta.trend.ema_indicator(df_s['Close'], window=params.get('ema_long', 26))
    
    signals['signal'] = 0
    signals.loc[ema_short > ema_long, 'signal'] = 1
    signals.loc[ema_short < ema_long, 'signal'] = -1
    
    signals['stop'] = df_s.apply(lambda r: r['Low'] * 0.98 if signals.loc[r.name, 'signal'] == 1 else r['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda r: r['High'] * 1.05 if signals.loc[r.name, 'signal'] == 1 else r['Low'] * 0.95, axis=1)
    
    return signals[['signal', 'stop', 'target']]

def sma_crossover_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)

    sma_short = ta.trend.sma_indicator(df_s['Close'], window=params.get('sma_short', 12))
    sma_long = ta.trend.sma_indicator(df_s['Close'], window=params.get('sma_long', 26))

    signals['signal'] = 0
    signals.loc[sma_short > sma_long, 'signal'] = 1
    signals.loc[sma_short < sma_long, 'signal'] = -1

    signals['stop'] = df_s.apply(lambda r: r['Low'] * 0.98 if signals.loc[r.name, 'signal'] == 1 else r['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda r: r['High'] * 1.05 if signals.loc[r.name, 'signal'] == 1 else r['Low'] * 0.95, axis=1)

    return signals[['signal', 'stop', 'target']]
