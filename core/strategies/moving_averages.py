import pandas as pd
import ta

EMA_CROSSOVER_PARAMS = {'ema_short': 12, 'ema_long': 26}
SMA_CROSSOVER_PARAMS = {'sma_short': 12, 'sma_long': 26}

def ema_crossover_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    ema_short_period = params.get('ema_short', 12)
    ema_long_period = params.get('ema_long', 26)

    df_s['ema_short'] = ta.trend.ema_indicator(df_s['Close'], window=ema_short_period)
    df_s['ema_long'] = ta.trend.ema_indicator(df_s['Close'], window=ema_long_period)

    df_s['signal'] = 0
    df_s.loc[df_s['ema_short'] > df_s['ema_long'], 'signal'] = 1
    df_s.loc[df_s['ema_short'] < df_s['ema_long'], 'signal'] = -1
    
    signals = pd.DataFrame(index=df_s.index)
    signals['signal'] = df_s['signal']
    signals['stop'] = df_s.apply(lambda row: row['Low'] * 0.98 if signals.loc[row.name, 'signal'] == 1 else row['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda row: row['High'] * 1.05 if signals.loc[row.name, 'signal'] == 1 else row['Low'] * 0.95, axis=1)
    
    return signals[['signal', 'stop', 'target']]

def sma_crossover_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    sma_short_period = params.get('sma_short', 12)
    sma_long_period = params.get('sma_long', 26)

    df_s['sma_short'] = ta.trend.sma_indicator(df_s['Close'], window=sma_short_period)
    df_s['sma_long'] = ta.trend.sma_indicator(df_s['Close'], window=sma_long_period)

    df_s['signal'] = 0
    df_s.loc[df_s['sma_short'] > df_s['sma_long'], 'signal'] = 1
    df_s.loc[df_s['sma_short'] < df_s['sma_long'], 'signal'] = -1
    
    signals = pd.DataFrame(index=df_s.index)
    signals['signal'] = df_s['signal']
    signals['stop'] = df_s.apply(lambda row: row['Low'] * 0.98 if signals.loc[row.name, 'signal'] == 1 else row['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda row: row['High'] * 1.05 if signals.loc[row.name, 'signal'] == 1 else row['Low'] * 0.95, axis=1)
    
    return signals[['signal', 'stop', 'target']]
