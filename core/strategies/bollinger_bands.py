import pandas as pd
import ta

BOLLINGER_BREAKOUT_PARAMS = {'period': 20, 'std_dev': 2}
BOLLINGER_MEAN_REVERSION_PARAMS = {'period': 20, 'std_dev': 2}

def bollinger_breakout_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)
    
    indicator = ta.volatility.BollingerBands(close=df_s['Close'], window=params.get('period', 20), window_dev=params.get('std_dev', 2))
    bb_upper = indicator.bollinger_hband()
    bb_lower = indicator.bollinger_lband()
    
    signals['signal'] = 0
    signals.loc[df_s['Close'] > bb_upper, 'signal'] = 1
    signals.loc[df_s['Close'] < bb_lower, 'signal'] = -1
    
    signals['stop'] = df_s.apply(lambda r: bb_lower[r.name] if signals.loc[r.name, 'signal'] == 1 else bb_upper[r.name], axis=1)
    signals['target'] = pd.NA

    return signals[['signal', 'stop', 'target']]

def bollinger_mean_reversion_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)

    indicator = ta.volatility.BollingerBands(close=df_s['Close'], window=params.get('period', 20), window_dev=params.get('std_dev', 2))
    bb_upper = indicator.bollinger_hband()
    bb_lower = indicator.bollinger_lband()
    bb_ma = indicator.bollinger_mavg()
    
    signals['signal'] = 0
    signals.loc[df_s['Close'] < bb_lower, 'signal'] = 1
    signals.loc[df_s['Close'] > bb_upper, 'signal'] = -1
    
    signals['stop'] = df_s.apply(lambda r: r['Low'] * 0.98 if signals.loc[r.name, 'signal'] == 1 else r['High'] * 1.02, axis=1)
    signals['target'] = bb_ma

    return signals[['signal', 'stop', 'target']]
