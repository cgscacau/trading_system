import pandas as pd

HIGH_LOW_BREAKOUT_PARAMS = {'period': 20}

def high_low_breakout_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)
    period = params.get('period', 20)

    high_roll = df_s['High'].shift(1).rolling(window=period).max()
    low_roll = df_s['Low'].shift(1).rolling(window=period).min()
    
    signals['signal'] = 0
    signals.loc[df_s['Close'] > high_roll, 'signal'] = 1
    signals.loc[df_s['Close'] < low_roll, 'signal'] = -1

    signals['stop'] = low_roll.where(signals['signal'] == 1, high_roll)
    signals['target'] = pd.NA

    return signals[['signal', 'stop', 'target']]
