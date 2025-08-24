import pandas as pd
import ta

RSI_IFR2_PARAMS = {'period': 2, 'entry': 10, 'exit': 70}
RSI_STANDARD_PARAMS = {'period': 14, 'entry_buy': 30, 'entry_sell': 70}

def rsi_ifr2_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)

    rsi = ta.momentum.rsi(df_s['Close'], window=params.get('period', 2))
    
    signals['signal'] = 0
    signals.loc[rsi < params.get('entry', 10), 'signal'] = 1
    signals.loc[rsi > params.get('exit', 70), 'signal'] = -1
    
    signals['stop'] = df_s.apply(lambda r: r['Low'] * 0.97 if signals.loc[r.name, 'signal'] == 1 else pd.NA, axis=1)
    signals['target'] = pd.NA

    return signals[['signal', 'stop', 'target']]

def rsi_standard_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)
    
    rsi = ta.momentum.rsi(df_s['Close'], window=params.get('period', 14))
    
    signals['signal'] = 0
    signals.loc[rsi < params.get('entry_buy', 30), 'signal'] = 1
    signals.loc[rsi > params.get('entry_sell', 70), 'signal'] = -1
    
    signals['stop'] = df_s.apply(lambda r: r['Low'] * 0.98 if signals.loc[r.name, 'signal'] == 1 else r['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda r: r['High'] * 1.05 if signals.loc[r.name, 'signal'] == 1 else r['Low'] * 0.95, axis=1)
    
    return signals[['signal', 'stop', 'target']]
