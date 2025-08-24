import pandas as pd
import ta

RSI_IFR2_PARAMS = {'period': 2, 'entry': 10, 'exit': 70}
RSI_STANDARD_PARAMS = {'period': 14, 'entry_buy': 30, 'entry_sell': 70}

def rsi_ifr2_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    period = params.get('period', 2)
    entry_level = params.get('entry', 10)
    exit_level = params.get('exit', 70)

    df_s['rsi'] = ta.momentum.rsi(df_s['Close'], window=period)
    
    signals = pd.DataFrame(index=df_s.index)
    signals['signal'] = 0
    signals.loc[df_s['rsi'] < entry_level, 'signal'] = 1
    signals.loc[df_s['rsi'] > exit_level, 'signal'] = -1
    
    signals['stop'] = df_s.apply(lambda row: row['Low'] * 0.97 if signals.loc[row.name, 'signal'] == 1 else pd.NA, axis=1)
    signals['target'] = pd.NA

    return signals[['signal', 'stop', 'target']]

def rsi_standard_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    period = params.get('period', 14)
    entry_buy = params.get('entry_buy', 30)
    entry_sell = params.get('entry_sell', 70)

    df_s['rsi'] = ta.momentum.rsi(df_s['Close'], window=period)
    
    signals = pd.DataFrame(index=df_s.index)
    signals['signal'] = 0
    signals.loc[df_s['rsi'] < entry_buy, 'signal'] = 1
    signals.loc[df_s['rsi'] > entry_sell, 'signal'] = -1
    
    signals['stop'] = df_s.apply(lambda row: row['Low'] * 0.98 if signals.loc[row.name, 'signal'] == 1 else row['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda row: row['High'] * 1.05 if signals.loc[row.name, 'signal'] == 1 else row['Low'] * 0.95, axis=1)
    
    return signals[['signal', 'stop', 'target']]
