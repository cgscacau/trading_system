import pandas as pd

CANDLE_PATTERNS_PARAMS = {}

def candle_patterns_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)
    
    engulfing_buy = (df_s['Close'] > df_s['Open']) & \
                    (df_s['Close'].shift(1) < df_s['Open'].shift(1)) & \
                    (df_s['Open'] < df_s['Open'].shift(1)) & \
                    (df_s['Close'] > df_s['Close'].shift(1))
    
    signals['signal'] = 0
    signals.loc[engulfing_buy, 'signal'] = 1
    
    signals['stop'] = df_s.apply(lambda r: r['Low'] * 0.99 if signals.loc[r.name]['signal'] == 1 else pd.NA, axis=1)
    signals['target'] = df_s.apply(lambda r: r['High'] * 1.05 if signals.loc[r.name]['signal'] == 1 else pd.NA, axis=1)
    
    return signals[['signal', 'stop', 'target']]
