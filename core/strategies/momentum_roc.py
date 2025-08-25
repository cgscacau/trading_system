import pandas as pd
import ta

MOMENTUM_ROC_PARAMS = {'period': 12, 'threshold': 0}

def momentum_roc_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)
    period = params.get('period', 12)
    threshold = params.get('threshold', 0)

    roc = ta.momentum.roc(df_s['Close'], window=period)

    signals['signal'] = 0
    signals.loc[roc > threshold, 'signal'] = 1
    signals.loc[roc < -threshold, 'signal'] = -1
    
    signals['stop'] = df_s.apply(lambda r: r['Low'] * 0.98 if signals.loc[r.name]['signal'] == 1 else r['High'] * 1.02, axis=1)
    signals['target'] = pd.NA

    return signals[['signal', 'stop', 'target']]
