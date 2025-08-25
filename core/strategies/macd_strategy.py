import pandas as pd
import ta

MACD_PARAMS = {'fast': 12, 'slow': 26, 'signal': 9}

def macd_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)

    macd_indicator = ta.trend.MACD(
        close=df_s['Close'],
        window_slow=params.get('slow', 26),
        window_fast=params.get('fast', 12),
        window_sign=params.get('signal', 9)
    )
    
    macd_line = macd_indicator.macd()
    signal_line = macd_indicator.macd_signal()
    
    signals['signal'] = 0
    signals.loc[macd_line > signal_line, 'signal'] = 1
    signals.loc[macd_line < signal_line, 'signal'] = -1
    
    signals['stop'] = df_s.apply(lambda r: r['Low'] * 0.98 if signals.loc[r.name]['signal'] == 1 else r['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda r: r['High'] * 1.05 if signals.loc[r.name]['signal'] == 1 else r['Low'] * 0.95, axis=1)
    
    return signals[['signal', 'stop', 'target']]
