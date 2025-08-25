import pandas as pd
import ta

ADX_DMI_PARAMS = {'period': 14, 'adx_threshold': 25}

def adx_dmi_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)
    period = params.get('period', 14)
    adx_threshold = params.get('adx_threshold', 25)

    adx_indicator = ta.trend.ADXIndicator(high=df_s['High'], low=df_s['Low'], close=df_s['Close'], window=period)
    adx = adx_indicator.adx()
    dmi_pos = adx_indicator.adx_pos()
    dmi_neg = adx_indicator.adx_neg()
    
    signals['signal'] = 0
    strong_trend = adx > adx_threshold
    
    signals.loc[strong_trend & (dmi_pos > dmi_neg), 'signal'] = 1
    signals.loc[strong_trend & (dmi_neg > dmi_pos), 'signal'] = -1

    signals['stop'] = df_s.apply(lambda r: r['Low'] * 0.98 if signals.loc[r.name]['signal'] == 1 else r['High'] * 1.02, axis=1)
    signals['target'] = pd.NA
    
    return signals[['signal', 'stop', 'target']]
