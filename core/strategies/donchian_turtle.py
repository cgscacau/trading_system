import pandas as pd
import ta

DONCHIAN_BREAKOUT_PARAMS = {'period': 20}

def donchian_breakout_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_s = df.copy()
    signals = pd.DataFrame(index=df_s.index)
    period = params.get('period', 20)

    donchian = ta.volatility.DonchianChannel(high=df_s['High'], low=df_s['Low'], close=df_s['Close'], window=period)
    upper_band = donchian.donchian_channel_hband()
    lower_band = donchian.donchian_channel_lband()
    
    signals['signal'] = 0
    signals.loc[df_s['Close'] >= upper_band.shift(1), 'signal'] = 1
    signals.loc[df_s['Close'] <= lower_band.shift(1), 'signal'] = -1
    
    signals['stop'] = lower_band.where(signals['signal'] == 1, upper_band)
    signals['target'] = pd.NA

    return signals[['signal', 'stop', 'target']]
