import pandas as pd
# --- CORREÇÃO APLICADA AQUI ---
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator

def vol_regime_switch_strategy(data: pd.DataFrame):
    """
    Estratégia que alterna entre seguir a tendência e reverter à média
    com base no regime de volatilidade do mercado.
    """
    # --- CORREÇÃO APLICADA AQUI ---
    atr_indicator = AverageTrueRange(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        window=14
    )
    data['atr'] = atr_indicator.average_true_range()
    data['atr_sma'] = data['atr'].rolling(window=50).mean()

    trend_sma = SMAIndicator(close=data['Close'], window=50).sma_indicator()
    data['trend_sma'] = trend_sma

    data['vol_regime'] = data['atr'] > data['atr_sma'] * 1.5

    data['signal'] = 0
    
    high_vol_buy = (data['vol_regime'] == True) & (data['Close'] < data['trend_sma'])
    high_vol_sell = (data['vol_regime'] == True) & (data['Close'] > data['trend_sma'])
    data.loc[high_vol_buy, 'signal'] = 1
    data.loc[high_vol_sell, 'signal'] = -1

    low_vol_buy = (data['vol_regime'] == False) & (data['Close'] > data['trend_sma'])
    low_vol_sell = (data['vol_regime'] == False) & (data['Close'] < data['trend_sma'])
    data.loc[low_vol_buy, 'signal'] = 1
    data.loc[low_vol_sell, 'signal'] = -1
    
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10

    return data

def meta_ensemble_strategy(data: pd.DataFrame):
    """
    Estratégia que combina os sinais de dois indicadores (EMA Crossover e RSI)
    para gerar um sinal final de negociação.
    """
    close_prices = data['Close']

    ema_short = EMAIndicator(close=close_prices, window=12).ema_indicator()
    ema_long = EMAIndicator(close=close_prices, window=26).ema_indicator()
    
    ema_signal = pd.Series(0, index=data.index)
    ema_signal[ema_short > ema_long] = 1
    ema_signal[ema_short < ema_long] = -1

    rsi = RSIIndicator(close=close_prices, window=14).rsi()
    
    rsi_signal = pd.Series(0, index=data.index)
    rsi_signal[rsi < 30] = 1
    rsi_signal[rsi > 70] = -1

    combined_votes = ema_signal + rsi_signal
    
    data['signal'] = 0
    data.loc[combined_votes >= 2, 'signal'] = 1
    data.loc[combined_votes <= -2, 'signal'] = -1
    
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10
    
    return data

def pullback_trend_bias_strategy(data: pd.DataFrame):
    """
    Estratégia que procura comprar em recuos (pullbacks) durante uma
    tendência de alta estabelecida.
    """
    close_prices = data['Close']
    low_prices = data['Low']

    data['trend'] = SMAIndicator(close=close_prices, window=50).sma_indicator()
    
    data['pullback'] = low_prices.rolling(window=5).min()

    data['signal'] = 0
    buy_condition = (data['Close'] > data['trend']) & (low_prices == data['pullback'])
    data.loc[buy_condition, 'signal'] = 1
    
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10

    return data
