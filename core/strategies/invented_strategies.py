import pandas as pd
from ta.volatility import ATRIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator

def vol_regime_switch_strategy(data: pd.DataFrame):
    """
    Estratégia que alterna entre seguir a tendência e reverter à média
    com base no regime de volatilidade do mercado.
    """
    # --- CORREÇÃO APLICADA ---
    # As funções de indicadores precisam de Séries (1D), não de DataFrames (2D).
    # Usamos data['High'] em vez de data[['High']], etc.
    atr_indicator = ATRIndicator(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        window=14
    )
    data['atr'] = atr_indicator.average_true_range()
    data['atr_sma'] = data['atr'].rolling(window=50).mean()

    # --- CORREÇÃO APLICADA ---
    trend_sma = SMAIndicator(close=data['Close'], window=50).sma_indicator()
    data['trend_sma'] = trend_sma

    # Define o regime de volatilidade
    data['vol_regime'] = data['atr'] > data['atr_sma'] * 1.5

    # Gera sinais
    data['signal'] = 0
    
    # Condições para o regime de alta volatilidade (reversão à média)
    high_vol_buy = (data['vol_regime'] == True) & (data['Close'] < data['trend_sma'])
    high_vol_sell = (data['vol_regime'] == True) & (data['Close'] > data['trend_sma'])
    data.loc[high_vol_buy, 'signal'] = 1
    data.loc[high_vol_sell, 'signal'] = -1

    # Condições para o regime de baixa volatilidade (seguidor de tendência)
    low_vol_buy = (data['vol_regime'] == False) & (data['Close'] > data['trend_sma'])
    low_vol_sell = (data['vol_regime'] == False) & (data['Close'] < data['trend_sma'])
    data.loc[low_vol_buy, 'signal'] = 1
    data.loc[low_vol_sell, 'signal'] = -1
    
    data['stop'] = data['Close'] * 0.95 # Exemplo: stop de 5%
    data['target'] = data['Close'] * 1.10 # Exemplo: alvo de 10%

    return data[['signal', 'stop', 'target']]

def meta_ensemble_strategy(data: pd.DataFrame):
    """
    Estratégia que combina os sinais de dois indicadores (EMA Crossover e RSI)
    para gerar um sinal final de negociação.
    """
    # --- CORREÇÃO APLICADA ---
    close_prices = data['Close'] # Seleciona a coluna como uma Série 1D

    # 1. Indicador EMA Crossover
    ema_short = EMAIndicator(close=close_prices, window=12).ema_indicator()
    ema_long = EMAIndicator(close=close_prices, window=26).ema_indicator()
    
    ema_signal = pd.Series(0, index=data.index)
    ema_signal[ema_short > ema_long] = 1
    ema_signal[ema_short < ema_long] = -1

    # 2. Indicador RSI
    rsi = RSIIndicator(close=close_prices, window=14).rsi()
    
    rsi_signal = pd.Series(0, index=data.index)
    rsi_signal[rsi < 30] = 1
    rsi_signal[rsi > 70] = -1

    # Combina os votos dos indicadores
    combined_votes = ema_signal + rsi_signal
    
    data['signal'] = 0
    data.loc[combined_votes >= 2, 'signal'] = 1   # Concordância forte de compra
    data.loc[combined_votes <= -2, 'signal'] = -1 # Concordância forte de venda
    
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10
    
    return data[['signal', 'stop', 'target']]

def pullback_trend_bias_strategy(data: pd.DataFrame):
    """
    Estratégia que procura comprar em recuos (pullbacks) durante uma
    tendência de alta estabelecida.
    """
    # --- CORREÇÃO APLICADA ---
    close_prices = data['Close']
    low_prices = data['Low']

    # 1. Identifica a tendência de alta (preço acima da média móvel longa)
    data['trend'] = SMAIndicator(close=close_prices, window=50).sma_indicator()
    
    # 2. Identifica um pullback (preço atinge a mínima dos últimos 5 períodos)
    data['pullback'] = low_prices.rolling(window=5).min()

    # Gera o sinal de compra apenas quando ambas as condições são verdadeiras
    data['signal'] = 0
    buy_condition = (data['Close'] > data['trend']) & (low_prices == data['pullback'])
    data.loc[buy_condition, 'signal'] = 1
    
    # Esta estratégia não define um sinal de venda claro, apenas de compra/manutenção.
    
    data['stop'] = data['Close'] * 0.95
    data['target'] = data['Close'] * 1.10

    return data[['signal', 'stop', 'target']]```

### Próximos Passos

1.  **Aplique a mesma lógica aos outros ficheiros:** A sua lista de erros mencionava muitas outras estratégias (MACD, Bollinger Bands, Donchian Breakout, etc.). Você precisa de abrir os ficheiros onde essas estratégias estão definidas e aplicar a mesma correção: encontre todas as chamadas a indicadores e certifique-se de que está a passar `data['Coluna']` em vez de `data[['Coluna']]`.

2.  **Verifique os erros de índice:** Se os erros `"None of [...] are in the [index]"` persistirem após esta correção, verifique a parte do seu código que carrega os dados. Use `print(seu_dataframe.columns)` e `print(seu_dataframe.head())` para garantir que os nomes das colunas são os esperados (`Open`, `High`, `Low`, `Close`, `Volume`) e que o índice está formatado corretamente como uma data.

Ao aplicar estas correções de forma sistemática, todos os erros que listou deverão ser resolvidos.
