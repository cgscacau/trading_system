import pandas as pd
import ta

# Parâmetros default (pode ajustar conforme necessário)
BOLLINGER_BREAKOUT_PARAMS = {'period': 20, 'std_dev': 2}
BOLLINGER_MEAN_REVERSION_PARAMS = {'period': 20, 'std_dev': 2}

def bollinger_breakout_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Estratégia de rompimento de Bandas de Bollinger."""
    df_s = df.copy()
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2)

    # Adiciona as bandas de Bollinger ao DataFrame
    indicator_bb = ta.volatility.BollingerBands(close=df_s['Close'], window=period, window_dev=std_dev)
    df_s['bb_upper'] = indicator_bb.bollinger_hband()
    df_s['bb_lower'] = indicator_bb.bollinger_lband()
    
    df_s['signal'] = 0
    # Sinal de compra: Fechamento rompe a banda superior
    df_s.loc[df_s['Close'] > df_s['bb_upper'], 'signal'] = 1
    # Sinal de venda: Fechamento rompe a banda inferior
    df_s.loc[df_s['Close'] < df_s['bb_lower'], 'signal'] = -1
    
    df_s['stop'] = df_s.apply(lambda row: row['bb_upper'] if row['signal'] == -1 else (row['bb_lower'] if row['signal'] == 1 else pd.NA), axis=1)
    df_s['target'] = pd.NA # Breakout geralmente não tem alvo fixo

    return df_s[['signal', 'stop', 'target']]

def bollinger_mean_reversion_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Estratégia de reversão à média com Bandas de Bollinger."""
    df_s = df.copy()
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2)

    # CORREÇÃO: Passar df_s['Close'] (1-dimensional) para o indicador
    indicator_bb = ta.volatility.BollingerBands(close=df_s['Close'], window=period, window_dev=std_dev)
    df_s['bb_upper'] = indicator_bb.bollinger_hband()
    df_s['bb_lower'] = indicator_bb.bollinger_lband()
    df_s['bb_ma'] = indicator_bb.bollinger_mavg()
    
    df_s['signal'] = 0
    # Sinal de compra: Preço toca ou cruza abaixo da banda inferior
    df_s.loc[df_s['Close'] < df_s['bb_lower'], 'signal'] = 1
    # Sinal de venda: Preço toca ou cruza acima da banda superior
    df_s.loc[df_s['Close'] > df_s['bb_upper'], 'signal'] = -1
    # Sinal de saída/zerar: Preço volta para a média
    # (Esta parte pode ser implementada no backtester, mas aqui apenas geramos entradas)
    
    df_s['stop'] = df_s.apply(lambda row: row['Low'] * 0.98 if row['signal'] == 1 else (row['High'] * 1.02 if row['signal'] == -1 else pd.NA), axis=1)
    df_s['target'] = df_s['bb_ma'] # O alvo é a média móvel central

    return df_s[['signal', 'stop', 'target']]
