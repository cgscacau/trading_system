import pandas as pd
import ta

# Parâmetros default
RSI_IFR2_PARAMS = {'period': 2, 'entry': 10, 'exit': 70}
RSI_STANDARD_PARAMS = {'period': 14, 'entry_buy': 30, 'entry_sell': 70}

def rsi_ifr2_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Estratégia IFR2 de Larry Connors."""
    df_s = df.copy()
    period = params.get('period', 2)
    entry_level = params.get('entry', 10)
    exit_level = params.get('exit', 70)

    # CORREÇÃO: Passar df_s['Close'] para o indicador
    df_s['rsi'] = ta.momentum.rsi(df_s['Close'], window=period)
    
    df_s['signal'] = 0
    # Condição de compra: RSI cruza para baixo do nível de entrada
    df_s.loc[df_s['rsi'] < entry_level, 'signal'] = 1
    # Condição de saída: RSI cruza para cima do nível de saída
    df_s.loc[df_s['rsi'] > exit_level, 'signal'] = -1 # Sinaliza para vender/zerar

    df_s['stop'] = df_s.apply(lambda row: row['Low'] * 0.97 if row['signal'] == 1 else pd.NA, axis=1)
    df_s['target'] = pd.NA # IFR2 geralmente não usa alvo

    return df_s[['signal', 'stop', 'target']]

def rsi_standard_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Estratégia de RSI padrão (sobrecompra/sobrevenda)."""
    df_s = df.copy()
    period = params.get('period', 14)
    entry_buy = params.get('entry_buy', 30)
    entry_sell = params.get('entry_sell', 70)

    # CORREÇÃO: Passar df_s['Close'] para o indicador
    df_s['rsi'] = ta.momentum.rsi(df_s['Close'], window=period)
    
    df_s['signal'] = 0
    df_s.loc[df_s['rsi'] < entry_buy, 'signal'] = 1
    df_s.loc[df_s['rsi'] > entry_sell, 'signal'] = -1
    
    df_s['stop'] = df_s.apply(lambda row: row['Low'] * 0.98 if row['signal'] == 1 else row['High'] * 1.02, axis=1)
    df_s['target'] = df_s.apply(lambda row: row['High'] * 1.05 if row['signal'] == 1 else row['Low'] * 0.95, axis=1)
    
    return df_s[['signal', 'stop', 'target']]
