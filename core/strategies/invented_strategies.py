import pandas as pd
from .moving_averages import ema_crossover_strategy, EMA_CROSSOVER_PARAMS
from .rsi_strategies import rsi_standard_strategy, RSI_STANDARD_PARAMS
from .bollinger_bands import bollinger_mean_reversion_strategy, BOLLINGER_MEAN_REVERSION_PARAMS

# Mantenha suas outras estratégias e PARAMS aqui...
META_ENSEMBLE_PARAMS = {} # Geralmente não precisa de params

def meta_ensemble_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Combina os sinais de múltiplas estratégias para um sinal mais robusto.
    """
    df_s = df.copy()
    
    # CORREÇÃO: Inicializa o DataFrame de sinais com o índice de datas correto
    signals = pd.DataFrame(index=df_s.index)
    signals['signal'] = 0
    
    # Gera sinais das estratégias base
    ema_signals = ema_crossover_strategy(df_s, EMA_CROSSOVER_PARAMS)
    rsi_signals = rsi_standard_strategy(df_s, RSI_STANDARD_PARAMS)
    bb_signals = bollinger_mean_reversion_strategy(df_s, BOLLINGER_MEAN_REVERSION_PARAMS)
    
    # Lógica de Votação/Ensemble
    # Compra forte: se EMA e (RSI ou BB) derem compra
    signals.loc[(ema_signals['signal'] == 1) & ((rsi_signals['signal'] == 1) | (bb_signals['signal'] == 1)), 'signal'] = 1
    
    # Venda forte: se EMA e (RSI ou BB) derem venda
    signals.loc[(ema_signals['signal'] == -1) & ((rsi_signals['signal'] == -1) | (bb_signals['signal'] == -1)), 'signal'] = -1
    
    # Adiciona stop/target genéricos (pode ser melhorado)
    signals['stop'] = df_s.apply(lambda row: row['Low'] * 0.98 if row['signal'] == 1 else row['High'] * 1.02, axis=1)
    signals['target'] = df_s.apply(lambda row: row['High'] * 1.05 if row['signal'] == 1 else row['Low'] * 0.95, axis=1)

    return signals[['signal', 'stop', 'target']]

# ... (mantenha o código das suas outras estratégias inventadas aqui)
