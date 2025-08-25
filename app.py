import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import date
import numpy as np
import itertools

# --- Bloco de Importação das Estratégias ---
from core.strategies.invented_strategies import vol_regime_switch_strategy, meta_ensemble_strategy, pullback_trend_bias_strategy
from core.strategies.standard_strategies import sma_crossover_strategy, ema_crossover_strategy, rsi_strategy, macd_strategy, bollinger_mean_reversion_strategy, bollinger_breakout_strategy, adx_dmi_strategy, donchian_breakout_strategy

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Lab de Estratégias de Trading", layout="wide")
st.title("📈 Lab de Estratégias & Sizing BRL→USD")
st.markdown("Teste, compare e dimensione estratégias de trading com gestão de risco realista.")

# --- DICIONÁRIOS E LISTAS ---
ALL_STRATEGIES = {
    "Cruzamento de Médias Móveis (SMA)": sma_crossover_strategy, "Cruzamento de Médias Móveis (EMA)": ema_crossover_strategy,
    "Índice de Força Relativa (RSI)": rsi_strategy, "MACD": macd_strategy,
    "Reversão à Média (Bandas de Bollinger)": bollinger_mean_reversion_strategy, "Rompimento (Bandas de Bollinger)": bollinger_breakout_strategy,
    "Rompimento (Canais de Donchian)": donchian_breakout_strategy, "ADX + DMI": adx_dmi_strategy,
    "Meta-Ensemble (EMA+RSI)": meta_ensemble_strategy, "Pullback em Tendência": pullback_trend_bias_strategy,
    "Switch de Regime de Volatilidade": vol_regime_switch_strategy,
}

# NOVO: Apenas estratégias com parâmetros ajustáveis para o otimizador
OPTIMIZABLE_STRATEGIES = {
    "Cruzamento de Médias Móveis (SMA)": sma_crossover_strategy, "Cruzamento de Médias Móveis (EMA)": ema_crossover_strategy,
    "Índice de Força Relativa (RSI)": rsi_strategy, "MACD": macd_strategy,
    "Reversão à Média (Bandas de Bollinger)": bollinger_mean_reversion_strategy, "Rompimento (Bandas de Bollinger)": bollinger_breakout_strategy,
    "Rompimento (Canais de Donchian)": donchian_breakout_strategy, "ADX + DMI": adx_dmi_strategy,
}

PRESET_TICKERS = {
    "Ações Brasileiras (IBOV)": "PETR4.SA\nVALE3.SA\nITUB4.SA\nBBDC4.SA\nBBAS3.SA\nITSA4.SA\nWEGE3.SA\nJBSS3.SA",
    "Criptomoedas": "BTC-USD\nETH-USD\nSOL-USD\nXRP-USD\nDOGE-USD",
    "Ações Americanas (Tech)": "AAPL\nMSFT\nGOOGL\nAMZN\nNVDA\nTSLA",
}

# --- FUNÇÕES DE LÓGICA ---
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty: return None
        new_cols = [col[0].capitalize() if isinstance(col, tuple) else col.capitalize() for col in data.columns]
        data.columns = new_cols
        return data
    except Exception: return None

def calculate_performance(results_df):
    results_df['signal_change'] = results_df['signal'].diff()
    entries = results_df[(results_df['signal_change'] != 0) & (results_df['signal'] != 0)]
    exits = results_df[(results_df['signal_change'] != 0) & (results_df['signal'].shift(1) != 0)]
    
    if len(entries) == 0 or len(exits) == 0:
        return {"returns": pd.Series(dtype=float), "metrics": {"Total Return (%)": 0, "Win Rate (%)": 0, "Profit Factor": "N/A", "Total Trades": 0, "Max Drawdown (%)": 0}}

    trade_returns_list = []
    for entry_index, entry_trade in entries.iterrows():
        exit_trade = exits[exits.index > entry_index]
        if not exit_trade.empty:
            exit_price = exit_trade.iloc[0]['Close']
            entry_price = entry_trade['Close']
            signal = entry_trade['signal']
            if signal == 1: trade_returns_list.append((exit_price / entry_price) - 1)
            elif signal == -1: trade_returns_list.append((entry_price / exit_price) - 1)
    
    trade_returns = pd.Series(trade_returns_list)
    if trade_returns.empty:
        return {"returns": pd.Series(dtype=float), "metrics": {"Total Return (%)": 0, "Win Rate (%)": 0, "Profit Factor": "N/A", "Total Trades": 0, "Max Drawdown (%)": 0}}
        
    wins, losses = trade_returns[trade_returns > 0], trade_returns[trade_returns < 0]
    gross_profit, gross_loss = wins.sum(), abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    cumulative_returns = (1 + trade_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    
    metrics = {"Total Return (%)": (cumulative_returns.iloc[-1] - 1) * 100, "Win Rate (%)": (len(wins) / len(trade_returns) * 100),
               "Profit Factor": profit_factor, "Total Trades": len(trade_returns), "Max Drawdown (%)": (drawdown.min() * 100) if not pd.isna(drawdown.min()) else 0}
    return {"returns": trade_returns, "metrics": metrics}

# --- INTERFACE DO USUÁRIO ---
st.sidebar.header("Modo de Operação")
operation_mode = st.sidebar.selectbox("Escolha o modo", ["Backtest de Ativo Único", "Screener de Múltiplos Ativos", "Otimizador Walk-Forward"])
params = {}

if operation_mode == "Backtest de Ativo Único":
    st.sidebar.header("Parâmetros do Backtest")
    preset_selection = st.sidebar.selectbox("Listas Predefinidas", list(PRESET_TICKERS.keys()))
    ticker = st.sidebar.text_input("Ativo", PRESET_TICKERS[preset_selection].split('\n')[0])
    start_date = st.sidebar.date_input("Data de Início", date(2022, 1, 1))
    end_date = st.sidebar.date_input("Data de Fim", date.today())
    trade_direction = st.sidebar.selectbox("Direção do Trade", ["Comprado e Vendido", "Apenas Comprado", "Apenas Vendido"])
    selected_strategy_name = st.sidebar.selectbox("Escolha a Estratégia", list(ALL_STRATEGIES.keys()))

    st.sidebar.header("Parâmetros da Estratégia")
    if "SMA" in selected_strategy_name or "EMA" in selected_strategy_name:
        params['short_window'] = st.sidebar.number_input("Janela Curta", value=20, min_value=1, step=1)
        params['long_window'] = st.sidebar.number_input("Janela Longa", value=50, min_value=1, step=1)
    elif "RSI" in selected_strategy_name:
        params['window'] = st.sidebar.number_input("Janela do RSI", value=14, min_value=1, step=1)
        params['buy_level'] = st.sidebar.number_input("Nível de Compra", value=30)
        params['sell_level'] = st.sidebar.number_input("Nível de Venda", value=70)
    elif "MACD" in selected_strategy_name:
        params['window_fast'] = st.sidebar.number_input("Janela Rápida", value=12)
        params['window_slow'] = st.sidebar.number_input("Janela Lenta", value=26)
        params['window_sign'] = st.sidebar.number_input("Janela do Sinal", value=9)
    # ... outros elif para outras estratégias ...

    if st.sidebar.button("Executar Backtest"):
        data = load_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            st.header(f"Resultados para {ticker} com a estratégia '{selected_strategy_name}'")
            strategy_function = ALL_STRATEGIES[selected_strategy_name]
            results = strategy_function(data.copy(), **params)
            
            if trade_direction == "Apenas Comprado": results.loc[results['signal'] == -1, 'signal'] = 0
            elif trade_direction == "Apenas Vendido": results.loc[results['signal'] == 1, 'signal'] = 0

            st.subheader("Resumo da Performance Histórica")
            performance = calculate_performance(results.copy())['metrics']
            # ... (código de exibição de métricas) ...

# ... O resto do código para o modo Backtest e Screener continua aqui ...

elif operation_mode == "Otimizador Walk-Forward":
    st.sidebar.header("Parâmetros da Otimização")
    ticker_opt = st.sidebar.text_input("Ativo para Otimizar", "PETR4.SA")
    start_date_opt = st.sidebar.date_input("Data de Início (Período Completo)", date(2020, 1, 1))
    end_date_opt = st.sidebar.date_input("Data de Fim", date.today())
    strategy_opt_name = st.sidebar.selectbox("Estratégia para Otimizar", list(OPTIMIZABLE_STRATEGIES.keys())) # USA A LISTA NOVA

    st.sidebar.subheader("Janelas Walk-Forward")
    in_sample_years = st.sidebar.slider("Anos para Otimização (In-Sample)", 1, 5, 2)
    out_of_sample_months = st.sidebar.slider("Meses para Teste (Out-of-Sample)", 3, 12, 6)

    st.sidebar.header("Intervalo dos Parâmetros")
    st.sidebar.warning("Use intervalos pequenos para evitar lentidão excessiva!")
    param_ranges = {}
    
    # --- PAINEL DE PARÂMETROS COMPLETO E CORRIGIDO ---
    if "SMA" in strategy_opt_name or "EMA" in strategy_opt_name:
        param_ranges['short_window'] = st.sidebar.slider("Intervalo Janela Curta", 5, 40, (10, 20), step=5)
        param_ranges['long_window'] = st.sidebar.slider("Intervalo Janela Longa", 40, 100, (40, 60), step=5)
    elif "RSI" in strategy_opt_name:
        param_ranges['window'] = st.sidebar.slider("Intervalo Janela RSI", 7, 21, (10, 14), step=1)
        param_ranges['buy_level'] = st.sidebar.slider("Intervalo Nível Compra", 20, 40, (25, 35), step=5)
        param_ranges['sell_level'] = st.sidebar.slider("Intervalo Nível Venda", 60, 80, (65, 75), step=5)
    elif "MACD" in strategy_opt_name:
        param_ranges['window_fast'] = st.sidebar.slider("Intervalo Janela Rápida", 5, 20, (10, 15))
        param_ranges['window_slow'] = st.sidebar.slider("Intervalo Janela Lenta", 20, 40, (25, 30))
        param_ranges['window_sign'] = st.sidebar.slider("Intervalo Janela Sinal", 5, 15, (8, 12))
    elif "Bollinger" in strategy_opt_name:
        param_ranges['window'] = st.sidebar.slider("Intervalo Janela", 15, 30, (18, 22))
        param_ranges['window_dev'] = st.sidebar.slider("Intervalo Desvio Padrão", 1.8, 3.0, (2.0, 2.5), step=0.1)
    elif "Donchian" in strategy_opt_name:
        param_ranges['window'] = st.sidebar.slider("Intervalo Janela", 15, 50, (20, 30), step=5)
    elif "ADX" in strategy_opt_name:
        param_ranges['window'] = st.sidebar.slider("Intervalo Janela", 10, 20, (12, 16))
        param_ranges['adx_threshold'] = st.sidebar.slider("Intervalo Limiar ADX", 20, 40, (22, 28), step=2)

    if st.sidebar.button("Iniciar Otimização Walk-Forward"):
        full_data = load_data(ticker_opt, start_date_opt, end_date_opt)
        if full_data is not None and not full_data.empty:
            st.header(f"Otimização Walk-Forward de '{strategy_opt_name}' para {ticker_opt}")
            # ... (Lógica de otimização, que agora funcionará pois param_ranges não estará vazio) ...
