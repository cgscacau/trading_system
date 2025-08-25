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
STRATEGIES = {
    "Cruzamento de Médias Móveis (SMA)": sma_crossover_strategy, "Cruzamento de Médias Móveis (EMA)": ema_crossover_strategy,
    "Índice de Força Relativa (RSI)": rsi_strategy, "MACD": macd_strategy,
    # Adicione outras estratégias aqui...
}
PRESET_TICKERS = {
    "Ações Brasileiras (IBOV)": "PETR4.SA\nVALE3.SA\nITUB4.SA\nBBDC4.SA\nBBAS3.SA",
    "Criptomoedas": "BTC-USD\nETH-USD\nSOL-USD",
    "Ações Americanas (Tech)": "AAPL\nMSFT\nGOOGL\nAMZN\nNVDA",
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

if operation_mode == "Otimizador Walk-Forward":
    st.sidebar.header("Parâmetros da Otimização")
    ticker_opt = st.sidebar.text_input("Ativo para Otimizar", "PETR4.SA")
    start_date_opt = st.sidebar.date_input("Data de Início (Período Completo)", date(2020, 1, 1))
    end_date_opt = st.sidebar.date_input("Data de Fim", date.today())
    strategy_opt_name = st.sidebar.selectbox("Estratégia para Otimizar", list(STRATEGIES.keys()))

    st.sidebar.subheader("Janelas Walk-Forward")
    in_sample_years = st.sidebar.slider("Anos para Otimização (In-Sample)", 1, 5, 2)
    out_of_sample_months = st.sidebar.slider("Meses para Teste (Out-of-Sample)", 3, 12, 6)

    st.sidebar.header("Intervalo dos Parâmetros")
    st.sidebar.warning("Use intervalos pequenos para evitar lentidão excessiva!")
    param_ranges = {}
    if "SMA" in strategy_opt_name or "EMA" in strategy_opt_name:
        param_ranges['short_window'] = st.sidebar.slider("Intervalo Janela Curta", 5, 40, (10, 20), step=5)
        param_ranges['long_window'] = st.sidebar.slider("Intervalo Janela Longa", 40, 100, (40, 60), step=5)
    elif "RSI" in strategy_opt_name:
        param_ranges['window'] = st.sidebar.slider("Intervalo Janela RSI", 7, 21, (10, 14), step=1)
        param_ranges['buy_level'] = st.sidebar.slider("Intervalo Nível Compra", 20, 40, (25, 35), step=5)
        param_ranges['sell_level'] = st.sidebar.slider("Intervalo Nível Venda", 60, 80, (65, 75), step=5)
    
    if st.sidebar.button("Iniciar Otimização Walk-Forward"):
        full_data = load_data(ticker_opt, start_date_opt, end_date_opt)
        if full_data is not None and not full_data.empty:
            st.header(f"Otimização Walk-Forward de '{strategy_opt_name}' para {ticker_opt}")
            
            # Geração das Janelas
            windows = []
            current_start = full_data.index[0]
            while True:
                in_sample_end = current_start + pd.DateOffset(years=in_sample_years)
                out_of_sample_end = in_sample_end + pd.DateOffset(months=out_of_sample_months)
                if out_of_sample_end > full_data.index[-1]: break
                windows.append((current_start, in_sample_end, out_of_sample_end))
                current_start += pd.DateOffset(months=out_of_sample_months)
            
            st.info(f"Analisando {len(windows)} janelas de otimização/teste. Isso pode ser demorado.")
            
            all_oos_returns = []
            walk_forward_summary = []
            progress_bar = st.progress(0, text="Analisando janelas...")

            strategy_function = STRATEGIES[strategy_opt_name]
            keys, values = zip(*param_ranges.items())
            
            for i, (start, mid, end) in enumerate(windows):
                in_sample_data = full_data.loc[start:mid]
                out_of_sample_data = full_data.loc[mid:end]
                
                # Otimização In-Sample
                best_params_in_sample = {}
                best_profit_factor = -1
                
                param_combinations = [dict(zip(keys, v)) for v in itertools.product(*(range(r[0], r[1] + 1, (r[1]-r[0])//5+1) for r in values))]
                
                for p_set in param_combinations:
                    if 'short_window' in p_set and p_set.get('short_window', 0) >= p_set.get('long_window', float('inf')): continue
                    results_is = strategy_function(in_sample_data.copy(), **p_set)
                    perf_is = calculate_performance(results_is)['metrics']
                    if isinstance(perf_is['Profit Factor'], (int, float)) and perf_is['Profit Factor'] > best_profit_factor:
                        best_profit_factor = perf_is['Profit Factor']
                        best_params_in_sample = p_set
                
                if not best_params_in_sample: continue

                # Validação Out-of-Sample
                results_oos = strategy_function(out_of_sample_data.copy(), **best_params_in_sample)
                performance_oos = calculate_performance(results_oos)
                all_oos_returns.append(performance_oos['returns'])
                
                summary = {"Período": f"{mid.date()} a {end.date()}"}
                summary.update(best_params_in_sample)
                summary.update(performance_oos['metrics'])
                walk_forward_summary.append(summary)
                
                progress_bar.progress((i + 1) / len(windows), text=f"Janela {i+1}/{len(windows)}: Melhores Params {best_params_in_sample}")

            progress_bar.empty()

            st.subheader("Performance Final (Out-of-Sample)")
            final_returns = pd.concat(all_oos_returns)
            final_performance = calculate_performance(pd.DataFrame({'Close': (1 + final_returns).cumprod(), 'signal': [1]*len(final_returns)}))['metrics'] # Gambiarra para reutilizar a função
            cols = st.columns(5)
            cols[0].metric("Retorno Total", f"{final_performance['Total Return (%)']:.2f}%")
            cols[1].metric("Win Rate", f"{final_performance['Win Rate (%)']:.2f}%")
            cols[2].metric("Profit Factor", f"{final_performance['Profit Factor']:.2f}" if isinstance(final_performance['Profit Factor'], (int, float)) else "N/A")
            cols[3].metric("Nº de Trades", final_performance['Total Trades'])
            cols[4].metric("Max Drawdown", f"{final_performance['Max Drawdown (%)']:.2f}%")

            st.subheader("Resumo da Otimização por Período")
            st.dataframe(pd.DataFrame(walk_forward_summary).style.format(precision=2), use_container_width=True)

# Os outros modos (Backtest e Screener) foram omitidos aqui para focar na nova funcionalidade, mas devem ser mantidos no seu ficheiro.
# Por favor, copie e cole o código completo que inclui todos os modos.
