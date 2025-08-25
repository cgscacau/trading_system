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

# MODO 1: BACKTEST DE ATIVO ÚNICO
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
        params['buy_level'] = st.sidebar.number_input("Nível de Compra", value=30, min_value=1, max_value=100)
        params['sell_level'] = st.sidebar.number_input("Nível de Venda", value=70, min_value=1, max_value=100)
    elif "MACD" in selected_strategy_name:
        params['window_fast'] = st.sidebar.number_input("Janela Rápida", value=12, min_value=1, step=1)
        params['window_slow'] = st.sidebar.number_input("Janela Lenta", value=26, min_value=1, step=1)
        params['window_sign'] = st.sidebar.number_input("Janela do Sinal", value=9, min_value=1, step=1)
    elif "Bollinger" in selected_strategy_name:
        params['window'] = st.sidebar.number_input("Janela", value=20, min_value=1, step=1)
        params['window_dev'] = st.sidebar.number_input("Desvios Padrão", value=2.0, min_value=0.1, step=0.1)
    elif "Donchian" in selected_strategy_name:
        params['window'] = st.sidebar.number_input("Janela", value=20, min_value=1, step=1)
    elif "ADX" in selected_strategy_name:
        params['window'] = st.sidebar.number_input("Janela", value=14, min_value=1, step=1)
        params['adx_threshold'] = st.sidebar.number_input("Limiar do ADX", value=25, min_value=1)
    
    if st.sidebar.button("Executar Backtest"):
        # ... Lógica do backtest único ...
        pass # Código completo abaixo

# MODO 2: SCREENER DE MÚLTIPLOS ATIVOS
elif operation_mode == "Screener de Múltiplos Ativos":
    st.sidebar.header("Parâmetros do Screener")
    preset_choice = st.sidebar.selectbox("Carregar Lista Predefinida", list(PRESET_TICKERS.keys()))
    tickers_input = st.sidebar.text_area("Ativos para Rastrear", PRESET_TICKERS[preset_choice], height=200)
    start_date_scr = st.sidebar.date_input("Data de Início", date(2022, 1, 1))
    end_date_scr = st.sidebar.date_input("Data de Fim", date.today())
    with st.sidebar.expander("Parâmetros Globais das Estratégias"):
        params = {
            'short_window': st.number_input("Janela Curta (SMA/EMA)", value=20),
            'long_window': st.number_input("Janela Longa (SMA/EMA)", value=50),
            # Adicione outros parâmetros globais aqui...
        }
    if st.sidebar.button("Executar Screener"):
        # ... Lógica do Screener ...
        pass # Código completo abaixo

# MODO 3: OTIMIZADOR WALK-FORWARD
elif operation_mode == "Otimizador Walk-Forward":
    st.sidebar.header("Parâmetros da Otimização")
    ticker_opt = st.sidebar.text_input("Ativo para Otimizar", "PETR4.SA")
    start_date_opt = st.sidebar.date_input("Data de Início (Período Completo)", date(2020, 1, 1))
    end_date_opt = st.sidebar.date_input("Data de Fim", date.today())
    strategy_opt_name = st.sidebar.selectbox("Estratégia para Otimizar", list(OPTIMIZABLE_STRATEGIES.keys()))
    st.sidebar.subheader("Janelas Walk-Forward")
    in_sample_years = st.sidebar.slider("Anos para Otimização (In-Sample)", 1, 5, 2)
    out_of_sample_months = st.sidebar.slider("Meses para Teste (Out-of-Sample)", 3, 12, 6)
    st.sidebar.header("Intervalo dos Parâmetros")
    st.sidebar.warning("Use intervalos pequenos para evitar lentidão!")
    param_ranges = {}
    if "SMA" in strategy_opt_name or "EMA" in strategy_opt_name:
        param_ranges['short_window'] = st.sidebar.slider("Intervalo Janela Curta", 5, 40, (10, 20), step=5)
        param_ranges['long_window'] = st.sidebar.slider("Intervalo Janela Longa", 40, 100, (40, 60), step=5)
    # Adicione mais `elif` para outras estratégias...
    
    if st.sidebar.button("Iniciar Otimização Walk-Forward"):
        full_data = load_data(ticker_opt, start_date_opt, end_date_opt)
        if full_data is not None and not full_data.empty:
            st.header(f"Otimização Walk-Forward de '{strategy_opt_name}' para {ticker_opt}")
            windows = []
            current_start = full_data.index[0]
            while True:
                in_sample_end = current_start + pd.DateOffset(years=in_sample_years)
                out_of_sample_end = in_sample_end + pd.DateOffset(months=out_of_sample_months)
                if out_of_sample_end > full_data.index[-1]: break
                windows.append((current_start, in_sample_end, out_of_sample_end))
                current_start += pd.DateOffset(months=out_of_sample_months)
            
            st.info(f"Analisando {len(windows)} janelas de otimização/teste.")
            all_oos_returns, walk_forward_summary = [], []
            progress_bar = st.progress(0, text="Analisando janelas...")
            strategy_function = OPTIMIZABLE_STRATEGIES[strategy_opt_name]
            keys, values = zip(*param_ranges.items())
            
            for i, (start, mid, end) in enumerate(windows):
                in_sample_data, out_of_sample_data = full_data.loc[start:mid], full_data.loc[mid:end]
                best_params_in_sample, best_profit_factor = {}, -np.inf
                
                param_combinations = [dict(zip(keys, v)) for v in itertools.product(*(np.arange(r[0], r[1] + (r[2] if len(r) > 2 else 1), (r[2] if len(r) > 2 else 1)) for r in values))]
                
                for p_set in param_combinations:
                    if 'short_window' in p_set and p_set.get('short_window', 0) >= p_set.get('long_window', float('inf')): continue
                    results_is = strategy_function(in_sample_data.copy(), **p_set)
                    perf_is = calculate_performance(results_is)['metrics']
                    if isinstance(perf_is['Profit Factor'], (int, float)) and perf_is['Profit Factor'] > best_profit_factor:
                        best_profit_factor, best_params_in_sample = perf_is['Profit Factor'], p_set
                
                if not best_params_in_sample: continue
                
                results_oos = strategy_function(out_of_sample_data.copy(), **best_params_in_sample)
                performance_oos = calculate_performance(results_oos)
                all_oos_returns.append(performance_oos['returns'])
                
                summary = {"Período": f"{mid.date()} a {end.date()}"}
                summary.update(best_params_in_sample)
                summary.update(performance_oos['metrics'])
                walk_forward_summary.append(summary)
                progress_bar.progress((i + 1) / len(windows), text=f"Janela {i+1}/{len(windows)}")
            
            progress_bar.empty()

            st.subheader("Performance Final (Out-of-Sample)")
            final_returns = pd.concat(all_oos_returns)
            
            # --- CORREÇÃO FINAL NA LÓGICA DE CÁLCULO ---
            if not final_returns.empty:
                wins = final_returns[final_returns > 0]
                losses = final_returns[final_returns < 0]
                gross_profit = wins.sum()
                gross_loss = abs(losses.sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                cumulative_returns = (1 + final_returns).cumprod()
                total_return_pct = (cumulative_returns.iloc[-1] - 1) * 100
                peak = cumulative_returns.expanding(min_periods=1).max()
                drawdown = (cumulative_returns / peak) - 1
                max_drawdown_pct = drawdown.min() * 100 if not pd.isna(drawdown.min()) else 0
                
                final_performance = {
                    "Total Return (%)": total_return_pct,
                    "Win Rate (%)": len(wins) / len(final_returns) * 100,
                    "Profit Factor": profit_factor,
                    "Total Trades": len(final_returns),
                    "Max Drawdown (%)": max_drawdown_pct
                }
            else:
                final_performance = {"Total Return (%)": 0, "Win Rate (%)": 0, "Profit Factor": "N/A", "Total Trades": 0, "Max Drawdown (%)": 0}
            
            cols = st.columns(5)
            cols[0].metric("Retorno Total", f"{final_performance['Total Return (%)']:.2f}%")
            cols[1].metric("Win Rate", f"{final_performance['Win Rate (%)']:.2f}%")
            cols[2].metric("Profit Factor", f"{final_performance['Profit Factor']:.2f}" if isinstance(final_performance['Profit Factor'], (int, float)) else "N/A")
            cols[3].metric("Nº de Trades", final_performance['Total Trades'])
            cols[4].metric("Max Drawdown", f"{final_performance['Max Drawdown (%)']:.2f}%")

            st.subheader("Resumo da Otimização por Período")
            summary_df = pd.DataFrame(walk_forward_summary)
            st.dataframe(summary_df.style.format(precision=2), use_container_width=True)
