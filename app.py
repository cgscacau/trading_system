import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import date

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
    "Reversão à Média (Bandas de Bollinger)": bollinger_mean_reversion_strategy, "Rompimento (Bandas de Bollinger)": bollinger_breakout_strategy,
    "Rompimento (Canais de Donchian)": donchian_breakout_strategy, "ADX + DMI": adx_dmi_strategy,
    "Meta-Ensemble (EMA+RSI)": meta_ensemble_strategy, "Pullback em Tendência": pullback_trend_bias_strategy,
    "Switch de Regime de Volatilidade": vol_regime_switch_strategy,
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
    trades = results_df[results_df['signal'] != results_df['signal'].shift(1)]
    trades = trades[trades['signal'] != 0]
    if len(trades) < 2: return {"Total Return (%)": 0, "Win Rate (%)": 0, "Profit Factor": "N/A", "Total Trades": 0, "Max Drawdown (%)": 0}
    actual_trades = trades['Close'].pct_change().dropna()[trades['signal'].shift(1) != trades['signal']].iloc[1:]
    if actual_trades.empty: return {"Total Return (%)": 0, "Win Rate (%)": 0, "Profit Factor": "N/A", "Total Trades": 0, "Max Drawdown (%)": 0}
    wins = actual_trades[actual_trades > 0]
    losses = actual_trades[actual_trades < 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    cumulative_returns = (1 + actual_trades).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return {"Total Return (%)": ((1 + actual_trades).prod() - 1) * 100, "Win Rate (%)": (len(wins) / len(actual_trades) * 100) if len(actual_trades) > 0 else 0,
            "Profit Factor": profit_factor, "Total Trades": len(actual_trades), "Max Drawdown (%)": (drawdown.min() * 100) if not pd.isna(drawdown.min()) else 0}

# --- INTERFACE DO USUÁRIO ---
st.sidebar.header("Modo de Operação")
operation_mode = st.sidebar.selectbox("Escolha o modo", ["Backtest de Ativo Único", "Screener de Múltiplos Ativos"])

if operation_mode == "Backtest de Ativo Único":
    st.sidebar.header("Parâmetros do Backtest")
    preset_selection = st.sidebar.selectbox("Listas Predefinidas", list(PRESET_TICKERS.keys()))
    ticker = st.sidebar.text_input("Ativo", PRESET_TICKERS[preset_selection].split('\n')[0])
    start_date = st.sidebar.date_input("Data de Início", date(2022, 1, 1))
    end_date = st.sidebar.date_input("Data de Fim", date.today())
    selected_strategy_name = st.sidebar.selectbox("Escolha a Estratégia", list(STRATEGIES.keys()))

    st.sidebar.header("Parâmetros da Estratégia")
    params = {}
    if "SMA" in selected_strategy_name or "EMA" in selected_strategy_name:
        params['short_window'] = st.sidebar.number_input("Janela Curta", value=20, min_value=1, step=1)
        params['long_window'] = st.sidebar.number_input("Janela Longa", value=50, min_value=1, step=1)
    elif "RSI" in selected_strategy_name:
        params['window'] = st.sidebar.number_input("Janela do RSI", value=14, min_value=1, step=1)
        params['buy_level'] = st.sidebar.number_input("Nível de Compra", value=30, min_value=1, max_value=100)
        params['sell_level'] = st.sidebar.number_input("Nível de Venda", value=70, min_value=1, max_value=100)

    if st.sidebar.button("Executar Backtest"):
        data = load_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            st.header(f"Resultados para {ticker} com a estratégia '{selected_strategy_name}'")
            strategy_function = STRATEGIES[selected_strategy_name]
            results = strategy_function(data.copy(), **params)
            
            st.subheader("Resumo da Performance Histórica")
            performance = calculate_performance(results)
            cols = st.columns(5)
            cols[0].metric("Retorno Total", f"{performance['Total Return (%)']:.2f}%")
            cols[1].metric("Win Rate", f"{performance['Win Rate (%)']:.2f}%")
            cols[2].metric("Profit Factor", f"{performance['Profit Factor']:.2f}" if isinstance(performance['Profit Factor'], (int, float)) else "N/A")
            cols[3].metric("Nº de Trades", performance['Total Trades'])
            cols[4].metric("Max Drawdown", f"{performance['Max Drawdown (%)']:.2f}%")

            st.subheader("Sinal Atual")
            last_row = results.iloc[-1]
            if last_row['signal'] == 1:
                st.success(f"🟢 SINAL DE COMPRA ATIVO")
                st.markdown(f"**Entrada:** `{last_row['Close']:.2f}` | **Stop:** `{last_row['stop']:.2f}` | **Alvo:** `{last_row['target']:.2f}`")
            elif last_row['signal'] == -1:
                st.error(f"🔴 SINAL DE VENDA ATIVO")
                st.markdown(f"**Entrada:** `{last_row['Close']:.2f}` | **Stop:** `{last_row['stop']:.2f}` | **Alvo:** `{last_row['target']:.2f}`")
            else:
                st.info("⚪ SINAL NEUTRO / AGUARDAR")

            st.subheader("Gráfico de Operações")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=results.index, open=results['Open'], high=results['High'], low=results['Low'], close=results['Close'], name='Preço'))
            trades = results[results['signal'] != results['signal'].shift(1)]
            for i in range(len(trades)):
                trade_entry = trades.iloc[i]
                color = "green" if trade_entry['signal'] == 1 else "red"
                fig.add_trace(go.Scatter(x=[trade_entry.name], y=[trade_entry['Close']], mode='markers', marker=dict(color=color, symbol='circle', size=12, line=dict(color='white', width=2)), name=f"Entrada {i+1}"))
            
            if last_row['signal'] != 0:
                fig.add_hline(y=last_row['stop'], line_dash="dash", line_color="orange", annotation_text="STOP ATUAL", annotation_position="bottom right")
                fig.add_hline(y=last_row['target'], line_dash="dash", line_color="cyan", annotation_text="ALVO ATUAL", annotation_position="top right")

            fig.update_layout(title=f"Sinais de Trading para {ticker}", xaxis_title="Data", yaxis_title="Preço", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Ver dados e operações"):
                st.dataframe(results)

else: # Modo Screener
    st.sidebar.header("Parâmetros do Screener")
    preset_choice = st.sidebar.selectbox("Carregar Lista Predefinida", list(PRESET_TICKERS.keys()))
    tickers_input = st.sidebar.text_area("Ativos para Rastrear (um por linha)", PRESET_TICKERS[preset_choice], height=200)
    start_date_scr = st.sidebar.date_input("Data de Início", date(2022, 1, 1))
    end_date_scr = st.sidebar.date_input("Data de Fim", date.today())
    
    if st.sidebar.button("Executar Screener"):
        tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
        st.header("Resultados do Screener")
        
        all_results, failed_tickers = [], []
        progress_bar = st.progress(0, text="Rastreando ativos...")

        for i, ticker in enumerate(tickers):
            data = load_data(ticker, start_date_scr, end_date_scr)
            if data is not None and not data.empty:
                for strategy_name, strategy_func in STRATEGIES.items():
                    results = strategy_func(data.copy())
                    performance = calculate_performance(results)
                    last_signal = results['signal'].iloc[-1]
                    all_results.append({"Ativo": ticker, "Estratégia": strategy_name, "Sinal Atual": "COMPRA" if last_signal == 1 else "VENDA" if last_signal == -1 else "NEUTRO",
                                        "Retorno Total (%)": performance['Total Return (%)'], "Win Rate (%)": performance['Win Rate (%)'],
                                        "Profit Factor": performance['Profit Factor'], "Nº Trades": performance['Total Trades']})
            else:
                failed_tickers.append(ticker)
            progress_bar.progress((i + 1) / len(tickers), text=f"Analisando {ticker}...")
        
        progress_bar.empty()
        
        if failed_tickers:
            st.warning(f"Não foi possível carregar dados para os seguintes ativos: {', '.join(failed_tickers)}")
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            st.subheader("Tabela de Oportunidades")
            st.info("Clique nos cabeçalhos das colunas para ordenar e encontrar as melhores combinações.")
            
            def highlight_signals(s):
                return ['background-color: #2E7D32' if v == 'COMPRA' else ('background-color: #C62828' if v == 'VENDA' else '') for v in s]

            st.dataframe(results_df.style.apply(highlight_signals, subset=['Sinal Atual'])
                                         .format({"Retorno Total (%)": "{:.2f}%", "Win Rate (%)": "{:.2f}%",
                                                  "Profit Factor": lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
                                         }), use_container_width=True)
