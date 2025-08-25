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

# --- DICIONÁRIO DE ESTRATÉGIAS ---
STRATEGIES = {
    "Cruzamento de Médias Móveis (SMA)": sma_crossover_strategy,
    "Cruzamento de Médias Móveis (EMA)": ema_crossover_strategy,
    "Índice de Força Relativa (RSI)": rsi_strategy,
    "MACD": macd_strategy,
    "Reversão à Média (Bandas de Bollinger)": bollinger_mean_reversion_strategy,
    "Rompimento (Bandas de Bollinger)": bollinger_breakout_strategy,
    "Rompimento (Canais de Donchian)": donchian_breakout_strategy,
    "ADX + DMI": adx_dmi_strategy,
    "Meta-Ensemble (EMA+RSI)": meta_ensemble_strategy,
    "Pullback em Tendência": pullback_trend_bias_strategy,
    "Switch de Regime de Volatilidade": vol_regime_switch_strategy,
}

# --- FUNÇÕES DE LÓGICA ---

@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            st.error(f"Não foi possível obter dados para o ativo '{ticker}'. Verifique o símbolo.")
            return None
        data.columns = [col.capitalize() for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os dados: {e}")
        return None

def calculate_performance(results_df):
    """Calcula as métricas de performance de um backtest."""
    trades = results_df[results_df['signal'] != results_df['signal'].shift(1)]
    trades = trades[trades['signal'] != 0]

    if len(trades) < 2:
        return {
            "Total Return (%)": 0, "Win Rate (%)": 0, "Profit Factor": 0,
            "Total Trades": len(trades), "Max Drawdown (%)": 0
        }

    trades['pnl_pct'] = trades['Close'].pct_change()
    # Considera apenas os PnLs de fechamento de posição (quando o sinal muda)
    trade_returns = trades['pnl_pct'][trades['signal'] != trades['signal'].shift(1)].dropna()
    
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    # Cálculo do Retorno Total
    total_return = (1 + trade_returns).prod() - 1
    
    # Win Rate
    win_rate = (len(wins) / len(trade_returns) * 100) if len(trade_returns) > 0 else 0
    
    # Profit Factor
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max Drawdown
    cumulative_returns = (1 + trade_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min() * 100

    return {
        "Total Return (%)": total_return * 100,
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
        "Total Trades": len(trade_returns),
        "Max Drawdown (%)": max_drawdown
    }

# --- INTERFACE DO USUÁRIO ---

st.sidebar.header("Modo de Operação")
operation_mode = st.sidebar.selectbox("Escolha o modo", ["Backtest de Ativo Único", "Screener de Múltiplos Ativos"])

if operation_mode == "Backtest de Ativo Único":
    st.sidebar.header("Parâmetros do Backtest")
    ticker = st.sidebar.text_input("Ativo (símbolo do Yahoo Finance)", "BBAS3.SA")
    start_date = st.sidebar.date_input("Data de Início", date(2024, 1, 1))
    end_date = st.sidebar.date_input("Data de Fim", date.today())
    selected_strategy_name = st.sidebar.selectbox("Escolha a Estratégia", list(STRATEGIES.keys()))

    st.sidebar.header("Parâmetros da Estratégia")
    params = {}
    if "SMA" in selected_strategy_name:
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

            # --- MÓDULO 1: PAINEL DE PERFORMANCE ---
            st.subheader("Resumo da Performance Histórica")
            performance = calculate_performance(results)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Retorno Total", f"{performance['Total Return (%)']:.2f}%")
            col2.metric("Win Rate", f"{performance['Win Rate (%)']:.2f}%")
            col3.metric("Profit Factor", f"{performance['Profit Factor']:.2f}")
            col4.metric("Nº de Trades", performance['Total Trades'])
            col5.metric("Max Drawdown", f"{performance['Max Drawdown (%)']:.2f}%")

            # --- MÓDULO 2: SINAL ATUAL ---
            st.subheader("Sinal Atual")
            last_signal = results['signal'].iloc[-1]
            if last_signal == 1:
                st.success("🟢 SINAL DE COMPRA ATIVO")
            elif last_signal == -1:
                st.error("🔴 SINAL DE VENDA ATIVO")
            else:
                st.info("⚪ SINAL NEUTRO / AGUARDAR")

            # --- VISUALIZAÇÃO DOS RESULTADOS ---
            st.subheader("Gráfico de Operações")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=results.index, open=results['Open'], high=results['High'], low=results['Low'], close=results['Close'], name='Preço'))
            buy_signals = results[results['signal'] == 1]
            sell_signals = results[results['signal'] == -1]
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Compra'))
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Venda'))
            fig.update_layout(title=f"Sinais de Trading para {ticker}", xaxis_title="Data", yaxis_title="Preço", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Ver dados e operações"):
                st.dataframe(results)

else: # Modo Screener
    st.sidebar.header("Parâmetros do Screener")
    tickers_input = st.sidebar.text_area("Ativos para Rastrear (um por linha)", "PETR4.SA\nVALE3.SA\nITUB4.SA\nBBDC4.SA\nBBAS3.SA")
    start_date_scr = st.sidebar.date_input("Data de Início", date(2024, 1, 1))
    end_date_scr = st.sidebar.date_input("Data de Fim", date.today())
    
    if st.sidebar.button("Executar Screener"):
        tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
        st.header("Resultados do Screener")
        
        all_results = []
        progress_bar = st.progress(0, text="Rastreando ativos...")

        for i, ticker in enumerate(tickers):
            data = load_data(ticker, start_date_scr, end_date_scr)
            if data is not None:
                for strategy_name, strategy_func in STRATEGIES.items():
                    results = strategy_func(data.copy())
                    performance = calculate_performance(results)
                    last_signal = results['signal'].iloc[-1]
                    
                    all_results.append({
                        "Ativo": ticker,
                        "Estratégia": strategy_name,
                        "Sinal Atual": "COMPRA" if last_signal == 1 else "VENDA" if last_signal == -1 else "NEUTRO",
                        "Retorno Total (%)": performance['Total Return (%)'],
                        "Win Rate (%)": performance['Win Rate (%)'],
                        "Profit Factor": performance['Profit Factor'],
                        "Nº Trades": performance['Total Trades']
                    })
            progress_bar.progress((i + 1) / len(tickers), text=f"Analisando {ticker}...")
        
        progress_bar.empty()
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            st.subheader("Tabela de Oportunidades")
            st.info("Clique nos cabeçalhos das colunas para ordenar e encontrar as melhores combinações.")
            
            # Estilização para destacar sinais de COMPRA
            def highlight_buy(s):
                return ['background-color: #2E7D32' if v == 'COMPRA' else '' for v in s]

            st.dataframe(results_df.style.apply(highlight_buy, subset=['Sinal Atual'])
                                         .format({
                                             "Retorno Total (%)": "{:.2f}",
                                             "Win Rate (%)": "{:.2f}",
                                             "Profit Factor": "{:.2f}"
                                         }), use_container_width=True)
