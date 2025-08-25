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

# --- BARRA LATERAL (INPUTS DO UTILIZADOR) ---
st.sidebar.header("Parâmetros do Backtest")
ticker = st.sidebar.text_input("Ativo (digite qualquer símbolo do Yahoo Finance)", "PETR4.SA")
start_date = st.sidebar.date_input("Data de Início", date(2024, 1, 1))
end_date = st.sidebar.date_input("Data de Fim", date.today())
selected_strategy_name = st.sidebar.selectbox("Escolha a Estratégia", list(STRATEGIES.keys()))

# --- PARÂMETROS DINÂMICOS DAS ESTRATÉGIAS ---
st.sidebar.header("Parâmetros da Estratégia")
params = {}
# Adicione `elif` para outras estratégias que você queira customizar
if selected_strategy_name == "Cruzamento de Médias Móveis (SMA)":
    params['short_window'] = st.sidebar.number_input("Janela Curta", value=20, min_value=1, step=1)
    params['long_window'] = st.sidebar.number_input("Janela Longa", value=50, min_value=1, step=1)
elif selected_strategy_name == "Cruzamento de Médias Móveis (EMA)":
    params['short_window'] = st.sidebar.number_input("Janela Curta (EMA)", value=12, min_value=1, step=1)
    params['long_window'] = st.sidebar.number_input("Janela Longa (EMA)", value=26, min_value=1, step=1)
elif selected_strategy_name == "Índice de Força Relativa (RSI)":
    params['window'] = st.sidebar.number_input("Janela do RSI", value=14, min_value=1, step=1)
    params['buy_level'] = st.sidebar.number_input("Nível de Compra (Sobrevenda)", value=30, min_value=1, max_value=100)
    params['sell_level'] = st.sidebar.number_input("Nível de Venda (Sobrecompra)", value=70, min_value=1, max_value=100)

# --- FUNÇÃO PARA CARREGAR E PREPARAR DADOS (CORREÇÃO FINAL) ---
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            st.error(f"Não foi possível obter dados para o ativo '{ticker}'. Verifique o símbolo.")
            return None
        
        # --- CORREÇÃO FINAL AQUI ---
        # Esta lógica lida com nomes de colunas que são strings ou tuplas.
        new_cols = []
        for col in data.columns:
            if isinstance(col, tuple):
                new_cols.append(col[0].capitalize())
            else:
                new_cols.append(col.capitalize())
        data.columns = new_cols
        
        return data
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os dados: {e}")
        return None

# --- LÓGICA PRINCIPAL ---
if st.sidebar.button("Executar Backtest"):
    data = load_data(ticker, start_date, end_date)

    if data is not None and not data.empty:
        required_cols = {'Open', 'High', 'Low', 'Close'}
        if not required_cols.issubset(data.columns):
            st.error(f"Os dados carregados para '{ticker}' não contêm as colunas necessárias: Open, High, Low, Close. Colunas encontradas: {list(data.columns)}")
        else:
            st.subheader(f"Resultados para {ticker} com a estratégia '{selected_strategy_name}'")
            
            strategy_function = STRATEGIES[selected_strategy_name]
            results = strategy_function(data.copy(), **params)

            # --- VISUALIZAÇÃO DOS RESULTADOS ---
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=results.index, open=results['Open'], high=results['High'], low=results['Low'], close=results['Close'], name='Preço'))
            
            buy_signals = results[results['signal'] == 1]
            sell_signals = results[results['signal'] == -1]
            
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Sinal de Compra'))
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sinal de Venda'))
            
            fig.update_layout(title=f"Sinais de Trading para {ticker}", xaxis_title="Data", yaxis_title="Preço", xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Últimas Operações Geradas")
            last_trades = results[results['signal'] != 0].tail(10)
            st.dataframe(last_trades[['Close', 'signal', 'stop', 'target']])

else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Backtest' para começar.")
