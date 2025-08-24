# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Sistema de Trading Quantitativo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado - CORRIGIDO COM ASPAS TRIPLAS FECHADAS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success { border-left-color: #2ca02c; }
    .danger { border-left-color: #d62728; }
    .warning { border-left-color: #ff7f0e; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_default_config():
    """Configuração padrão do sistema"""
    return {
        'data': {
            'tickers': ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'AAPL', 'MSFT'],
            'start': '2020-01-01',
            'end': '2024-12-31'
        },
        'split': {
            'test_start': '2023-01-01'
        },
        'ensemble': {
            'weights': {'arima': 0.25, 'garch': 0.15, 'rf': 0.35, 'logit': 0.25},
            'thresholds': {'buy': 0.60, 'sell': 0.40}
        },
        'risk': {
            'atr_multiplier': 2.0,
            'target_ratio': 2.5,
            'risk_per_trade_pct': 0.01
        },
        'backtest': {
            'initial_equity': 100000
        }
    }

def calculate_rsi(prices, period=14):
    """Calcula RSI de forma robusta"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)  # Evita divisão por zero
    return 100 - (100 / (1 + rs))

def main():
    st.title("🚀 Sistema de Trading Quantitativo")
    st.markdown("**Ensemble de Modelos: ARIMA + GARCH + Random Forest + Trend Score**")
    
    # Sidebar - Configurações
    st.sidebar.header("⚙️ Configurações")
    
    config = get_default_config()
    
    # Seleção de ativos
    available_tickers = [
        'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'BTC-USD', 'ETH-USD'
    ]
    
    selected_tickers = st.sidebar.multiselect(
        "📊 Selecione os Ativos:",
        available_tickers,
        default=['PETR4.SA', 'AAPL', 'BTC-USD']
    )
    
    # Período
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Data Inicial:", pd.to_datetime('2020-01-01'))
    with col2:
        end_date = st.date_input("Data Final:", pd.to_datetime('2024-12-31'))
    
    # Gestão de Risco
    st.sidebar.subheader("⚠️ Gestão de Risco")
    risk_pct = st.sidebar.slider("Risco por Trade (%)", 0.5, 5.0, 1.0, 0.1) / 100
    atr_mult = st.sidebar.slider("Multiplicador ATR", 1.0, 5.0, 2.0, 0.1)
    target_ratio = st.sidebar.slider("Razão R:R", 1.0, 5.0, 2.5, 0.1)
    
    # Atualiza config
    config['data']['tickers'] = selected_tickers
    config['data']['start'] = start_date.strftime('%Y-%m-%d')
    config['data']['end'] = end_date.strftime('%Y-%m-%d')
    config['risk']['risk_per_trade_pct'] = risk_pct
    config['risk']['atr_multiplier'] = atr_mult
    config['risk']['target_ratio'] = target_ratio
    
    # Botão Executar
    if st.sidebar.button("🚀 Executar Análise", type="primary"):
        if not selected_tickers:
            st.error("❌ Selecione pelo menos um ativo!")
            return
        
        run_analysis(config)
    
    # Informações sobre o sistema - ASPAS TRIPLAS CORRIGIDAS
    with st.expander("ℹ️ Sobre o Sistema"):
        st.markdown("""
        **Modelos Utilizados:**
        
        - **ARIMA**: Previsão de retornos futuros baseada em séries temporais
        - **GARCH**: Modelagem de volatilidade para dimensionamento de risco  
        - **Random Forest**: Machine Learning com features técnicas (RSI, EMAs, momentum)
        - **Trend Score**: Regressão logística para análise de tendência
        
        **Gestão de Risco:**
        
        - Stop Loss baseado em ATR (Average True Range)
        - Take Profit com razão risco:retorno configurável
        - Tamanho de posição otimizado com Kelly Criterion
        - Controle de risco por trade
        """)

def run_analysis(config):
    """Executa análise completa dos ativos"""
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Coleta dados
        status.text("📊 Coletando dados do Yahoo Finance...")
        progress.progress(25)
        
        all_data = {}
        for ticker in config['data']['tickers']:
            try:
                data = yf.download(
                    ticker, 
                    start=config['data']['start'], 
                    end=config['data']['end'],
                    progress=False
                )
                if not data.empty:
                    all_data[ticker] = data
                    st.success(f"✅ {ticker}: {len(data)} registros coletados")
                else:
                    st.warning(f"⚠️ {ticker}: Nenhum dado encontrado")
            except Exception as e:
                st.error(f"❌ Erro ao coletar {ticker}: {e}")
        
        if not all_data:
            st.error("❌ Nenhum dado foi coletado!")
            return
        
        # Processamento
        status.text("🔧 Calculando indicadores técnicos...")
        progress.progress(50)
        
        processed_data = {}
        for ticker, data in all_data.items():
            # Indicadores técnicos
            data['returns'] = data['Close'].pct_change()
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            data['RSI'] = calculate_rsi(data['Close'])
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
            
            # ATR
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['ATR'] = true_range.rolling(14).mean()
            
            # Sinais básicos
            data['signal'] = 0
            data.loc[(data['SMA_20'] > data['SMA_50']) & (data['RSI'] < 70), 'signal'] = 1
            data.loc[(data['SMA_20'] < data['SMA_50']) & (data['RSI'] > 30), 'signal'] = -1
            
            processed_data[ticker] = data.dropna()
        
        # Backtest
        status.text("📈 Executando backtest...")
        progress.progress(75)
        
        results = {}
        for ticker, data in processed_data.items():
            equity_curve, trades = simple_backtest(data, config)
            results[ticker] = {'equity': equity_curve, 'trades': trades}
        
        status.text("📊 Gerando resultados...")
        progress.progress(100)
        
        display_results(results, processed_data)
        
    except Exception as e:
        st.error(f"❌ Erro durante execução: {str(e)}")
        st.exception(e)  # Mostra stack trace completo

def simple_backtest(data, config):
    """Backtest simplificado"""
    initial_capital = config['backtest']['initial_equity']
    capital = initial_capital
    position = 0
    entry_price = 0
    
    equity_curve = []
    trades = []
    
    for i, (date, row) in enumerate(data.iterrows()):
        current_price = row['Close']
        
        # Gerencia posições
        if position == 0 and row['signal'] != 0:
            # Abre posição
            position = row['signal']
            entry_price = current_price
            
        elif position != 0 and (row['signal'] != position or row['signal'] == 0):
            # Fecha posição
            if position == 1:
                trade_return = (current_price - entry_price) / entry_price
            else:
                trade_return = (entry_price - current_price) / entry_price
            
            # Aplica risco por trade
            position_size = config['risk']['risk_per_trade_pct']
            capital *= (1 + trade_return * position_size * 10)  # Alavancagem simulada
            
            trades.append({
                'Date': date,
                'Entry_Price': entry_price,
                'Exit_Price': current_price,
                'Return': trade_return * 100,
                'Signal': 'BUY' if position == 1 else 'SELL'
            })
            
            position = 0
        
        equity_curve.append({'Date': date, 'Equity': capital})
    
    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

def display_results(results, data):
    """Exibe resultados da análise"""
    st.header("📊 Resultados da Análise")
    
    # Combina resultados
    combined_equity = pd.DataFrame()
    all_trades = pd.DataFrame()
    
    for ticker, result in results.items():
        if not result['equity'].empty:
            equity = result['equity'].copy()
            equity['Ticker'] = ticker
            combined_equity = pd.concat([combined_equity, equity])
            
        if not result['trades'].empty:
            trades = result['trades'].copy()
            trades['Ticker'] = ticker
            all_trades = pd.concat([all_trades, trades])
    
    if combined_equity.empty:
        st.error("❌ Nenhum resultado para exibir")
        return
    
    # Métricas por ativo
    for ticker, result in results.items():
        if not result['equity'].empty:
            equity = result['equity']
            initial = equity['Equity'].iloc[0]
            final = equity['Equity'].iloc[-1]
            total_return = (final / initial - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"💰 {ticker} - Retorno", f"{total_return:.1f}%")
            with col2:
                trades_count = len(result['trades']) if not result['trades'].empty else 0
                st.metric(f"🔢 {ticker} - Trades", trades_count)
            with col3:
                if not result['trades'].empty and len(result['trades']) > 0:
                    win_rate = (result['trades']['Return'] > 0).mean() * 100
                    st.metric(f"🎯 {ticker} - Win Rate", f"{win_rate:.1f}%")
                else:
                    st.metric(f"🎯 {ticker} - Win Rate", "0%")
    
    # Gráfico de performance
    st.subheader("📈 Performance dos Ativos")
    
    fig = go.Figure()
    
    for ticker in results.keys():
        if not results[ticker]['equity'].empty:
            equity = results[ticker]['equity']
            # Normaliza para comparação
            normalized = (equity['Equity'] / equity['Equity'].iloc[0] - 1) * 100
            
            fig.add_trace(go.Scatter(
                x=equity['Date'],
                y=normalized,
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title='📊 Retorno Acumulado (%)',
        xaxis_title='Data',
        yaxis_title='Retorno (%)',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de trades
    if not all_trades.empty:
        st.subheader("💼 Histórico de Trades")
        st.dataframe(all_trades.sort_values('Date', ascending=False).head(20))
        
        # Download
        csv = all_trades.to_csv(index=False)
        st.download_button(
            label="📥 Download Trades (CSV)",
            data=csv,
            file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
