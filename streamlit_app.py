# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yaml
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Imports do sistema (assumindo que os módulos estão na pasta src/)
try:
    from src.data.data_loader import DataLoader
    from src.data.features import FeatureEngine
    from src.models.arima_model import ARIMAModel
    from src.models.garch_model import GARCHModel
    from src.models.rf_model import RandomForestModel
    from src.models.trend_model import TrendScoreModel
    from src.models.ensemble import EnsemblePredictor
    from src.risk.risk_manager import RiskManager
    from src.backtest.backtester import Backtester
    from src.backtest.metrics import calculate_performance_metrics
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

# Configuração da página
st.set_page_config(
    page_title="Sistema de Trading Quantitativo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
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
        'models': {
            'arima': {'order': [2, 1, 2]},
            'garch': {'p': 1, 'q': 1, 'dist': 'normal'},
            'rf': {'n_estimators': 200, 'max_depth': 8, 'random_state': 42},
            'logit': {'C': 1.0, 'penalty': 'l2', 'max_iter': 300}
        },
        'ensemble': {
            'weights': {'arima': 0.25, 'garch': 0.15, 'rf': 0.35, 'logit': 0.25},
            'thresholds': {'buy': 0.60, 'sell': 0.40}
        },
        'risk': {
            'atr_window': 14,
            'atr_multiplier': 2.0,
            'target_ratio': 2.5,
            'risk_per_trade_pct': 0.01,
            'use_kelly': True,
            'kelly_fraction': 0.25,
            'kelly_cap': 0.05
        },
        'backtest': {
            'initial_equity': 100000,
            'slippage_bps': 3,
            'commission_bps': 2
        }
    }

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
    
    # Pesos do Ensemble
    st.sidebar.subheader("🤖 Pesos do Ensemble")
    arima_w = st.sidebar.slider("ARIMA", 0.0, 1.0, 0.25, 0.05)
    garch_w = st.sidebar.slider("GARCH", 0.0, 1.0, 0.15, 0.05)
    rf_w = st.sidebar.slider("Random Forest", 0.0, 1.0, 0.35, 0.05)
    logit_w = st.sidebar.slider("Trend Score", 0.0, 1.0, 0.25, 0.05)
    
    # Normaliza pesos
    total_w = arima_w + garch_w + rf_w + logit_w
    if total_w > 0:
        config['ensemble']['weights'] = {
            'arima': arima_w / total_w,
            'garch': garch_w / total_w,
            'rf': rf_w / total_w,
            'logit': logit_w / total_w
        }
    
    # Thresholds
    st.sidebar.subheader("🎯 Thresholds")
    buy_thresh = st.sidebar.slider("BUY Threshold", 0.5, 0.9, 0.60, 0.01)
    sell_thresh = st.sidebar.slider("SELL Threshold", 0.1, 0.5, 0.40, 0.01)
    
    config['ensemble']['thresholds'] = {'buy': buy_thresh, 'sell': sell_thresh}
    
    # Gestão de Risco
    st.sidebar.subheader("⚠️ Gestão de Risco")
    risk_pct = st.sidebar.slider("Risco por Trade (%)", 0.5, 5.0, 1.0, 0.1) / 100
    atr_mult = st.sidebar.slider("Multiplicador ATR", 1.0, 5.0, 2.0, 0.1)
    target_ratio = st.sidebar.slider("Razão R:R", 1.0, 5.0, 2.5, 0.1)
    use_kelly = st.sidebar.checkbox("Usar Kelly Criterion", True)
    
    config['risk'].update({
        'risk_per_trade_pct': risk_pct,
        'atr_multiplier': atr_mult,
        'target_ratio': target_ratio,
        'use_kelly': use_kelly
    })
    
    # Atualiza config
    config['data']['tickers'] = selected_tickers
    config['data']['start'] = start_date.strftime('%Y-%m-%d')
    config['data']['end'] = end_date.strftime('%Y-%m-%d')
    
    # Botão Executar
    if st.sidebar.button("🚀 Executar Backtest", type="primary"):
        if not selected_tickers:
            st.error("❌ Selecione pelo menos um ativo!")
            return
        
        with st.spinner("🔄 Executando sistema de trading..."):
            results = run_trading_system(config)
        
        if results:
            display_results(results)
    
    # Informações sobre o sistema
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

@st.cache_data
def run_trading_system(config):
    """Executa o sistema de trading"""
    try:
        progress = st.progress(0)
        status = st.empty()
        
        # Coleta dados
        status.text("📊 Coletando dados do Yahoo Finance...")
        progress.progress(20)
        
        data_loader = DataLoader(config)
        raw_data = data_loader.download_data(
            config['data']['tickers'],
            config['data']['start'],
            config['data']['end']
        )
        
        if not raw_data:
            st.error("❌ Erro ao coletar dados!")
            return None
        
        # Processa features
        status.text("🔧 Calculando indicadores técnicos...")
        progress.progress(40)
        
        feature_engine = FeatureEngine(config)
        processed_data = {}
        for symbol, data in raw_data.items():
            processed_data[symbol] = feature_engine.calculate_technical_indicators(data)
        
        # Inicializa modelos
        status.text("🤖 Inicializando modelos de IA...")
        progress.progress(60)
        
        models = {
            'arima': ARIMAModel(config),
            'garch': GARCHModel(config),
            'rf': RandomForestModel(config),
            'trend': TrendScoreModel(config),
            'ensemble': EnsemblePredictor(config)
        }
        
        # Executa backtest
        status.text("📈 Executando backtest...")
        progress.progress(80)
        
        risk_manager = RiskManager(config['risk'])
        backtester = Backtester(config, models, risk_manager)
        results = backtester.run_backtest(processed_data)
        
        # Calcula métricas
        status.text("📊 Calculando métricas de performance...")
        progress.progress(100)
        
        metrics = calculate_performance_metrics(
            results['combined_equity'],
            results['all_trades']
        )
        
        status.text("✅ Concluído!")
        
        return {'results': results, 'metrics': metrics}
        
    except Exception as e:
        st.error(f"❌ Erro durante execução: {str(e)}")
        return None

def display_results(system_results):
    """Exibe resultados do backtest"""
    results = system_results['results']
    metrics = system_results['metrics']
    
    st.header("📊 Resultados do Backtest")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Retorno Total", metrics['Total Return'])
    with col2:
        st.metric("📈 CAGR", metrics['CAGR'])
    with col3:
        st.metric("⚡ Sharpe Ratio", metrics['Sharpe Ratio'])
    with col4:
        st.metric("📉 Max Drawdown", metrics['Max Drawdown'])
    
    # Métricas secundárias
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Volatilidade", metrics['Volatility'])
    with col2:
        st.metric("🎯 Win Rate", metrics['Win Rate'])
    with col3:
        st.metric("💎 Profit Factor", metrics['Profit Factor'])
    with col4:
        st.metric("🔢 Total Trades", metrics['Total Trades'])
    
    # Gráficos
    equity_data = results['combined_equity']
    
    # Curva de Equity
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_data['Date'],
        y=equity_data['Equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig_equity.update_layout(
        title='📈 Curva de Equity',
        xaxis_title='Data',
        yaxis_title='Valor da Carteira ($)',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Drawdown
    running_max = equity_data['Equity'].cummax()
    drawdown = (equity_data['Equity'] - running_max) / running_max * 100
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=equity_data['Date'],
        y=drawdown,
        mode='lines',
        fill='tonexty',
        name='Drawdown',
        line=dict(color='#d62728'),
        fillcolor='rgba(214, 39, 40, 0.3)'
    ))
    
    fig_dd.update_layout(
        title='📉 Drawdown',
        xaxis_title='Data',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Tabela de Trades
    if not results['all_trades'].empty:
        st.subheader("💼 Histórico de Trades")
        trades_df = results['all_trades']
        
        # Mostra primeiros 20 trades
        st.dataframe(trades_df.head(20), use_container_width=True)
        
        # Download
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trades (CSV)",
            data=csv,
            file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
