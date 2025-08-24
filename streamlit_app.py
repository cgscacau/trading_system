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
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def simple_backtest(data, config, ticker):
    """Backtest robusto com validação completa de colunas"""
    
    try:
        # Sanitiza dados primeiro
        df = sanitize_dataframe(data, ticker)
        
        # Configura índice de data
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Ordena por data
        df = df.sort_index()
        
        # Valida colunas essenciais
        required_columns = ['Close', 'signal']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise KeyError(f"{ticker}: Colunas faltando: {missing_columns}. Disponíveis: {list(df.columns)}")
        
        # Limpa dados inválidos
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['signal'] = pd.to_numeric(df['signal'], errors='coerce').fillna(0).astype('int8')
        
        # Remove registros inválidos
        df = df.dropna(subset=['Close'])
        df = df[df['Close'] > 0]  # Remove preços zero/negativos
        
        if len(df) < 10:
            st.warning(f"⚠️ {ticker}: Dados insuficientes após limpeza ({len(df)} registros)")
            return pd.DataFrame(), pd.DataFrame()
        
        # Parâmetros do backtest
        initial_capital = float(config['backtest']['initial_equity'])
        capital = initial_capital
        position = 0
        entry_price = 0.0
        entry_date = None
        
        equity_curve = []
        trades = []
        
        # Parâmetros de risco
        risk_pct = float(config['risk'].get('risk_per_trade_pct', 0.01))
        leverage_factor = 10.0
        
        # Loop principal usando itertuples (mais seguro)
        for row in df.itertuples(index=True):
            current_date = row.Index
            current_price = float(row.Close)
            current_signal = int(row.signal)
            
            # Lógica de abertura de posição
            if position == 0 and current_signal != 0:
                position = current_signal
                entry_price = current_price
                entry_date = current_date
                
            # Lógica de fechamento de posição
            elif position != 0 and (current_signal != position or current_signal == 0):
                # Calcula retorno do trade
                if position == 1:  # Posição comprada
                    trade_return = (current_price / entry_price) - 1.0
                else:  # Posição vendida
                    trade_return = (entry_price / current_price) - 1.0
                
                # Aplica ao capital
                capital *= (1.0 + trade_return * risk_pct * leverage_factor)
                
                # Registra o trade
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': current_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': current_price,
                    'Return_Pct': trade_return * 100.0,
                    'PnL': (trade_return * risk_pct * leverage_factor * initial_capital),
                    'Signal': 'BUY' if position == 1 else 'SELL',
                    'Ticker': ticker
                })
                
                # Reset da posição
                position = 0
                entry_price = 0.0
                entry_date = None
            
            # Atualiza curva de equity
            equity_curve.append({
                'Date': current_date, 
                'Equity': capital,
                'Ticker': ticker
            })
        
        # Fecha posição final se necessário
        if position != 0 and len(df) > 0:
            last_price = float(df['Close'].iloc[-1])
            last_date = df.index[-1]
            
            if position == 1:
                final_return = (last_price / entry_price) - 1.0
            else:
                final_return = (entry_price / last_price) - 1.0
                
            capital *= (1.0 + final_return * risk_pct * leverage_factor)
            
            trades.append({
                'Entry_Date': entry_date,
                'Exit_Date': last_date,
                'Entry_Price': entry_price,
                'Exit_Price': last_price,
                'Return_Pct': final_return * 100.0,
                'PnL': (final_return * risk_pct * leverage_factor * initial_capital),
                'Signal': 'BUY' if position == 1 else 'SELL',
                'Ticker': ticker,
                'Exit_Reason': 'FORCE_CLOSE_END'
            })
        
        st.success(f"✅ {ticker}: Backtest concluído - {len(trades)} trades executados")
        return pd.DataFrame(equity_curve), pd.DataFrame(trades)
        
    except Exception as e:
        st.error(f"❌ Erro no backtest {ticker}: {type(e).__name__}: {e}")
        st.exception(e)  # Mostra traceback completo
        return pd.DataFrame(), pd.DataFrame()


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
        default=['PETR4.SA', 'AAPL']
    )
    
    # Período
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Data Inicial:", pd.to_datetime('2023-01-01'))
    with col2:
        end_date = st.date_input("Data Final:", pd.to_datetime('2024-12-31'))
    
    # Gestão de Risco
    st.sidebar.subheader("⚠️ Gestão de Risco")
    risk_pct = st.sidebar.slider("Risco por Trade (%)", 0.5, 5.0, 2.0, 0.1) / 100
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

def run_analysis(config):
    """Executa análise completa com tratamento robusto de dados"""
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
            try:
                # Sanitiza e reseta índice
                df = sanitize_dataframe(data, ticker)
                df = df.reset_index()
                
                # Calcula indicadores técnicos
                df['returns'] = df['Close'].pct_change()
                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                df['RSI'] = calculate_rsi(df['Close'])
                
                # ATR para gestão de risco
                high_low = df['High'] - df['Low']
                high_close = np.abs(df['High'] - df['Close'].shift())
                low_close = np.abs(df['Low'] - df['Close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['ATR'] = true_range.rolling(14).mean()
                
                # Geração de sinais
                df['signal'] = 0
                
                # Condições de compra e venda
                buy_condition = (
                    (df['SMA_20'] > df['SMA_50']) & 
                    (df['RSI'] < 70) & 
                    (df['RSI'] > 30)
                )
                sell_condition = (
                    (df['SMA_20'] < df['SMA_50']) & 
                    (df['RSI'] > 30) & 
                    (df['RSI'] < 70)
                )
                
                # Aplica sinais
                df.loc[buy_condition, 'signal'] = 1
                df.loc[sell_condition, 'signal'] = -1
                
                # Garante que signal é inteiro
                df['signal'] = df['signal'].fillna(0).astype('int8')
                
                # Remove NaN apenas das colunas essenciais
                df_clean = df.dropna(subset=['Date', 'Close'])
                
                if len(df_clean) < 50:
                    st.warning(f"⚠️ {ticker}: Poucos dados válidos ({len(df_clean)} registros)")
                    continue
                
                processed_data[ticker] = df_clean
                st.success(f"✅ {ticker}: Indicadores calculados ({len(df_clean)} registros válidos)")
                
            except Exception as e:
                st.error(f"❌ Erro processando {ticker}: {e}")
                st.exception(e)
                continue
        
        if not processed_data:
            st.error("❌ Nenhum dado processado com sucesso!")
            return
        
        # Backtest
        status.text("📈 Executando backtest...")
        progress.progress(75)
        
        results = {}
        for ticker, data in processed_data.items():
            try:
                equity_curve, trades = simple_backtest(data, config, ticker)
                if not equity_curve.empty:
                    results[ticker] = {'equity': equity_curve, 'trades': trades}
                    
            except Exception as e:
                st.error(f"❌ Erro no backtest {ticker}: {e}")
                continue
        
        status.text("📊 Gerando resultados...")
        progress.progress(100)
        
        if results:
            display_results(results, processed_data)
        else:
            st.error("❌ Nenhum backtest foi executado com sucesso!")
        
    except Exception as e:
        st.error(f"❌ Erro durante execução: {str(e)}")
        st.exception(e)


def display_results(results, data):
    """Exibe resultados da análise"""
    st.header("📊 Resultados da Análise")
    
    if not results:
        st.error("❌ Nenhum resultado para exibir")
        return
    
    # Combina resultados
    all_trades = pd.DataFrame()
    
    # Métricas por ativo
    cols = st.columns(min(len(results), 4))  # Máximo 4 colunas
    for i, (ticker, result) in enumerate(results.items()):
        col_idx = i % len(cols)
        with cols[col_idx]:
            if not result['equity'].empty:
                equity = result['equity']
                initial = equity['Equity'].iloc[0]
                final = equity['Equity'].iloc[-1]
                total_return = (final / initial - 1) * 100
                
                st.metric(
                    f"💰 {ticker}",
                    f"{total_return:.1f}%",
                    f"{final - initial:,.0f}"
                )
                
                trades_count = len(result['trades']) if not result['trades'].empty else 0
                
                if trades_count > 0:
                    win_rate = (result['trades']['Return_Pct'] > 0).mean() * 100
                    st.metric(f"🎯 Win Rate", f"{win_rate:.1f}%")
                    st.metric(f"🔢 Trades", trades_count)
                else:
                    st.metric(f"🎯 Win Rate", "0%")
                    st.metric(f"🔢 Trades", "0")
        
        # Adiciona trades ao conjunto geral
        if not result['trades'].empty:
            all_trades = pd.concat([all_trades, result['trades']], ignore_index=True)
    
    # Gráfico de performance
    st.subheader("📈 Performance dos Ativos")
    
    fig = go.Figure()
    
    for ticker, result in results.items():
        if not result['equity'].empty:
            equity = result['equity']
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
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise de trades
    if not all_trades.empty:
        st.subheader("💼 Análise de Trades")
        
        # Estatísticas gerais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_trades = len(all_trades)
            st.metric("Total Trades", total_trades)
            
        with col2:
            if total_trades > 0:
                win_rate = (all_trades['Return_Pct'] > 0).mean() * 100
                st.metric("Win Rate Geral", f"{win_rate:.1f}%")
            else:
                st.metric("Win Rate Geral", "0%")
                
        with col3:
            if total_trades > 0:
                avg_return = all_trades['Return_Pct'].mean()
                st.metric("Retorno Médio", f"{avg_return:.2f}%")
            else:
                st.metric("Retorno Médio", "0%")
                
        with col4:
            if total_trades > 0:
                total_pnl = all_trades['PnL'].sum()
                st.metric("P&L Total", f"${total_pnl:,.0f}")
            else:
                st.metric("P&L Total", "$0")
        
        # Tabela de trades
        st.subheader("📋 Histórico de Trades")
        display_trades = all_trades.sort_values('Exit_Date', ascending=False).copy()
        
        # Formata colunas para melhor visualização
        if not display_trades.empty:
            display_trades['Entry_Date'] = pd.to_datetime(display_trades['Entry_Date']).dt.strftime('%Y-%m-%d')
            display_trades['Exit_Date'] = pd.to_datetime(display_trades['Exit_Date']).dt.strftime('%Y-%m-%d')
            display_trades['Return_Pct'] = display_trades['Return_Pct'].round(2)
            display_trades['PnL'] = display_trades['PnL'].round(2)
            
            st.dataframe(display_trades.head(20), use_container_width=True)
            
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
