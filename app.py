"""
Lab de Estratégias & Sizing BRL→USD - Aplicação Streamlit
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Imports dos módulos
from data.loader import load_price_data, get_usd_brl_rate, get_asset_info
from core.backtest import BacktestEngine
from core.metrics import calculate_performance_metrics, rank_strategies, create_metrics_table
from utils.risk import calculate_position_size, get_risk_checklist, validate_risk_parameters
from viz.plots import (plot_candlestick_with_trades, plot_equity_curve, 
                      plot_strategy_comparison, plot_ml_feature_importance, 
                      plot_roc_curve, plot_monthly_returns)
from ml.modeling import MLStrategy

# Importar estratégias
from core.strategies.moving_averages import (ema_crossover_strategy, sma_crossover_strategy,
                                           EMA_CROSSOVER_PARAMS, SMA_CROSSOVER_PARAMS)
from core.strategies.rsi_strategies import (rsi_ifr2_strategy, rsi_standard_strategy,
                                          RSI_IFR2_PARAMS, RSI_STANDARD_PARAMS)
from core.strategies.macd_strategy import (macd_strategy, MACD_PARAMS)
from core.strategies.bollinger_bands import (bollinger_breakout_strategy, bollinger_mean_reversion_strategy,
                                           BOLLINGER_BREAKOUT_PARAMS, BOLLINGER_MEAN_REVERSION_PARAMS)
from core.strategies.donchian_turtle import (donchian_breakout_strategy, DONCHIAN_BREAKOUT_PARAMS)
from core.strategies.momentum_roc import (momentum_roc_strategy, MOMENTUM_ROC_PARAMS)
from core.strategies.breakout_strategies import (high_low_breakout_strategy, HIGH_LOW_BREAKOUT_PARAMS)
from core.strategies.adx_dmi import (adx_dmi_strategy, ADX_DMI_PARAMS)
from core.strategies.candle_patterns import (candle_patterns_strategy, CANDLE_PATTERNS_PARAMS)
from core.strategies.invented_strategies import (vol_regime_switch_strategy, meta_ensemble_strategy, pullback_trend_bias_strategy,
                                               VOL_REGIME_SWITCH_PARAMS, META_ENSEMBLE_PARAMS, PULLBACK_TREND_BIAS_PARAMS)

# Atualizar lista de estratégias na sidebar
strategies_selected = st.multiselect(
    "Selecione as estratégias:",
    options=[
        "EMA Crossover",
        "SMA Crossover", 
        "RSI IFR2",
        "RSI Padrão",
        "MACD",
        "Bollinger Breakout",
        "Bollinger Mean Reversion",
        "Donchian Breakout",
        "Momentum/ROC",
        "High/Low Breakout",
        "ADX + DMI",
        "Padrões de Velas",
        "Vol-Regime Switch",
        "Meta-Ensemble",
        "Pullback Trend-Bias",
        "Machine Learning"
    ],
    default=["EMA Crossover", "RSI IFR2", "Meta-Ensemble"]
)



# Configuração da página
st.set_page_config(
    page_title="Lab de Estratégias & Sizing BRL→USD",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
with open('assets/theme.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Título principal
st.markdown("""
# 📈 Lab de Estratégias & Sizing BRL→USD
### Testagem e comparação de estratégias de trading com gestão de risco integrada
""")

# Sidebar - Configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Dados do ativo
    with st.expander("📊 Dados do Ativo", expanded=True):
        ticker = st.text_input("Ticker", value="PETR4.SA", help="Ex: PETR4.SA, AAPL, BTC-USD").upper()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Data Início", value=datetime.now() - timedelta(days=2*365))
        with col2:
            end_date = st.date_input("Data Fim", value=datetime.now())
        
        interval = st.selectbox("Timeframe", 
                               options=['1d', '1h', '30m', '15m', '5m'],
                               index=0)
    
    # Parâmetros de backtest
    with st.expander("🔧 Parâmetros de Backtest"):
        fee_pct = st.slider("Taxa por lado (%)", 0.0, 1.0, 0.1, 0.01) / 100
        slippage_pct = st.slider("Slippage (%)", 0.0, 0.5, 0.05, 0.01) / 100
        max_bars = st.number_input("Máx barras em posição (0=ilimitado)", 0, 1000, 0)
        max_bars = None if max_bars == 0 else max_bars
    
    # Gestão de risco
    with st.expander("💰 Gestão de Risco"):
        capital_brl = st.number_input("Capital (BRL)", 1000.0, 10000000.0, 100000.0, 1000.0)
        risk_pct = st.slider("Risco por trade (%)", 0.1, 5.0, 1.0, 0.1) / 100
        
        # Câmbio
        auto_fx = st.checkbox("Câmbio automático", value=True)
        if auto_fx:
            fx_rate, fx_time = get_usd_brl_rate()
            if fx_rate:
                st.success(f"USDBRL: {fx_rate:.4f}")
                st.caption(f"Atualizado: {fx_time.strftime('%H:%M:%S')}")
            else:
                st.error("Erro ao obter câmbio")
                fx_rate = st.number_input("USDBRL manual", 1.0, 10.0, 5.0, 0.01)
        else:
            fx_rate = st.number_input("USDBRL manual", 1.0, 10.0, 5.0, 0.01)
    
    # Seleção de estratégias
    with st.expander("🎯 Estratégias", expanded=True):
        strategies_selected = st.multiselect(
            "Selecione as estratégias:",
            options=[
                "EMA Crossover",
                "SMA Crossover", 
                "RSI IFR2",
                "RSI Padrão",
                "Machine Learning"
            ],
            default=["EMA Crossover", "RSI IFR2"]
        )
    
    # Critério de ranking
    ranking_criterion = st.selectbox(
        "🏆 Critério de Ranking",
        options=['Calmar Ratio', 'Sharpe Ratio', 'CAGR', 'Total Return', 
                'Profit Factor', 'Hit Rate'],
        index=0
    )
    
    # Botão de execução
    run_analysis = st.button("🚀 Executar Análise", type="primary", use_container_width=True)

# Função para executar estratégias
def run_strategy(strategy_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Executa uma estratégia específica"""
    
    try:
        if strategy_name == "EMA Crossover":
            return ema_crossover_strategy(df, EMA_CROSSOVER_PARAMS)
        elif strategy_name == "SMA Crossover":
            return sma_crossover_strategy(df, SMA_CROSSOVER_PARAMS)
        elif strategy_name == "RSI IFR2":
            return rsi_ifr2_strategy(df, RSI_IFR2_PARAMS)
        elif strategy_name == "RSI Padrão":
            return rsi_standard_strategy(df, RSI_STANDARD_PARAMS)
        elif strategy_name == "MACD":
            return macd_strategy(df, MACD_PARAMS)
        elif strategy_name == "Bollinger Breakout":
            return bollinger_breakout_strategy(df, BOLLINGER_BREAKOUT_PARAMS)
        elif strategy_name == "Bollinger Mean Reversion":
            return bollinger_mean_reversion_strategy(df, BOLLINGER_MEAN_REVERSION_PARAMS)
        elif strategy_name == "Donchian Breakout":
            return donchian_breakout_strategy(df, DONCHIAN_BREAKOUT_PARAMS)
        elif strategy_name == "Momentum/ROC":
            return momentum_roc_strategy(df, MOMENTUM_ROC_PARAMS)
        elif strategy_name == "High/Low Breakout":
            return high_low_breakout_strategy(df, HIGH_LOW_BREAKOUT_PARAMS)
        elif strategy_name == "ADX + DMI":
            return adx_dmi_strategy(df, ADX_DMI_PARAMS)
        elif strategy_name == "Padrões de Velas":
            return candle_patterns_strategy(df, CANDLE_PATTERNS_PARAMS)
        elif strategy_name == "Vol-Regime Switch":
            return vol_regime_switch_strategy(df, VOL_REGIME_SWITCH_PARAMS)
        elif strategy_name == "Meta-Ensemble":
            return meta_ensemble_strategy(df, META_ENSEMBLE_PARAMS)
        elif strategy_name == "Pullback Trend-Bias":
            return pullback_trend_bias_strategy(df, PULLBACK_TREND_BIAS_PARAMS)
        elif strategy_name == "Machine Learning":
            ml_strategy = MLStrategy()
            try:
                ml_strategy.train_model(df, model_type='RandomForest')
                st.session_state['ml_strategy'] = ml_strategy
                return ml_strategy.generate_signals(df)
            except Exception as e:
                st.error(f"Erro no ML: {e}")
                return pd.DataFrame(columns=['signal', 'stop', 'target'])
        else:
            return pd.DataFrame(columns=['signal', 'stop', 'target'])
    except Exception as e:
        st.error(f"Erro em {strategy_name}: {e}")
        return pd.DataFrame(columns=['signal', 'stop', 'target'])
# Execução principal
if run_analysis:
    if not ticker:
        st.error("Por favor, insira um ticker válido")
        st.stop()
    
    if not strategies_selected:
        st.error("Selecione pelo menos uma estratégia")
        st.stop()
    
    # Validar parâmetros de risco
    risk_errors = validate_risk_parameters(capital_brl, risk_pct, fx_rate, 100)
    if risk_errors:
        for field, error in risk_errors.items():
            st.error(f"{field}: {error}")
        st.stop()
    
    # Carregar dados
    with st.spinner(f"Carregando dados de {ticker}..."):
        df = load_price_data(ticker, start_date.strftime('%Y-%m-%d'), 
                           end_date.strftime('%Y-%m-%d'), interval)
    
    if df.empty:
        st.error("Não foi possível carregar os dados. Verifique o ticker e período.")
        st.stop()
    
    # Informações do ativo
    asset_info = get_asset_info(ticker)
    
    # Executar backtests
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, strategy in enumerate(strategies_selected):
        status_text.text(f"Executando {strategy}...")
        
        try:
            # Gerar sinais
            signals = run_strategy(strategy, df)
            
            if signals.empty or signals['signal'].sum() == 0:
                st.warning(f"Nenhum sinal gerado para {strategy}")
                continue
            
            # Executar backtest
            engine = BacktestEngine(capital_brl, fee_pct, slippage_pct)
            backtest_result = engine.run_backtest(df, signals, max_bars)
            
            # Calcular métricas
            metrics = calculate_performance_metrics(
                backtest_result['trades'],
                backtest_result['equity_curve'],
                capital_brl
            )
            
            results[strategy] = {
                'metrics': metrics,
                'trades': backtest_result['trades'],
                'signals': signals
            }
            
        except Exception as e:
            st.error(f"Erro em {strategy}: {e}")
        
        progress_bar.progress((i + 1) / len(strategies_selected))
    
    status_text.text("Análise concluída!")
    
    if not results:
        st.error("Nenhuma estratégia foi executada com sucesso")
        st.stop()
    
    # Salvar resultados no session state
    st.session_state['results'] = results
    st.session_state['df'] = df
    st.session_state['asset_info'] = asset_info
    st.session_state['fx_rate'] = fx_rate

# Exibir resultados se disponíveis
if 'results' in st.session_state and st.session_state['results']:
    results = st.session_state['results']
    df = st.session_state['df']
    asset_info = st.session_state['asset_info']
    fx_rate = st.session_state['fx_rate']
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral", 
        "📈 Backtests", 
        "📉 Gráfico", 
        "🤖 Machine Learning", 
        "💼 Sizing & Risco"
    ])
    
    with tab1:
        st.header("📊 Visão Geral")
        
        # Ranking das estratégias
        metrics_only = {name: result['metrics'] for name, result in results.items()}
        rankings = rank_strategies(metrics_only, ranking_criterion)
        
        if rankings:
            st.subheader("🏆 Ranking das Estratégias")
            
            # Cards do top 3
            cols = st.columns(3)
            medals = ["🥇", "🥈", "🥉"]
            colors = ["best-strategy", "second-strategy", "third-strategy"]
            
            for i, (strategy_name, score) in enumerate(rankings[:3]):
                with cols[i]:
                    metrics = results[strategy_name]['metrics']
                    
                    st.markdown(f"""
                    <div class="strategy-card {colors[i]}">
                        <h3>{medals[i]} {strategy_name}</h3>
                        <p><strong>{ranking_criterion}:</strong> {score:.3f}</p>
                        <p><strong>Retorno Total:</strong> {metrics['Total Return']:.2%}</p>
                        <p><strong>Trades:</strong> {metrics['Num Trades']}</p>
                        <p><strong>Hit Rate:</strong> {metrics['Hit Rate']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Métricas principais da melhor estratégia
            best_strategy = rankings[0][0]
            best_metrics = results[best_strategy]['metrics']
            
            st.subheader(f"📈 Métricas da Melhor Estratégia: {best_strategy}")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("CAGR", f"{best_metrics['CAGR']:.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{best_metrics['Sharpe Ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{best_metrics['Max Drawdown']:.2%}")
            with col4:
                st.metric("Calmar Ratio", f"{best_metrics['Calmar Ratio']:.2f}")
            with col5:
                st.metric("Hit Rate", f"{best_metrics['Hit Rate']:.2%}")
            with col6:
                st.metric("Profit Factor", f"{best_metrics['Profit Factor']:.2f}")
            
            # Sinal atual
            st.subheader("🎯 Sinal Atual da Melhor Estratégia")
            
            current_signals = results[best_strategy]['signals']
            if not current_signals.empty:
                last_signal = current_signals.iloc[-1]
                current_price = df['Close'].iloc[-1]
                
                signal_text = "FLAT"
                signal_color = "gray"
                
                if last_signal['signal'] == 1:
                    signal_text = "COMPRA"
                    signal_color = "green"
                elif last_signal['signal'] == -1:
                    signal_text = "VENDA"
                    signal_color = "red"
                
                st.markdown(f"""
                <div class="signal-{signal_text.lower()}">
                    <h3>Sinal: {signal_text}</h3>
                    <p>Preço Atual: {current_price:.2f}</p>
                    <p>Stop Sugerido: {last_signal.get('stop', 'N/A')}</p>
                    <p>Target Sugerido: {last_signal.get('target', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Comparação visual
            st.subheader("📊 Comparação de Estratégias")
            fig_comparison = plot_strategy_comparison(metrics_only)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab2:
        st.header("📈 Resultados dos Backtests")
        
        # Tabela de métricas
        metrics_table = create_metrics_table(metrics_only)
        st.dataframe(metrics_table, use_container_width=True)
        
        # Seleção de estratégia para detalhes
        selected_strategy = st.selectbox(
            "Selecione uma estratégia para detalhes:",
            options=list(results.keys())
        )
        
        if selected_strategy:
            strategy_metrics = results[selected_strategy]['metrics']
            
            # Curva de capital
            st.subheader(f"💹 Curva de Capital - {selected_strategy}")
            fig_equity = plot_equity_curve(
                strategy_metrics['Equity Curve'],
                strategy_metrics['Drawdown Series'],
                f"Evolução do Capital - {selected_strategy}"
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Retornos mensais
            if len(strategy_metrics['Equity Curve']) > 30:
                st.subheader("📅 Retornos Mensais")
                fig_monthly = plot_monthly_returns(strategy_metrics['Equity Curve'])
                st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab3:
        st.header("📉 Gráfico de Trades")
        
        # Seleção de estratégia
        chart_strategy = st.selectbox(
            "Estratégia para visualizar:",
            options=list(results.keys()),
            key="chart_strategy"
        )
        
        if chart_strategy:
            trades = results[chart_strategy]['trades']
            
            # Gráfico de candlestick com trades
            fig_trades = plot_candlestick_with_trades(
                df, trades, f"{ticker} - {chart_strategy}"
            )
            st.plotly_chart(fig_trades, use_container_width=True)
            
            # Estatísticas dos trades
            if trades:
                st.subheader("📊 Estatísticas dos Trades")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total de Trades", len(trades))
                with col2:
                    winning_trades = [t for t in trades if t.net_pnl > 0]
                    st.metric("Trades Vencedores", len(winning_trades))
                with col3:
                    avg_pnl = np.mean([t.net_pnl for t in trades])
                    st.metric("PnL Médio", f"{avg_pnl:.2f}")
                with col4:
                    avg_bars = np.mean([t.bars_held for t in trades])
                    st.metric("Barras Médias", f"{avg_bars:.1f}")
    
    with tab4:
        st.header("🤖 Machine Learning")
        
        if "Machine Learning" in results:
            ml_results = results["Machine Learning"]
            
            # Verificar se temos dados ML específicos
            if hasattr(st.session_state.get('ml_strategy', None), 'validation_results'):
                ml_data = st.session_state.ml_strategy.validation_results
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("AUC Score", f"{ml_data.get('auc_score', 0):.3f}")
                    st.metric("Accuracy", f"{ml_data.get('accuracy', 0):.3f}")
                
                with col2:
                    # Feature importance
                    if 'feature_importance' in ml_data:
                        fig_features = plot_ml_feature_importance(ml_data['feature_importance'])
                        st.plotly_chart(fig_features, use_container_width=True)
                
                # Matriz de confusão
                if 'confusion_matrix' in ml_data:
                    st.subheader("📊 Matriz de Confusão")
                    cm = ml_data['confusion_matrix']
                    cm_df = pd.DataFrame(cm, 
                                       index=['Real Negativo', 'Real Positivo'],
                                       columns=['Pred Negativo', 'Pred Positivo'])
                    st.dataframe(cm_df)
        else:
            st.info("Machine Learning não foi selecionado ou falhou na execução.")
    
    with tab5:
        st.header("💼 Sizing & Gestão de Risco")
        
        # Informações do ativo
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ℹ️ Informações do Ativo")
            st.write(f"**Nome:** {asset_info['name']}")
            st.write(f"**Setor:** {asset_info['sector']}")
            st.write(f"**Moeda:** {asset_info['currency']}")
            st.write(f"**Preço Atual:** {df['Close'].iloc[-1]:.2f}")
        
        with col2:
            st.subheader("💱 Parâmetros de Risco")
            st.write(f"**Capital:** R$ {capital_brl:,.2f}")
            st.write(f"**Risco por Trade:** {risk_pct:.1%}")
            st.write(f"**Taxa USDBRL:** {fx_rate:.4f}")
        
        # Cálculo de posição
        st.subheader("📊 Cálculo de Posição")
        
        # Usar a melhor estratégia para calcular stop
        if rankings:
            best_strategy = rankings[0][0]
            best_signals = results[best_strategy]['signals']
            
            if not best_signals.empty:
                last_signal = best_signals.iloc[-1]
                current_price = df['Close'].iloc[-1]
                
                # Stop distance
                if not pd.isna(last_signal.get('stop', np.nan)):
                    stop_distance = abs(current_price - last_signal['stop'])
                else:
                    # Usar ATR como fallback
                    from utils.risk import calculate_atr
                    atr = calculate_atr(df).iloc[-1]
                    stop_distance = atr * 2
                
                # Calcular posição
                position_calc = calculate_position_size(
                    capital_brl=capital_brl,
                    risk_percent=risk_pct,
                    fx_rate=fx_rate,
                    asset_price=current_price,
                    stop_distance=stop_distance,
                    fee_percent=fee_pct,
                    slippage_percent=slippage_pct,
                    asset_currency=asset_info['currency']
                )
                
                # Exibir resultados
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Quantidade", f"{position_calc.position_size:.2f}")
                    st.metric("Risco (BRL)", f"R$ {position_calc.risk_brl:.2f}")
                
                with col2:
                    st.metric("Exposição (USD)", f"$ {position_calc.exposure_usd:.2f}")
                    st.metric("Exposição (BRL)", f"R$ {position_calc.exposure_brl:.2f}")
                
                with col3:
                    st.metric("R:R Ratio", f"{position_calc.r_ratio:.2f}")
                    st.metric("Alavancagem", f"{position_calc.leverage:.2f}x")
                
                # Checklist de risco
                st.subheader("✅ Checklist de Risco")
                checklist = get_risk_checklist(df, current_price, stop_distance)
                
                for item, status in checklist.items():
                    st.write(f"**{item}:** {status}")

else:
    # Tela inicial
    st.markdown("""
    ## 🚀 Bem-vindo ao Lab de Estratégias!
    
    Este aplicativo permite testar e comparar múltiplas estratégias de trading com:
    
    - ✅ **Estratégias Clássicas**: EMA/SMA Crossover, RSI, MACD, Bollinger Bands, etc.
    - 🤖 **Machine Learning**: Modelos preditivos com validação temporal
    - 💰 **Gestão de Risco**: Sizing automático BRL→USD
    - 📊 **Backtests Robustos**: Engine sem look-ahead com custos realistas
    - 📈 **Visualizações Interativas**: Gráficos e métricas detalhadas
    
    ### Como usar:
    1. Configure os parâmetros na barra lateral
    2. Selecione as estratégias desejadas
    3. Clique em "🚀 Executar Análise"
    4. Explore os resultados nas abas
    
    **Configure os parâmetros na barra lateral e clique em "Executar Análise" para começar!**
    """)
    
    # Mostrar exemplo de dados
    st.subheader("📊 Exemplo de Dados")
    sample_data = pd.DataFrame({
        'Data': pd.date_range('2024-01-01', periods=5),
        'Open': [100, 102, 101, 103, 105],
        'High': [103, 104, 103, 106, 107],
        'Low': [99, 101, 100, 102, 104],
        'Close': [102, 101, 103, 105, 106],
        'Volume': [1000, 1200, 800, 1500, 1100]
    })
    st.dataframe(sample_data, use_container_width=True)

