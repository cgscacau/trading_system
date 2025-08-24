"""
Lab de Estratégias & Sizing BRL→USD - Aplicação Streamlit
Versão completa com todas as estratégias implementadas e tratamento robusto de erros
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Imports dos módulos principais
from data.loader import load_price_data, get_usd_brl_rate, get_asset_info
from core.backtest import BacktestEngine
from core.metrics import calculate_performance_metrics, rank_strategies, create_metrics_table
from utils.risk import calculate_position_size, get_risk_checklist, validate_risk_parameters
from viz.plots import (plot_candlestick_with_trades, plot_equity_curve, 
                      plot_strategy_comparison, plot_ml_feature_importance, 
                      plot_roc_curve, plot_monthly_returns)
from ml.modeling import MLStrategy

# Imports de todas as estratégias clássicas
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

# Imports das estratégias inventadas
from core.strategies.invented_strategies import (vol_regime_switch_strategy, meta_ensemble_strategy, pullback_trend_bias_strategy,
                                               VOL_REGIME_SWITCH_PARAMS, META_ENSEMBLE_PARAMS, PULLBACK_TREND_BIAS_PARAMS)

# Configuração da página
st.set_page_config(
    page_title="Lab de Estratégias & Sizing BRL→USD",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado com fallback inline
try:
    with open('assets/theme.css', 'r', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except:
    # CSS inline como fallback para compatibilidade
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1f77b4, #4a90e2);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .strategy-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .best-strategy {
        border-left-color: #2ca02c;
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
    }
    
    .second-strategy {
        border-left-color: #ff7f0e;
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    }
    
    .third-strategy {
        border-left-color: #17a2b8;
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #2ca02c, #28a745);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #d62728, #dc3545);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .signal-flat {
        background: linear-gradient(135deg, #6c757d, #adb5bd);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4, #4a90e2);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    </style>
    """, unsafe_allow_html=True)

# Header principal
st.markdown("""
# 📈 Lab de Estratégias & Sizing BRL→USD
### Teste, compare e dimensione estratégias de trading com gestão de risco realista
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
        
        # Câmbio automático/manual
        auto_fx = st.checkbox("Câmbio automático", value=True)
        if auto_fx:
            fx_rate, fx_time = get_usd_brl_rate()
            if fx_rate:
                st.success(f"USDBRL: {fx_rate:.4f}")
                if fx_time:
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
    
    # Configurações específicas do ML
    ml_config = {}
    if "Machine Learning" in strategies_selected:
        with st.expander("🤖 Configuração ML"):
            ml_model = st.selectbox("Modelo", ["RandomForest", "GradientBoosting", "LogisticRegression"])
            ml_horizon = st.slider("Horizonte (dias)", 2, 30, 5)
            ml_threshold = st.slider("Threshold alvo (%)", 0.1, 5.0, 1.0) / 100
            ml_buy_th = st.slider("Threshold compra", 0.5, 0.9, 0.6, 0.01)
            ml_sell_th = st.slider("Threshold venda", 0.1, 0.5, 0.4, 0.01)
            
            ml_config = {
                'model_type': ml_model,
                'horizon': ml_horizon,
                'threshold': ml_threshold,
                'buy_threshold': ml_buy_th,
                'sell_threshold': ml_sell_th
            }
    
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
def run_strategy(strategy_name: str, df: pd.DataFrame, ml_cfg: dict = None) -> pd.DataFrame:
    """Executa uma estratégia específica e retorna sinais"""
    
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
            ml_strategy.train_model(
                df, 
                model_type=ml_cfg.get('model_type', 'RandomForest'),
                horizon=ml_cfg.get('horizon', 5),
                threshold=ml_cfg.get('threshold', 0.01)
            )
            st.session_state['ml_strategy'] = ml_strategy
            return ml_strategy.generate_signals(
                df,
                buy_threshold=ml_cfg.get('buy_threshold', 0.6),
                sell_threshold=ml_cfg.get('sell_threshold', 0.4)
            )
        else:
            return pd.DataFrame(columns=['signal', 'stop', 'target'])
            
    except Exception as e:
        st.error(f"Erro em {strategy_name}: {str(e)}")
        return pd.DataFrame(columns=['signal', 'stop', 'target'])

# Validação de inputs
def validate_inputs():
    if not ticker:
        st.error("Insira um ticker válido")
        return False
    if not strategies_selected:
        st.error("Selecione pelo menos uma estratégia")
        return False
    
    risk_errors = validate_risk_parameters(capital_brl, risk_pct, fx_rate, 100)
    if risk_errors:
        for field, error in risk_errors.items():
            st.error(f"{field}: {error}")
        return False
    return True

# Execução principal
if run_analysis:
    if not validate_inputs():
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
            signals = run_strategy(strategy, df, ml_config)
            
            if signals.empty or (signals['signal'] == 0).all():
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
            st.error(f"Erro em {strategy}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(strategies_selected))
    
    status_text.success("Análise concluída!")
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("Nenhuma estratégia foi executada com sucesso")
        st.stop()
    
    # Salvar no session state
    st.session_state['results'] = results
    st.session_state['df'] = df
    st.session_state['asset_info'] = asset_info
    st.session_state['fx_rate'] = fx_rate

# Exibir resultados
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
            if len(rankings) >= 3:
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
                            <p><strong>Trades:</strong> {int(metrics['Num Trades'])}</p>
                            <p><strong>Hit Rate:</strong> {metrics['Hit Rate']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Métricas da melhor estratégia
            best_strategy = rankings[0][0]
            best_metrics = results[best_strategy]['metrics']
            
            st.subheader(f"📈 Métricas da Melhor Estratégia: {best_strategy}")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("CAGR", f"{best_metrics['CAGR']:.2%}")
            col2.metric("Sharpe", f"{best_metrics['Sharpe Ratio']:.2f}")
            col3.metric("Max DD", f"{best_metrics['Max Drawdown']:.2%}")
            col4.metric("Calmar", f"{best_metrics['Calmar Ratio']:.2f}")
            col5.metric("Hit Rate", f"{best_metrics['Hit Rate']:.2%}")
            col6.metric("Profit Factor", f"{best_metrics['Profit Factor']:.2f}")
            
            # Sinal atual
            st.subheader("🎯 Sinal Atual da Melhor Estratégia")
            
            current_signals = results[best_strategy]['signals']
            if not current_signals.empty:
                last_signal = current_signals.iloc[-1]
                current_price = df['Close'].iloc[-1]
                
                signal_text = "FLAT"
                signal_class = "signal-flat"
                
                if last_signal['signal'] == 1:
                    signal_text = "COMPRA"
                    signal_class = "signal-buy"
                elif last_signal['signal'] == -1:
                    signal_text = "VENDA"
                    signal_class = "signal-sell"
                
                stop_text = f"{last_signal.get('stop', 'N/A'):.2f}" if not pd.isna(last_signal.get('stop', np.nan)) else 'N/A'
                target_text = f"{last_signal.get('target', 'N/A'):.2f}" if not pd.isna(last_signal.get('target', np.nan)) else 'N/A'
                
                confidence_text = ""
                if 'probability' in last_signal and not pd.isna(last_signal['probability']):
                    confidence_text = f"<p>Confiança (ML): {last_signal['probability']:.2%}</p>"
                
                st.markdown(f"""
                <div class="{signal_class}">
                    <h3>Sinal: {signal_text}</h3>
                    <p>Preço Atual: {current_price:.2f}</p>
                    <p>Stop Sugerido: {stop_text}</p>
                    <p>Target Sugerido: {target_text}</p>
                    {confidence_text}
                </div>
                """, unsafe_allow_html=True)
            
            # Comparação visual
            st.subheader("📊 Comparação de Estratégias")
            try:
                fig_comparison = plot_strategy_comparison(metrics_only)
                st.plotly_chart(fig_comparison, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao plotar comparação: {e}")
    
    with tab2:
        st.header("📈 Resultados dos Backtests")
        
        # Tabela de métricas
        try:
            metrics_table = create_metrics_table(metrics_only)
            if not metrics_table.empty:
                st.dataframe(metrics_table, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao criar tabela: {e}")
        
        # Detalhes por estratégia
        selected_strategy = st.selectbox(
            "Selecione uma estratégia para detalhes:",
            options=list(results.keys())
        )
        
        if selected_strategy:
            strategy_metrics = results[selected_strategy]['metrics']
            
            # Curva de capital
            st.subheader(f"💹 Curva de Capital - {selected_strategy}")
            try:
                fig_equity = plot_equity_curve(
                    strategy_metrics['Equity Curve'],
                    strategy_metrics['Drawdown Series'],
                    f"Evolução do Capital - {selected_strategy}"
                )
                st.plotly_chart(fig_equity, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao plotar curva: {e}")
            
            # Retornos mensais
            try:
                if len(strategy_metrics['Equity Curve']) > 30:
                    st.subheader("📅 Retornos Mensais")
                    fig_monthly = plot_monthly_returns(strategy_metrics['Equity Curve'])
                    st.plotly_chart(fig_monthly, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao plotar retornos mensais: {e}")
    
    with tab3:
        st.header("📉 Gráfico de Trades")
        
        chart_strategy = st.selectbox(
            "Estratégia para visualizar:",
            options=list(results.keys()),
            key="chart_strategy"
        )
        
        if chart_strategy:
            trades = results[chart_strategy]['trades']
            
            try:
                fig_trades = plot_candlestick_with_trades(
                    df, trades, f"{ticker} - {chart_strategy}"
                )
                st.plotly_chart(fig_trades, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao plotar gráfico: {e}")
            
            # Estatísticas dos trades
            if trades:
                st.subheader("📊 Estatísticas dos Trades")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total de Trades", len(trades))
                
                winning_trades = [t for t in trades if t.net_pnl > 0]
                col2.metric("Trades Vencedores", len(winning_trades))
                
                avg_pnl = np.mean([t.net_pnl for t in trades])
                col3.metric("PnL Médio", f"{avg_pnl:.2f}")
                
                avg_bars = np.mean([t.bars_held for t in trades])
                col4.metric("Barras Médias", f"{avg_bars:.1f}")
    
    with tab4:
        st.header("🤖 Machine Learning")
        
        if "Machine Learning" in results and 'ml_strategy' in st.session_state:
            try:
                ml_data = st.session_state.ml_strategy.validation_results
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("AUC Score", f"{ml_data.get('auc_score', 0):.3f}")
                    st.metric("Accuracy", f"{ml_data.get('accuracy', 0):.3f}")
                
                with col2:
                    if 'feature_importance' in ml_data and ml_data['feature_importance']:
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
            except Exception as e:
                st.error(f"Erro ao exibir dados ML: {e}")
        else:
            st.info("Machine Learning não foi selecionado ou falhou na execução.")
    
        with tab5:
        st.header("💼 Sizing & Gestão de Risco")
        
        # Informações do ativo
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ℹ️ Informações do Ativo")
            if asset_info:
                st.write(f"**Nome:** {asset_info.get('name', 'N/A')}")
                st.write(f"**Setor:** {asset_info.get('sector', 'N/A')}")
                st.write(f"**Moeda:** {asset_info.get('currency', 'N/A')}")
            st.write(f"**Preço Atual:** {df['Close'].iloc[-1]:.2f}")
        
        with col2:
            st.subheader("💱 Parâmetros de Risco")
            st.metric("Capital Total", f"R$ {capital_brl:,.2f}")
            st.metric("Risco por Trade", f"{risk_pct:.2%}")
            st.metric("Taxa de Câmbio USDBRL", f"{fx_rate:.4f}")

        st.divider()
        
        # Cálculo de posição
        st.subheader("📊 Cálculo de Posição Sugerido")
        st.write("Baseado no sinal mais recente da estratégia com melhor ranking.")
        
        if rankings:
            best_strategy = rankings[0][0]
            best_signals = results[best_strategy]['signals']
            
            if not best_signals.empty:
                last_signal = best_signals.iloc[-1]
                current_price = df['Close'].iloc[-1]
                
                # Verifica se o sinal atual é para estar posicionado (compra ou venda)
                if last_signal['signal'] != 0 and not pd.isna(last_signal.get('stop', np.nan)):
                    stop_price = last_signal['stop']
                    stop_distance = abs(current_price - stop_price)

                    # Se a distância do stop for zero, evitamos divisão por zero
                    if stop_distance > 0:
                        is_usd_asset = asset_info.get('currency', '').upper() == 'USD'
                        
                        # Chama a função importada para calcular o tamanho da posição
                        position_size = calculate_position_size(
                            capital=capital_brl,
                            risk_per_trade_pct=risk_pct,
                            stop_loss_distance=stop_distance,
                            price=current_price,
                            fx_rate=fx_rate if is_usd_asset else 1.0 # Usa o câmbio somente se o ativo for em USD
                        )
                        
                        # Calcula o valor financeiro da posição e o risco
                        financial_position_brl = position_size * current_price * (fx_rate if is_usd_asset else 1.0)
                        risk_amount_brl = capital_brl * risk_pct
                        
                        st.info(f"Sinal da estratégia **{best_strategy}**: {'COMPRA' if last_signal['signal'] == 1 else 'VENDA'}")

                        res_col1, res_col2, res_col3 = st.columns(3)
                        res_col1.metric("📈 Tamanho da Posição (unidades)", f"{position_size:,.0f}")
                        res_col2.metric("💰 Valor Financeiro (BRL)", f"R$ {financial_position_brl:,.2f}")
                        res_col3.metric("🔥 Risco Financeiro (BRL)", f"R$ {risk_amount_brl:,.2f}")
                        
                        st.caption(f"Cálculo baseado no preço atual de {current_price:.2f} e stop em {stop_price:.2f}.")

                    else:
                        st.warning("O preço atual é igual ao preço do stop. Não é possível calcular o tamanho da posição.")
                
                elif last_signal['signal'] == 0:
                     st.info("O sinal atual da melhor estratégia é **FLAT (neutro)**. Nenhum cálculo de posição é necessário.")

                else: # Sinal de compra/venda mas sem stop definido
                    st.error(f"A estratégia '{best_strategy}' gerou um sinal, mas não forneceu um preço de stop loss. O cálculo de dimensionamento não é possível.")

            else:
                st.warning(f"A estratégia '{best_strategy}' não gerou nenhum sinal no período analisado.")
        else:
            st.error("Não foi possível rankear as estratégias para calcular o sizing.")

else:
    # Tela inicial
    st.markdown("""
    ## 🚀 Bem-vindo ao Lab de Estratégias!
    
    Este aplicativo permite testar e comparar múltiplas estratégias de trading com:
    
    ### ✅ **Funcionalidades Principais:**
    - **16 Estratégias Disponíveis**: Clássicas renomadas + inventadas inovadoras
    - **Machine Learning**: Modelos preditivos com validação temporal
    - **Gestão de Risco**: Sizing automático BRL→USD
    - **Backtests Robustos**: Engine sem look-ahead com custos realistas
    - **Visualizações Interativas**: Gráficos e métricas detalhadas
    
    ### 🎯 **Estratégias Implementadas:**
    
    **Clássicas:**
    - EMA/SMA Crossover com filtros
    - RSI (IFR2 e padrão)
    - MACD com histograma
    - Bollinger Bands (breakout e mean reversion)
    - Donchian/Turtle Trading
    - Momentum/ROC
    - Breakout de máximas/mínimas
    - ADX + DMI
    - Padrões de Candlesticks
    
    **Inventadas:**
    - **Vol-Regime Switch**: Alterna entre mean-reversion e breakout
    - **Meta-Ensemble**: Voto entre múltiplas estratégias
    - **Pullback Trend-Bias**: Reteste de EMA com confirmação ADX
    
    ### 📋 **Como usar:**
    1. Configure os parâmetros na barra lateral
    2. Selecione as estratégias desejadas
    3. Clique em "🚀 Executar Análise"
    4. Explore os resultados nas abas
    
    **Configure os parâmetros na barra lateral e clique em "Executar Análise" para começar!**
    """)
    
    # Exemplo de dados
    st.subheader("📊 Exemplo de Dados")
    sample_data = pd.DataFrame({
        'Data': pd.date_range('2024-01-01', periods=5),
        'Open': [100, 102, 101, 103, 105],
        'High': [103, 104, 103, 106, 107],
        'Low': [99, 101, 100, 102, 104],
        'Close': [102, 101, 103, 105, 106],
        'Volume': [1000000, 1200000, 800000, 1500000, 1100000]
    })
    st.dataframe(sample_data, use_container_width=True)
