# main.py
import yaml
import pandas as pd
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
from src.backtest.plotting import TradingDashboard

def main():
    # Carrega configuração
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("🚀 Iniciando Sistema de Trading Quantitativo")
    
    # Inicializa componentes
    data_loader = DataLoader(config)
    feature_engine = FeatureEngine(config)
    
    # Coleta dados
    print("\n📊 Coletando dados...")
    raw_data = data_loader.download_data(
        config['data']['tickers'],
        config['data']['start'],
        config['data']['end']
    )
    
    # Processa features
    print("\n🔧 Processando features técnicas...")
    processed_data = {}
    for symbol, data in raw_data.items():
        processed_data[symbol] = feature_engine.calculate_technical_indicators(data)
    
    # Inicializa modelos
    models = {
        'arima': ARIMAModel(config),
        'garch': GARCHModel(config),
        'rf': RandomForestModel(config),
        'trend': TrendScoreModel(config),
        'ensemble': EnsemblePredictor(config)
    }
    
    risk_manager = RiskManager(config['risk'])
    
    # Executa backtest
    print("\n📈 Executando backtest...")
    backtester = Backtester(config, models, risk_manager)
    results = backtester.run_backtest(processed_data)
    
    # Calcula métricas
    print("\n📊 Calculando métricas de performance...")
    metrics = calculate_performance_metrics(
        results['combined_equity'],
        results['all_trades']
    )
    
    # Exibe resultados
    print("\n" + "="*50)
    print("📈 RESULTADOS DO BACKTEST")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Gera visualizações
    print("\n📊 Gerando visualizações...")
    dashboard = TradingDashboard()
    
    equity_fig = dashboard.create_equity_curve(results['combined_equity'])
    drawdown_fig = dashboard.create_drawdown_chart(results['combined_equity'])
    performance_fig = dashboard.create_performance_dashboard(results, metrics)
    
    # Salva resultados
    results['combined_equity'].to_csv('data/equity_curve.csv', index=False)
    results['all_trades'].to_csv('data/trades.csv', index=False)
    
    # Mostra gráficos
    equity_fig.show()
    drawdown_fig.show()
    performance_fig.show()
    
    print("\n✅ Sistema executado com sucesso!")
    print(f"📁 Resultados salvos em: data/")

if __name__ == "__main__":
    main()
