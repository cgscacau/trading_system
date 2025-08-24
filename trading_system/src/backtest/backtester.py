# src/backtest/backtester.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Backtester:
    def __init__(self, config, models, risk_manager):
        self.config = config
        self.models = models
        self.risk_manager = risk_manager
        
    def run_backtest(self, data_dict):
        """Executa backtesting completo"""
        results = {}
        all_trades = []
        
        for symbol, data in data_dict.items():
            print(f"Backtesting {symbol}...")
            equity_curve, trades = self._backtest_symbol(symbol, data)
            results[symbol] = {
                'equity_curve': equity_curve,
                'trades': trades
            }
            all_trades.extend(trades)
        
        # Combina resultados
        combined_equity = self._combine_equity_curves(results)
        
        return {
            'combined_equity': combined_equity,
            'individual_results': results,
            'all_trades': pd.DataFrame(all_trades)
        }
    
    def _backtest_symbol(self, symbol, data):
        """Backtest para um símbolo específico"""
        # Divide dados
        train_end = pd.to_datetime(self.config['split']['test_start'])
        train_data = data[data['Date'] < train_end].copy()
        test_data = data[data['Date'] >= train_end].copy()
        
        # Treina modelos iniciais
        self._train_models(train_data)
        
        # Inicializa variáveis
        equity = self.config['backtest']['initial_equity']
        equity_curve = []
        trades = []
        current_position = None
        
        for idx, row in test_data.iterrows():
            current_date = row['Date']
            
            # Re-treina modelos periodicamente
            if self._should_retrain(current_date):
                retrain_data = data[data['Date'] < current_date].copy()
                self._train_models(retrain_data)
            
            # Gerencia posição atual
            if current_position:
                equity, position_closed = self._manage_position(
                    current_position, row, equity
                )
                if position_closed:
                    trades.append(position_closed)
                    current_position = None
            
            # Gera novo sinal se não há posição
            if not current_position:
                signal_data = self._generate_signal(row)
                if signal_data['signal'] in ['BUY', 'SELL']:
                    current_position = self._open_position(
                        signal_data, row, equity
                    )
            
            equity_curve.append({
                'Date': current_date,
                'Equity': equity
            })
        
        return pd.DataFrame(equity_curve), trades
    
    def _generate_signal(self, row):
        """Gera sinal usando ensemble de modelos"""
        predictions = {}
        
        # ARIMA prediction
        if hasattr(self.models['arima'], 'fitted_model') and self.models['arima'].fitted_model:
            _, prob_arima = self.models['arima'].forecast_probability()
            predictions['arima'] = prob_arima
        
        # GARCH volatility (usado para ajustar confiança)
        vol_forecast = self.models['garch'].forecast_volatility()
        
        # ML predictions
        feature_row = self._extract_features(row)
        if feature_row is not None:
            predictions['rf'] = self.models['rf'].predict_proba(feature_row)
            predictions['logit'] = self.models['trend'].predict_proba(feature_row)
        
        # Ensemble
        ensemble_prob, signal = self.models['ensemble'].combine_predictions(predictions)
        
        return {
            'signal': signal,
            'probability': ensemble_prob,
            'volatility': vol_forecast,
            'predictions': predictions
        }
