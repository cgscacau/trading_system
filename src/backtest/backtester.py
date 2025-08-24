# src/backtest/backtester.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Backtester:
    def __init__(self, config, models, risk_manager):
        self.config = config
        self.models = models
        self.risk_manager = risk_manager
        self.last_retrain_date = None
        
        # Features padrão para consistência
        self.default_features = [
            'returns', 'log_returns', 'price_momentum_5', 'price_momentum_20',
            'RSI', 'volume_ratio', 'volatility_20', 'ema_cross_up',
            'price_above_ema21', 'rsi_oversold', 'rsi_overbought',
            'price_ema_9_ratio', 'price_ema_21_ratio', 'price_ema_50_ratio'
        ]

    def run_backtest(self, data_dict):
        """Executa backtesting completo"""
        results = {}
        all_trades = []
        
        for symbol, data in data_dict.items():
            print(f"🔄 Backtesting {symbol}...")
            try:
                equity_curve, trades = self._backtest_symbol(symbol, data.copy())
                results[symbol] = {
                    'equity_curve': equity_curve,
                    'trades': trades
                }
                all_trades.extend(trades)
                print(f"✅ {symbol}: {len(trades)} trades executados")
            except Exception as e:
                print(f"❌ Erro no backtest de {symbol}: {e}")
                continue
        
        # Combina resultados
        combined_equity = self._combine_equity_curves(results)
        
        return {
            'combined_equity': combined_equity,
            'individual_results': results,
            'all_trades': pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
        }

    def _backtest_symbol(self, symbol, data):
        """Backtest para um símbolo específico"""
        # Garante que Date seja datetime
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        
        # Divide dados
        train_end = pd.to_datetime(self.config['split']['test_start'])
        train_data = data[data['Date'] < train_end].copy()
        test_data = data[data['Date'] >= train_end].copy()
        
        if len(train_data) < 100 or len(test_data) < 10:
            print(f"⚠️ Dados insuficientes para {symbol}")
            return pd.DataFrame(), []
        
        # Treina modelos iniciais
        self._train_models(train_data, symbol)
        
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
                if len(retrain_data) >= 100:
                    self._train_models(retrain_data, symbol)
            
            # Gerencia posição atual
            if current_position:
                equity, position_closed = self._manage_position(
                    current_position, row, equity, symbol
                )
                if position_closed:
                    trades.append(position_closed)
                    current_position = None
            
            # Gera novo sinal se não há posição
            if not current_position:
                signal_data = self._generate_signal(row, symbol)
                if signal_data and signal_data['signal'] in ['BUY', 'SELL']:
                    current_position = self._open_position(
                        signal_data, row, equity, symbol
                    )
            
            # Atualiza curva de equity
            current_equity = equity
            if current_position:
                # Inclui P&L não realizado
                unrealized_pnl = self._calculate_unrealized_pnl(current_position, row)
                current_equity += unrealized_pnl
            
            equity_curve.append({
                'Date': current_date,
                'Equity': current_equity
            })
        
        # Fecha posição final se existir
        if current_position and not test_data.empty:
            final_row = test_data.iloc[-1]
            equity, final_trade = self._manage_position(
                current_position, final_row, equity, symbol, force_close=True
            )
            if final_trade:
                trades.append(final_trade)
        
        return pd.DataFrame(equity_curve), trades

    def _train_models(self, data, symbol):
        """Treina todos os modelos com dados disponíveis"""
        try:
            returns = data['returns'].dropna()
            
            # Treina ARIMA
            if len(returns) >= 50:
                try:
                    success = self.models['arima'].fit(returns)
                    if success:
                        print(f"✅ ARIMA treinado para {symbol}")
                except Exception as e:
                    print(f"⚠️ Erro ARIMA {symbol}: {e}")
            
            # Treina GARCH
            if len(returns) >= 100:
                try:
                    success = self.models['garch'].fit(returns)
                    if success:
                        print(f"✅ GARCH treinado para {symbol}")
                except Exception as e:
                    print(f"⚠️ Erro GARCH {symbol}: {e}")
            
            # Treina modelos ML
            try:
                # Filtra features disponíveis
                available_features = [f for f in self.default_features if f in data.columns]
                
                if available_features and 'target' in data.columns:
                    ml_data = data.dropna(subset=available_features + ['target'])
                    
                    if len(ml_data) > 20:
                        X = ml_data[available_features]
                        y = ml_data['target'].values
                        
                        if len(np.unique(y)) > 1:
                            # Random Forest
                            try:
                                success = self.models['rf'].fit(X, y)
                                if success:
                                    print(f"✅ Random Forest treinado para {symbol}")
                            except Exception as e:
                                print(f"⚠️ Erro RF {symbol}: {e}")
                            
                            # Trend Score
                            try:
                                success = self.models['trend'].fit(X, y)
                                if success:
                                    print(f"✅ Trend Score treinado para {symbol}")
                            except Exception as e:
                                print(f"⚠️ Erro Logit {symbol}: {e}")
                        
            except Exception as e:
                print(f"⚠️ Erro ML {symbol}: {e}")
            
            self.last_retrain_date = data['Date'].max()
            
        except Exception as e:
            print(f"❌ Erro geral no treinamento {symbol}: {e}")

    def _should_retrain(self, current_date):
        """Verifica se deve retreinar os modelos"""
        if self.last_retrain_date is None:
            return False
        
        freq = self.config['split'].get('retrain_frequency', 'M')
        
        if freq == 'M':  # Mensal
            return (current_date.year, current_date.month) != \
                   (self.last_retrain_date.year, self.last_retrain_date.month)
        elif freq == 'W':  # Semanal
            return (current_date - self.last_retrain_date).days >= 7
        elif freq == 'D':  # Diário
            return (current_date - self.last_retrain_date).days >= 1
        
        return False

    def _generate_signal(self, row, symbol):
        """Gera sinal usando ensemble de modelos"""
        try:
            predictions = {}
            
            # ARIMA prediction
            if hasattr(self.models['arima'], 'fitted_model') and \
               self.models['arima'].fitted_model:
                try:
                    _, prob_arima = self.models['arima'].forecast_probability()
                    predictions['arima'] = prob_arima
                except:
                    predictions['arima'] = 0.5
            
            # GARCH volatility
            vol_forecast = 0.02
            if hasattr(self.models['garch'], 'fitted_model') and \
               self.models['garch'].fitted_model:
                try:
                    vol_forecast = self.models['garch'].forecast_volatility()
                except:
                    pass
            
            # ML predictions
            feature_row = self._extract_features(row)
            if feature_row is not None:
                # Random Forest
                if hasattr(self.models['rf'], 'is_fitted') and \
                   self.models['rf'].is_fitted:
                    try:
                        predictions['rf'] = self.models['rf'].predict_proba(feature_row)
                    except:
                        predictions['rf'] = 0.5
                
                # Trend Score
                if hasattr(self.models['trend'], 'is_fitted') and \
                   self.models['trend'].is_fitted:
                    try:
                        predictions['logit'] = self.models['trend'].predict_proba(feature_row)
                    except:
                        predictions['logit'] = 0.5
            
            # Se não há predições válidas
            if not predictions:
                return {
                    'signal': 'NEUTRAL',
                    'probability': 0.5,
                    'volatility': vol_forecast,
                    'predictions': {}
                }
            
            # Ensemble
            ensemble_prob, signal = self.models['ensemble'].combine_predictions(predictions)
            
            return {
                'signal': signal,
                'probability': ensemble_prob,
                'volatility': vol_forecast,
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"⚠️ Erro na geração de sinal para {symbol}: {e}")
            return None

    def _extract_features(self, row):
        """Extrai features da linha atual"""
        try:
            # Usa features do RF se disponível, senão usa padrão
            feature_names = getattr(self.models['rf'], 'feature_names', self.default_features)
            
            # Filtra apenas features disponíveis na linha
            available_features = [f for f in feature_names if f in row.index]
            
            if not available_features:
                return None
            
            # Extrai valores das features
            feature_values = []
            for feature in available_features:
                value = row[feature]
                if pd.isna(value) or np.isinf(value):
                    feature_values.append(0.0)
                else:
                    feature_values.append(float(value))
            
            return feature_values
            
        except Exception as e:
            print(f"⚠️ Erro na extração de features: {e}")
            return None

    def _open_position(self, signal_data, row, equity, symbol):
        """Abre nova posição"""
        try:
            entry_price = float(row['Close'])
            atr = row.get('ATR', entry_price * 0.02)
            
            if pd.isna(atr) or atr <= 0:
                return None
            
            # Calcula stops e targets
            stop_loss, take_profit = self.risk_manager.calculate_stops_and_targets(
                entry_price, atr, signal_data['signal']
            )
            
            # Calcula tamanho da posição
            position_size = self.risk_manager.calculate_position_size(
                entry_price, stop_loss, equity, signal_data['probability']
            )
            
            if position_size <= 0:
                return None
            
            position = {
                'symbol': symbol,
                'signal': signal_data['signal'],
                'entry_price': entry_price,
                'entry_date': row['Date'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'probability': signal_data['probability']
            }
            
            print(f"📈 {symbol} {signal_data['signal']}: Entry={entry_price:.2f}, Size={position_size}")
            return position
            
        except Exception as e:
            print(f"⚠️ Erro ao abrir posição {symbol}: {e}")
            return None

    def _manage_position(self, position, row, equity, symbol, force_close=False):
        """Gerencia posição existente"""
        try:
            current_price = float(row['Close'])
            current_date = row['Date']
            
            # Verifica condições de saída
            exit_reason = None
            
            if force_close:
                exit_reason = 'FORCE_CLOSE'
            elif position['signal'] == 'BUY':
                if current_price <= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                elif current_price >= position['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
            else:  # SELL
                if current_price >= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                elif current_price <= position['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
            
            # Verifica tempo máximo
            days_held = (current_date - position['entry_date']).days
            max_hold = self.config['risk'].get('max_hold_days', 30)
            if days_held >= max_hold:
                exit_reason = 'MAX_TIME'
            
            if not exit_reason:
                return equity, None
            
            # Calcula P&L
            if position['signal'] == 'BUY':
                pnl_gross = (current_price - position['entry_price']) * position['position_size']
            else:  # SELL
                pnl_gross = (position['entry_price'] - current_price) * position['position_size']
            
            # Aplica custos
            notional = current_price * position['position_size']
            commission = notional * (self.config['backtest']['commission_bps'] / 10000)
            slippage = notional * (self.config['backtest']['slippage_bps'] / 10000)
            
            pnl_net = pnl_gross - commission - slippage
            new_equity = equity + pnl_net
            
            trade_record = {
                'Symbol': symbol,
                'Entry_Date': position['entry_date'],
                'Exit_Date': current_date,
                'Signal': position['signal'],
                'Entry_Price': position['entry_price'],
                'Exit_Price': current_price,
                'Position_Size': position['position_size'],
                'PnL': pnl_net,
                'Return_Pct': (pnl_net / (position['entry_price'] * position['position_size'])) * 100,
                'Exit_Reason': exit_reason,
                'Days_Held': days_held
            }
            
            print(f"📉 {symbol} fechado: {exit_reason}, P&L={pnl_net:.2f}")
            return new_equity, trade_record
            
        except Exception as e:
            print(f"⚠️ Erro no gerenciamento {symbol}: {e}")
            return equity, None

    def _calculate_unrealized_pnl(self, position, row):
        """Calcula P&L não realizado"""
        try:
            current_price = float(row['Close'])
            if position['signal'] == 'BUY':
                return (current_price - position['entry_price']) * position['position_size']
            else:  # SELL
                return (position['entry_price'] - current_price) * position['position_size']
        except:
            return 0.0

    def _combine_equity_curves(self, results):
        """Combina curvas de equity de múltiplos símbolos"""
        if not results:
            return pd.DataFrame({
                'Date': [pd.Timestamp.now()],
                'Equity': [self.config['backtest']['initial_equity']]
            })
        
        if len(results) == 1:
            return list(results.values())[0]['equity_curve']
        
        # Combina múltiplos símbolos
        all_curves = []
        initial_equity = self.config['backtest']['initial_equity']
        
        for symbol, result in results.items():
            curve = result['equity_curve'].copy()
            if not curve.empty:
                curve['Date'] = pd.to_datetime(curve['Date'])
                # Normaliza para retornos
                curve['Return'] = curve['Equity'] / initial_equity
                curve = curve[['Date', 'Return']].rename(columns={'Return': f'Return_{symbol}'})
                all_curves.append(curve)
        
        if not all_curves:
            return pd.DataFrame({
                'Date': [pd.Timestamp.now()],
                'Equity': [initial_equity]
            })
        
        # Merge por data
        combined = all_curves[0]
        for curve in all_curves[1:]:
            combined = pd.merge(combined, curve, on='Date', how='outer')
        
        combined = combined.sort_values('Date').fillna(method='ffill').fillna(1.0)
        
        # Calcula retorno médio do portfólio
        return_cols = [col for col in combined.columns if col.startswith('Return_')]
        combined['Portfolio_Return'] = combined[return_cols].mean(axis=1)
        combined['Equity'] = initial_equity * combined['Portfolio_Return']
        
        return combined[['Date', 'Equity']]
