"""
Engine de backtest sem look-ahead com gestão de custos e stops/targets
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    """Estrutura de um trade executado"""
    entry_date: datetime
    exit_date: datetime
    side: str  # 'long' ou 'short'
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    quantity: float
    gross_pnl: float
    costs: float
    net_pnl: float
    r_multiple: float
    bars_held: int
    exit_reason: str

class BacktestEngine:
    """Engine de backtest discreto sem look-ahead"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 fee_pct: float = 0.001,
                 slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.trades: List[Trade] = []
        self.equity_curve = pd.Series(dtype=float)
        
    def run_backtest(self, 
                    df: pd.DataFrame,
                    signals: pd.DataFrame,
                    max_bars_in_trade: Optional[int] = None) -> Dict:
        """
        Executa backtest com sinais discretos
        
        Args:
            df: DataFrame com OHLCV
            signals: DataFrame com colunas 'signal', 'stop', 'target'
            max_bars_in_trade: Máximo de barras em posição
            
        Returns:
            Dict com resultados do backtest
        """
        # Resetar estado
        self.trades = []
        equity = [self.initial_capital]
        dates = [df.index[0]]
        
        # Estado da posição
        in_position = False
        position_side = None
        entry_price = 0.0
        stop_price = 0.0
        target_price = 0.0
        entry_date = None
        bars_held = 0
        
        # Alinhar dados
        df_aligned = df.reindex(signals.index).fillna(method='ffill')
        signals_aligned = signals.reindex(df.index).fillna(0)
        
        for i in range(1, len(df_aligned)):
            current_date = df_aligned.index[i]
            current_row = df_aligned.iloc[i]
            prev_signal_row = signals_aligned.iloc[i-1]
            
            # Preços atuais
            open_price = current_row['Open']
            high_price = current_row['High'] 
            low_price = current_row['Low']
            close_price = current_row['Close']
            
            # Sinal anterior (sem look-ahead)
            signal = prev_signal_row['signal']
            signal_stop = prev_signal_row.get('stop', np.nan)
            signal_target = prev_signal_row.get('target', np.nan)
            
            current_equity = equity[-1]
            
            # Gerenciar posição existente
            if in_position:
                bars_held += 1
                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""
                
                # Verificar stops/targets intrabar
                if position_side == 'long':
                    # Stop loss
                    if not np.isnan(stop_price) and low_price <= stop_price:
                        exit_price = self._apply_slippage(stop_price, 'sell')
                        exit_reason = "Stop Loss"
                        exit_triggered = True
                    # Take profit
                    elif not np.isnan(target_price) and high_price >= target_price:
                        exit_price = self._apply_slippage(target_price, 'sell')
                        exit_reason = "Take Profit"
                        exit_triggered = True
                        
                elif position_side == 'short':
                    # Stop loss
                    if not np.isnan(stop_price) and high_price >= stop_price:
                        exit_price = self._apply_slippage(stop_price, 'buy')
                        exit_reason = "Stop Loss"
                        exit_triggered = True
                    # Take profit
                    elif not np.isnan(target_price) and low_price <= target_price:
                        exit_price = self._apply_slippage(target_price, 'buy')
                        exit_reason = "Take Profit"
                        exit_triggered = True
                
                # Sinal oposto
                if not exit_triggered and signal != 0 and signal != (1 if position_side == 'long' else -1):
                    exit_price = self._apply_slippage(open_price, 'sell' if position_side == 'long' else 'buy')
                    exit_reason = "Opposite Signal"
                    exit_triggered = True
                
                # Tempo máximo
                if not exit_triggered and max_bars_in_trade and bars_held >= max_bars_in_trade:
                    exit_price = self._apply_slippage(close_price, 'sell' if position_side == 'long' else 'buy')
                    exit_reason = "Max Time"
                    exit_triggered = True
                
                # Executar saída
                if exit_triggered:
                    trade = self._create_trade(
                        entry_date, current_date, position_side,
                        entry_price, exit_price, stop_price, target_price,
                        bars_held, exit_reason
                    )
                    self.trades.append(trade)
                    current_equity += trade.net_pnl
                    
                    # Reset posição
                    in_position = False
                    position_side = None
                    bars_held = 0
            
            # Nova entrada
            if not in_position and signal != 0:
                position_side = 'long' if signal == 1 else 'short'
                entry_price = self._apply_slippage(open_price, 'buy' if signal == 1 else 'sell')
                stop_price = signal_stop if not np.isnan(signal_stop) else np.nan
                target_price = signal_target if not np.isnan(signal_target) else np.nan
                entry_date = current_date
                in_position = True
                bars_held = 0
            
            # Atualizar equity
            equity.append(current_equity)
            dates.append(current_date)
        
        # Fechar posição final se necessário
        if in_position:
            final_price = self._apply_slippage(df_aligned['Close'].iloc[-1], 
                                             'sell' if position_side == 'long' else 'buy')
            trade = self._create_trade(
                entry_date, df_aligned.index[-1], position_side,
                entry_price, final_price, stop_price, target_price,
                bars_held, "End of Data"
            )
            self.trades.append(trade)
            equity[-1] += trade.net_pnl
        
        # Criar equity curve
        self.equity_curve = pd.Series(equity, index=dates)
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'total_return': (equity[-1] - self.initial_capital) / self.initial_capital,
            'num_trades': len(self.trades)
        }
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Aplica slippage e fees ao preço"""
        cost_pct = self.fee_pct + self.slippage_pct
        if side in ['buy', 'cover']:
            return price * (1 + cost_pct)
        else:  # sell, short
            return price * (1 - cost_pct)
    
    def _create_trade(self, entry_date, exit_date, side, entry_price, exit_price,
                     stop_price, target_price, bars_held, exit_reason) -> Trade:
        """Cria objeto Trade com métricas calculadas"""
        
        # PnL bruto
        if side == 'long':
            gross_pnl = exit_price - entry_price
        else:  # short
            gross_pnl = entry_price - exit_price
        
        # Custos
        entry_cost = entry_price * (self.fee_pct + self.slippage_pct)
        exit_cost = exit_price * (self.fee_pct + self.slippage_pct)
        total_costs = entry_cost + exit_cost
        
        # PnL líquido
        net_pnl = gross_pnl - total_costs
        
        # R múltiplo
        if not np.isnan(stop_price):
            risk = abs(entry_price - stop_price)
            r_multiple = net_pnl / risk if risk > 0 else 0
        else:
            r_multiple = 0
        
        return Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=1.0,  # Normalizado para 1 unidade
            gross_pnl=gross_pnl,
            costs=total_costs,
            net_pnl=net_pnl,
            r_multiple=r_multiple,
            bars_held=bars_held,
            exit_reason=exit_reason
        )

