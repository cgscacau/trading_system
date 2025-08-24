# src/risk/risk_manager.py
import numpy as np

class RiskManager:
    def __init__(self, config):
        self.config = config
    
    def calculate_kelly_fraction(self, prob_win, avg_win, avg_loss):
        """Calcula fração de Kelly otimizada"""
        if avg_loss <= 0 or prob_win <= 0:
            return 0.0
            
        win_loss_ratio = avg_win / avg_loss
        kelly_f = prob_win - (1 - prob_win) / win_loss_ratio
        
        # Aplica fração e cap
        kelly_adjusted = max(0, kelly_f) * self.config['kelly_fraction']
        return min(kelly_adjusted, self.config['kelly_cap'])
    
    def calculate_position_size(self, entry_price, stop_price, equity, prob_up):
        """Calcula tamanho da posição"""
        risk_amount = equity * self.config['risk_per_trade_pct']
        per_unit_risk = abs(entry_price - stop_price)
        
        if per_unit_risk <= 0:
            return 0
        
        base_quantity = risk_amount / per_unit_risk
        
        if self.config['use_kelly']:
            # Usa Kelly com probabilidade do ensemble
            target_ratio = self.config['target_ratio']
            kelly_f = self.calculate_kelly_fraction(prob_up, target_ratio, 1.0)
            base_quantity *= (1 + kelly_f)
        
        return max(int(base_quantity), 0)
    
    def calculate_stops_and_targets(self, entry_price, atr, signal):
        """Calcula stop loss e take profit"""
        atr_mult = self.config['atr_multiplier']
        target_ratio = self.config['target_ratio']
        
        if signal == 'BUY':
            stop_loss = entry_price - (atr_mult * atr)
            risk = entry_price - stop_loss
            take_profit = entry_price + (target_ratio * risk)
        else:  # SELL
            stop_loss = entry_price + (atr_mult * atr)
            risk = stop_loss - entry_price
            take_profit = entry_price - (target_ratio * risk)
        
        return stop_loss, take_profit
