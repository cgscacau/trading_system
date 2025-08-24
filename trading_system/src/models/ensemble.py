# src/models/ensemble.py
import numpy as np

class EnsemblePredictor:
    def __init__(self, config):
        self.weights = config['ensemble']['weights']
        self.buy_threshold = config['ensemble']['thresholds']['buy']
        self.sell_threshold = config['ensemble']['thresholds']['sell']
    
    def combine_predictions(self, predictions):
        """Combina previsões dos modelos com pesos ajustáveis"""
        available_models = []
        available_weights = []
        available_probs = []
        
        for model_name, prob in predictions.items():
            if prob is not None and not np.isnan(prob):
                available_models.append(model_name)
                available_weights.append(self.weights.get(model_name, 0))
                available_probs.append(prob)
        
        if not available_weights or sum(available_weights) == 0:
            return 0.5, 'NEUTRAL'
        
        # Normaliza pesos
        total_weight = sum(available_weights)
        normalized_weights = [w / total_weight for w in available_weights]
        
        # Calcula probabilidade ponderada
        ensemble_prob = sum(w * p for w, p in zip(normalized_weights, available_probs))
        
        # Gera sinal
        if ensemble_prob >= self.buy_threshold:
            signal = 'BUY'
        elif ensemble_prob <= self.sell_threshold:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'
            
        return float(ensemble_prob), signal
