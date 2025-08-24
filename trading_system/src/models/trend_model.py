# src/models/trend_model.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class TrendScoreModel:
    def __init__(self, config):
        self.model = LogisticRegression(
            C=config['models']['logit']['C'],
            penalty=config['models']['logit']['penalty'],
            max_iter=config['models']['logit']['max_iter'],
            solver='lbfgs'
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """Treina modelo Logit"""
        try:
            if len(np.unique(y)) < 2:
                return False
                
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            return True
            
        except Exception as e:
            print(f"Erro no treinamento Logit: {e}")
            return False
    
    def predict_proba(self, X_row):
        """Prediz probabilidade de alta"""
        if not self.is_fitted:
            return 0.5
            
        try:
            X_scaled = self.scaler.transform([X_row])
            prob = self.model.predict_proba(X_scaled)[0, 1]
            return float(np.clip(prob, 0.1, 0.9))
            
        except Exception:
            return 0.5
