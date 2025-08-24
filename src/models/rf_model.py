# src/models/rf_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class RandomForestModel:
    def __init__(self, config):
        self.model = RandomForestClassifier(
            n_estimators=config['models']['rf']['n_estimators'],
            max_depth=config['models']['rf']['max_depth'],
            random_state=config['models']['rf']['random_state'],
            class_weight='balanced',
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, X, y):
        """Treina Random Forest"""
        try:
            if len(np.unique(y)) < 2:
                return False
                
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.feature_names = list(X.columns)
            self.is_fitted = True
            return True
            
        except Exception as e:
            print(f"Erro no treinamento RF: {e}")
            return False
    
    def predict_proba(self, X_row):
        """Prediz probabilidade de alta"""
        if not self.is_fitted or not self.feature_names:
            return 0.5
            
        try:
            # Garante ordem correta das features
            if len(X_row) != len(self.feature_names):
                return 0.5
                
            X_scaled = self.scaler.transform([X_row])
            prob = self.model.predict_proba(X_scaled)[0, 1]
            return float(np.clip(prob, 0.1, 0.9))
            
        except Exception:
            return 0.5
