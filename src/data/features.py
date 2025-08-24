# src/data/features.py
import numpy as np
import pandas as pd
from scipy import stats

class FeatureEngine:
    def __init__(self, config):
        self.config = config
    
    def calculate_technical_indicators(self, df):
        """Calcula indicadores técnicos robustos"""
        data = df.copy()
        
        # Retornos
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Remove outliers extremos (>5 desvios padrão)
        ret_std = data['returns'].std()
        ret_mean = data['returns'].mean()
        outlier_mask = np.abs(data['returns'] - ret_mean) > 5 * ret_std
        data.loc[outlier_mask, 'returns'] = np.nan
        data['returns'] = data['returns'].fillna(method='ffill')
        
        # EMAs
        for period in [9, 21, 50]:
            data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI robusto
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(self.config['risk']['atr_window']).mean()
        
        # Features adicionais para ML
        data['price_momentum_5'] = data['Close'].pct_change(5)
        data['price_momentum_20'] = data['Close'].pct_change(20)
        data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['volatility_20'] = data['returns'].rolling(20).std() * np.sqrt(252)
        
        # Trend features
        data['ema_cross_up'] = ((data['EMA_9'] > data['EMA_21']) & 
                               (data['EMA_9'].shift(1) <= data['EMA_21'].shift(1))).astype(int)
        data['price_above_ema21'] = (data['Close'] > data['EMA_21']).astype(int)
        data['rsi_oversold'] = (data['RSI'] < 30).astype(int)
        data['rsi_overbought'] = (data['RSI'] > 70).astype(int)
        
        # Target para classificação (retorno 1 dia à frente > 0)
        data['target'] = (data['returns'].shift(-1) > 0).astype(int)
        
        return data.dropna()
    
    def get_feature_matrix(self, data):
        """Prepara matriz de features para ML"""
        feature_cols = [
            'returns', 'log_returns', 'price_momentum_5', 'price_momentum_20',
            'RSI', 'volume_ratio', 'volatility_20', 'ema_cross_up',
            'price_above_ema21', 'rsi_oversold', 'rsi_overbought'
        ]
        
        # Adiciona ratios de EMAs
        for period in [9, 21, 50]:
            col_name = f'price_ema_{period}_ratio'
            data[col_name] = data['Close'] / data[f'EMA_{period}'] - 1
            feature_cols.append(col_name)
        
        X = data[feature_cols].copy()
        y = data['target'].values
        
        return X, y, feature_cols
