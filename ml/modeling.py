"""
Módulo de Machine Learning para previsão de movimentos de preço
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MLStrategy:
    """Estratégia baseada em Machine Learning"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.validation_results = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features para o modelo ML
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com features calculadas
        """
        features = pd.DataFrame(index=df.index)
        
        # Retornos defasados
        for lag in [1, 2, 3, 5, 10]:
            features[f'return_{lag}d'] = df['Close'].pct_change(lag)
        
        # ATR normalizado
        atr = self._calculate_atr(df, 14)
        features['atr_norm'] = atr / df['Close']
        
        # RSI em diferentes períodos
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = self._calculate_rsi(df['Close'], period)
        
        # MACD
        macd, macd_signal, macd_hist = self._calculate_macd(df['Close'])
        features['macd'] = macd / df['Close']
        features['macd_signal'] = macd_signal / df['Close']
        features['macd_hist'] = macd_hist / df['Close']
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger(df['Close'], 20, 2)
        features['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # ADX
        features['adx'] = self._calculate_adx(df, 14)
        
        # Volatilidade
        features['volatility_10'] = df['Close'].pct_change().rolling(10).std()
        features['volatility_20'] = df['Close'].pct_change().rolling(20).std()
        
        # Slope (tendência)
        features['slope_5'] = self._calculate_slope(df['Close'], 5)
        features['slope_10'] = self._calculate_slope(df['Close'], 10)
        
        # Distância de máximas/mínimas
        features['dist_from_high_20'] = (df['Close'] / df['High'].rolling(20).max()) - 1
        features['dist_from_low_20'] = (df['Close'] / df['Low'].rolling(20).min()) - 1
        
        # Volume features (se disponível)
        if 'Volume' in df.columns:
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            features['price_volume'] = df['Close'].pct_change() * np.log(df['Volume'])
        
        # Remover NaNs
        features = features.dropna()
        
        return features
    
    def create_target(self, df: pd.DataFrame, horizon: int = 5, 
                     threshold: float = 0.01) -> pd.Series:
        """
        Cria target para classificação binária
        
        Args:
            df: DataFrame com dados
            horizon: Períodos à frente para avaliar
            threshold: Threshold para classificar como alta (1% = 0.01)
            
        Returns:
            Série com targets (0 ou 1)
        """
        future_return = df['Close'].pct_change(horizon).shift(-horizon)
        target = (future_return > threshold).astype(int)
        
        return target.dropna()
    
    def train_model(self, df: pd.DataFrame, model_type: str = 'RandomForest',
                   horizon: int = 5, threshold: float = 0.01,
                   test_size: float = 0.2) -> Dict:
        """
        Treina modelo ML com validação temporal
        
        Args:
            df: DataFrame com dados OHLCV
            model_type: Tipo do modelo ('LogisticRegression', 'RandomForest', 'GradientBoosting')
            horizon: Períodos à frente para previsão
            threshold: Threshold para target
            test_size: Proporção para teste
            
        Returns:
            Dict com resultados da validação
        """
        # Criar features e target
        features = self.create_features(df)
        target = self.create_target(df, horizon, threshold)
        
        # Alinhar features e target
        common_index = features.index.intersection(target.index)
        X = features.loc[common_index]
        y = target.loc[common_index]
        
        if len(X) < 100:
            raise ValueError("Dados insuficientes para treinamento (mínimo 100 amostras)")
        
        self.feature_names = X.columns.tolist()
        
        # Split temporal
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # Normalização
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Modelo
        if model_type == 'LogisticRegression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'GradientBoosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Modelo não suportado: {model_type}")
        
        # Treinar
        self.model.fit(X_train_scaled, y_train)
        
        # Previsões
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Métricas
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = 0.5
        
        accuracy = (y_pred == y_test).mean()
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            feature_importance = dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
        else:
            feature_importance = {}
        
        self.validation_results = {
            'auc_score': auc_score,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance,
            'test_predictions': y_pred_proba
        }
        
        self.is_trained = True
        
        return self.validation_results
    
    def generate_signals(self, df: pd.DataFrame, 
                        buy_threshold: float = 0.6,
                        sell_threshold: float = 0.4,
                        atr_stop_mult: float = 2.0,
                        target_r_mult: float = 2.0) -> pd.DataFrame:
        """
        Gera sinais baseados no modelo treinado
        
        Args:
            df: DataFrame com dados
            buy_threshold: Threshold para sinal de compra
            sell_threshold: Threshold para sinal de venda
            atr_stop_mult: Multiplicador ATR para stop
            target_r_mult: Múltiplo R para target
            
        Returns:
            DataFrame com sinais
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")
        
        # Criar features
        features = self.create_features(df)
        
        # Normalizar
        X_scaled = self.scaler.transform(features[self.feature_names])
        
        # Previsões
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Criar DataFrame de sinais
        signals = pd.DataFrame(index=features.index)
        signals['probability'] = probabilities
        signals['signal'] = 0
        
        # Gerar sinais
        signals.loc[probabilities >= buy_threshold, 'signal'] = 1
        signals.loc[probabilities <= sell_threshold, 'signal'] = -1
        
        # Calcular ATR para stops
        atr = self._calculate_atr(df.loc[signals.index], 14)
        
        # Stops e targets
        signals['stop'] = np.nan
        signals['target'] = np.nan
        
        buy_mask = signals['signal'] == 1
        sell_mask = signals['signal'] == -1
        
        if buy_mask.any():
            signals.loc[buy_mask, 'stop'] = df.loc[buy_mask.index, 'Close'] - (atr.loc[buy_mask.index] * atr_stop_mult)
            signals.loc[buy_mask, 'target'] = df.loc[buy_mask.index, 'Close'] + (atr.loc[buy_mask.index] * atr_stop_mult * target_r_mult)
        
        if sell_mask.any():
            signals.loc[sell_mask, 'stop'] = df.loc[sell_mask.index, 'Close'] + (atr.loc[sell_mask.index] * atr_stop_mult)
            signals.loc[sell_mask, 'target'] = df.loc[sell_mask.index, 'Close'] - (atr.loc[sell_mask.index] * atr_stop_mult * target_r_mult)
        
        return signals[['signal', 'stop', 'target', 'probability']].fillna(0)
    
    # Métodos auxiliares para indicadores técnicos
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calcula ATR"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger(self, prices: pd.Series, period: int, 
                           std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, lower, middle
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calcula ADX (simplificado)"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = self._calculate_atr(df, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_slope(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcula slope (inclinação) dos preços"""
        def slope(y):
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0] if len(y) == period else np.nan
        
        return prices.rolling(period).apply(slope, raw=False)

# Parâmetros padrão para ML
ML_DEFAULT_PARAMS = {
    'model_type': 'RandomForest',
    'horizon': 5,
    'threshold': 0.01,
    'buy_threshold': 0.6,
    'sell_threshold': 0.4,
    'atr_stop_mult': 2.0,
    'target_r_mult': 2.0
}

