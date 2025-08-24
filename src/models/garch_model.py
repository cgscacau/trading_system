# src/models/garch_model.py
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

class GARCHModel:
    def __init__(self, config):
        self.p = config['models']['garch']['p']
        self.q = config['models']['garch']['q']
        self.dist = config['models']['garch']['dist']
        self.fitted_model = None
        
    def fit(self, returns_series):
        """Treina modelo GARCH para volatilidade"""
        try:
            # Converte para percentual e limpa outliers
            returns_pct = returns_series * 100
            returns_clean = returns_pct[
                (returns_pct > returns_pct.quantile(0.01)) & 
                (returns_pct < returns_pct.quantile(0.99))
            ].dropna()
            
            if len(returns_clean) < 100:
                return False
            
            model = arch_model(
                returns_clean,
                vol='Garch',
                p=self.p,
                q=self.q,
                dist=self.dist,
                mean='Zero'
            )
            
            self.fitted_model = model.fit(disp='off', show_warning=False)
            return True
            
        except Exception as e:
            print(f"Erro no treinamento GARCH: {e}")
            return False
    
    def forecast_volatility(self, horizon=1):
        """Prediz volatilidade futura"""
        if self.fitted_model is None:
            return 0.02
            
        try:
            forecast = self.fitted_model.forecast(horizon=horizon, reindex=False)
            predicted_var = float(forecast.variance.values[-1, -1])
            predicted_vol = np.sqrt(max(predicted_var, 1e-8)) / 100
            
            return float(np.clip(predicted_vol, 0.005, 0.1))
            
        except Exception as e:
            print(f"Erro na previsão GARCH: {e}")
            return 0.02
