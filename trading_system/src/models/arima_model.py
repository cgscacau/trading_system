# src/models/arima_model.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    def __init__(self, config):
        self.order = tuple(config['models']['arima']['order'])
        self.model = None
        self.fitted_model = None
        
    def fit(self, returns_series):
        """Treina modelo ARIMA com tratamento robusto"""
        try:
            # Remove outliers usando IQR
            Q1 = returns_series.quantile(0.25)
            Q3 = returns_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            clean_returns = returns_series[
                (returns_series >= lower_bound) & 
                (returns_series <= upper_bound)
            ].dropna()
            
            if len(clean_returns) < 50:
                print("Dados insuficientes para ARIMA")
                return False
            
            self.model = ARIMA(clean_returns, order=self.order)
            self.fitted_model = self.model.fit(method_kwargs={"warn_convergence": False})
            return True
            
        except Exception as e:
            print(f"Erro no treinamento ARIMA: {e}")
            return False
    
    def forecast_probability(self, steps=1):
        """Calcula probabilidade de alta usando intervalo de confiança"""
        if self.fitted_model is None:
            return 0.0, 0.5
            
        try:
            # Previsão pontual
            forecast = self.fitted_model.forecast(steps=steps)
            predicted_return = float(forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0])
            
            # Intervalo de confiança
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            conf_int = forecast_result.conf_int()
            
            # Desvio padrão da previsão
            forecast_std = (conf_int.iloc[0, 1] - conf_int.iloc[0, 0]) / (2 * 1.96)
            
            if forecast_std > 0:
                # Probabilidade usando distribuição normal
                prob_positive = 1 - stats.norm.cdf(0, loc=predicted_return, scale=forecast_std)
            else:
                prob_positive = 0.5 if predicted_return >= 0 else 0.4
                
            return float(predicted_return), float(np.clip(prob_positive, 0.1, 0.9))
            
        except Exception as e:
            print(f"Erro na previsão ARIMA: {e}")
            return 0.0, 0.5
