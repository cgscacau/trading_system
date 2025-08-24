"""
Módulo para carregamento e cache de dados financeiros via yfinance
"""
import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

@st.cache_data(ttl=600, show_spinner=False)  # Cache por 10 minutos
def load_price_data(ticker: str, start: str, end: str, interval: str = '1d') -> pd.DataFrame:
    """
    Carrega dados OHLCV do yfinance com cache
    
    Args:
        ticker: Símbolo do ativo
        start: Data inicial (YYYY-MM-DD)
        end: Data final (YYYY-MM-DD)
        interval: Intervalo dos dados
    
    Returns:
        DataFrame com dados OHLCV
    """
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        
        if data.empty:
            st.warning(f"Nenhum dado encontrado para {ticker}")
            return pd.DataFrame()
        
        # Padronizar colunas
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Remover dados inválidos
        data = data.dropna()
        data = data[data['Volume'] > 0]
        
        return data
        
    except Exception as e:
        st.error(f"Erro ao carregar dados de {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)  # Cache por 5 minutos
def get_usd_brl_rate() -> Tuple[Optional[float], Optional[datetime]]:
    """
    Obtém a cotação USD/BRL atual via yfinance
    
    Returns:
        Tuple com (cotação, timestamp) ou (None, None) se falhar
    """
    try:
        usdbrl = yf.Ticker("USDBRL=X")
        hist = usdbrl.history(period="1d")
        
        if not hist.empty:
            rate = float(hist['Close'].iloc[-1])
            timestamp = hist.index[-1].to_pydatetime()
            return rate, timestamp
        
        return None, None
        
    except Exception:
        return None, None

def validate_ticker(ticker: str) -> bool:
    """Valida se o ticker existe"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return bool(info.get('symbol') or info.get('shortName'))
    except:
        return False

def get_asset_info(ticker: str) -> dict:
    """Obtém informações básicas do ativo"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'name': info.get('shortName', ticker),
            'currency': info.get('currency', 'USD'),
            'sector': info.get('sector', 'Unknown'),
            'market': info.get('market', 'Unknown')
        }
    except:
        return {
            'name': ticker,
            'currency': 'USD', 
            'sector': 'Unknown',
            'market': 'Unknown'
        }

