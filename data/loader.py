import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional # <-- ESTA É A LINHA ADICIONADA/CORRIGIDA

@st.cache_data(ttl=600, show_spinner="Carregando dados do ativo...")
def load_price_data(ticker, start, end, interval):
    """
    Carrega dados de preços históricos de um ativo usando o yfinance.
    Inclui tratamento de erros robusto.
    """
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, repair=True)
        
        if df.empty:
            st.warning(f"Nenhum dado encontrado para '{ticker}' no período selecionado. Verifique o ticker e as datas.")
            return pd.DataFrame()

        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            st.error(f"Os dados baixados para '{ticker}' estão incompletos.")
            return pd.DataFrame()
            
        return df

    except Exception as e:
        st.error(f"Falha ao baixar os dados de '{ticker}'. Erro: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_usd_brl_rate() -> Tuple[Optional[float], Optional[datetime]]:
    """
    Obtém a cotação USD/BRL atual via yfinance
    """
    try:
        ticker = yf.Ticker("BRL=X")
        # Pega dados do último dia no intervalo de 15 minutos para ter um valor recente
        data = ticker.history(period='1d', interval='15m')
        if not data.empty:
            last_price = data['Close'].iloc[-1]
            last_time = data.index[-1].to_pydatetime()
            return float(last_price), last_time
    except Exception:
        pass # Falha silenciosamente se não conseguir obter a cotação
    return None, None

@st.cache_data(ttl=3600, show_spinner="Buscando informações do ativo...")
def get_asset_info(ticker):
    """
    Obtém informações detalhadas de um ativo.
    """
    try:
        asset = yf.Ticker(ticker)
        info = asset.info
        
        # Retorna um dicionário com as informações mais relevantes
        return {
            "name": info.get('longName', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "currency": info.get('currency', 'N/A')
        }
    except Exception:
        return {
            "name": "Não encontrado",
            "sector": "N/A",
            "currency": "N/A"
        }
