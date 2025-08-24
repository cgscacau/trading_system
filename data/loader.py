"""
Módulo para carregamento e cache de dados financeiros via yfinance
"""
import yfinance as yf
import streamlit as st
from datetime import datetime

@st.cache_data(ttl=600) # Adiciona cache para evitar downloads repetidos
def load_price_data(ticker, start, end, interval):
    """
    Carrega dados de preços históricos de um ativo usando o yfinance.
    Inclui tratamento de erros robusto.
    """
    try:
        # A yfinance às vezes falha na primeira tentativa, adicionamos 'repair=True'
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, repair=True)
        
        if df.empty:
            st.error(f"Nenhum dado encontrado para '{ticker}' no período selecionado. O ativo pode não ser negociado nesse intervalo ou o ticker está incorreto.")
            return pd.DataFrame()

        # Verifica se há colunas essenciais
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            st.error(f"Os dados baixados para '{ticker}' estão incompletos. Faltam colunas essenciais.")
            return pd.DataFrame()
            
        return df

    except Exception as e:
        st.error(f"Ocorreu um erro ao tentar baixar os dados de '{ticker}'.")
        st.error(f"Detalhe do erro: {e}")
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

