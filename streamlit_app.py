# streamlit_app_minimal.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Trading Test", page_icon="📈", layout="wide")
st.title("📈 Sistema de Trading - Teste")

ticker = st.sidebar.selectbox("Ativo:", ['PETR4.SA', 'AAPL', 'BTC-USD'])
start_date = st.sidebar.date_input("Início:", pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input("Fim:", pd.to_datetime('2024-01-01'))

if st.sidebar.button("📊 Analisar"):
    with st.spinner("Coletando dados..."):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if not data.empty:
                st.success(f"✅ {len(data)} registros coletados")
                
                # Métricas
                total_return = (data['Close'][-1] / data['Close'][0] - 1) * 100
                volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Retorno Total", f"{total_return:.1f}%")
                with col2:
                    st.metric("Volatilidade", f"{volatility:.1f}%")
                
                # Gráfico
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Preço'))
                fig.update_layout(title=f'{ticker}', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("❌ Nenhum dado encontrado!")
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
