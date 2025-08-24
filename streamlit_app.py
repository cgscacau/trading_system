# streamlit_app_debug.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Sistema de Trading - Debug",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Sistema de Trading - Modo Debug")

# Teste básico de coleta de dados
st.sidebar.header("Teste Rápido")

ticker = st.sidebar.selectbox(
    "Selecione um ativo para teste:",
    ['PETR4.SA', 'VALE3.SA', 'AAPL', 'BTC-USD']
)

if st.sidebar.button("🧪 Testar Coleta de Dados"):
    with st.spinner("Testando coleta..."):
        try:
            # Testa coleta básica
            data = yf.download(ticker, start='2023-01-01', end='2024-01-01')
            
            if not data.empty:
                st.success(f"✅ Dados coletados: {len(data)} registros")
                
                # Mostra estrutura
                st.subheader("📊 Estrutura dos Dados")
                st.dataframe(data.head())
                
                # Testa cálculos básicos
                data['returns'] = data['Close'].pct_change()
                data['SMA_20'] = data['Close'].rolling(20).mean()
                
                st.subheader("📈 Gráfico de Preços")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Preço'
                ))
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    mode='lines',
                    name='SMA 20'
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retorno Total", f"{((data['Close'][-1]/data['Close'][0])-1)*100:.1f}%")
                with col2:
                    st.metric("Volatilidade", f"{data['returns'].std()*np.sqrt(252)*100:.1f}%")
                with col3:
                    st.metric("Registros", len(data))
                    
            else:
                st.error("❌ Nenhum dado encontrado!")
                
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")

# Instruções
st.markdown("""
---
## 📋 Próximos Passos

**1. Se o teste acima funcionou:**
- Substitua o arquivo `src/backtest/backtester.py` pelo código fornecido
- Atualize os modelos ML conforme mostrado
- Execute o sistema completo

**2. Se ainda há erros:**
- Verifique se todas as pastas `src/` têm arquivos `__init__.py`
- Confirme que todas as dependências estão instaladas
- Execute este teste primeiro para validar a coleta de dados

**3. Estrutura necessária:**
