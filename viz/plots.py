"""
Visualizações interativas com Plotly para o Lab de Estratégias
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

def plot_candlestick_with_trades(df: pd.DataFrame, trades: List, 
                                title: str = "Gráfico de Candles com Trades") -> go.Figure:
    """
    Cria gráfico de candlestick com marcações de trades
    
    Args:
        df: DataFrame com dados OHLCV
        trades: Lista de objetos Trade
        title: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Preço',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    if trades:
        # Separar trades por tipo
        long_trades = [t for t in trades if t.side == 'long']
        short_trades = [t for t in trades if t.side == 'short']
        
        # Entradas long
        if long_trades:
            fig.add_trace(go.Scatter(
                x=[t.entry_date for t in long_trades],
                y=[t.entry_price for t in long_trades],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(color='darkgreen', width=1)
                ),
                name='Entrada Long',
                hovertemplate='<b>Entrada Long</b><br>' +
                             'Data: %{x}<br>' +
                             'Preço: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
            
            # Saídas long
            fig.add_trace(go.Scatter(
                x=[t.exit_date for t in long_trades],
                y=[t.exit_price for t in long_trades],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='lightgreen'
                ),
                name='Saída Long',
                hovertemplate='<b>Saída Long</b><br>' +
                             'Data: %{x}<br>' +
                             'Preço: %{y:.2f}<br>' +
                             'PnL: %{customdata:.2f}<br>' +
                             '<extra></extra>',
                customdata=[t.net_pnl for t in long_trades]
            ))
        
        # Entradas short
        if short_trades:
            fig.add_trace(go.Scatter(
                x=[t.entry_date for t in short_trades],
                y=[t.entry_price for t in short_trades],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(color='darkred', width=1)
                ),
                name='Entrada Short',
                hovertemplate='<b>Entrada Short</b><br>' +
                             'Data: %{x}<br>' +
                             'Preço: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
            
            # Saídas short
            fig.add_trace(go.Scatter(
                x=[t.exit_date for t in short_trades],
                y=[t.exit_price for t in short_trades],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='lightcoral'
                ),
                name='Saída Short',
                hovertemplate='<b>Saída Short</b><br>' +
                             'Data: %{x}<br>' +
                             'Preço: %{y:.2f}<br>' +
                             'PnL: %{customdata:.2f}<br>' +
                             '<extra></extra>',
                customdata=[t.net_pnl for t in short_trades]
            ))
        
        # Linhas de stop e target
        for trade in trades[:20]:  # Limitar para não poluir o gráfico
            # Stop Loss
            if not np.isnan(trade.stop_price):
                fig.add_shape(
                    type="line",
                    x0=trade.entry_date,
                    y0=trade.stop_price,
                    x1=trade.exit_date,
                    y1=trade.stop_price,
                    line=dict(color="red", width=1, dash="dot"),
                    opacity=0.7
                )
            
            # Take Profit
            if not np.isnan(trade.target_price):
                fig.add_shape(
                    type="line",
                    x0=trade.entry_date,
                    y0=trade.target_price,
                    x1=trade.exit_date,
                    y1=trade.target_price,
                    line=dict(color="green", width=1, dash="dot"),
                    opacity=0.7
                )
    
    fig.update_layout(
        title=title,
        xaxis_title="Data",
        yaxis_title="Preço",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
        hovermode="x unified"
    )
    
    return fig

def plot_equity_curve(equity_curve: pd.Series, drawdown: pd.Series,
                     title: str = "Curva de Capital e Drawdown") -> go.Figure:
    """
    Plota curva de capital e drawdown
    
    Args:
        equity_curve: Série com evolução do capital
        drawdown: Série com drawdown
        title: Título do gráfico
        
    Returns:
        Figura Plotly com subplots
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Evolução do Capital', 'Drawdown'),
        vertical_spacing=0.1
    )
    
    # Curva de capital
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Capital',
            line=dict(color='blue', width=2),
            hovertemplate='Data: %{x}<br>Capital: %{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,  # Converter para percentual
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            hovertemplate='Data: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Data", row=2, col=1)
    fig.update_yaxes(title_text="Capital (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    return fig

def plot_strategy_comparison(metrics_dict: Dict[str, Dict]) -> go.Figure:
    """
    Compara métricas de múltiplas estratégias
    
    Args:
        metrics_dict: Dict com métricas por estratégia
        
    Returns:
        Figura Plotly com radar chart
    """
    if not metrics_dict:
        return go.Figure()
    
    strategies = list(metrics_dict.keys())
    
    # Métricas para comparar (normalizadas)
    metrics_to_compare = ['CAGR', 'Sharpe Ratio', 'Hit Rate', 'Profit Factor']
    
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (strategy, metrics) in enumerate(metrics_dict.items()):
        if not metrics:
            continue
            
        # Normalizar métricas para o radar chart
        values = []
        for metric in metrics_to_compare:
            value = metrics.get(metric, 0)
            if metric == 'CAGR':
                values.append(max(0, min(100, value * 100)))  # 0-100%
            elif metric == 'Sharpe Ratio':
                values.append(max(0, min(5, value)))  # 0-5
            elif metric == 'Hit Rate':
                values.append(value * 100)  # 0-100%
            elif metric == 'Profit Factor':
                values.append(max(0, min(5, value)))  # 0-5
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Fechar o polígono
            theta=metrics_to_compare + [metrics_to_compare[0]],
            fill='toself',
            name=strategy,
            line_color=colors[i % len(colors)],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title="Comparação de Estratégias",
        template="plotly_white"
    )
    
    return fig

def plot_ml_feature_importance(feature_importance: Dict[str, float],
                              top_n: int = 15) -> go.Figure:
    """
    Plota importância das features do modelo ML
    
    Args:
        feature_importance: Dict com importância por feature
        top_n: Número de features mais importantes para mostrar
        
    Returns:
        Figura Plotly
    """
    if not feature_importance:
        return go.Figure()
    
    # Ordenar por importância
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: x[1], reverse=True)[:top_n]
    
    features, importance = zip(*sorted_features)
    
    fig = go.Figure(go.Bar(
        x=list(importance),
        y=list(features),
        orientation='h',
        marker_color='lightblue',
        text=[f'{imp:.3f}' for imp in importance],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Features Mais Importantes",
        xaxis_title="Importância",
        yaxis_title="Features",
        template="plotly_white",
        height=max(400, len(features) * 25)
    )
    
    return fig

def plot_roc_curve(fpr: List[float], tpr: List[float], 
                   auc_score: float) -> go.Figure:
    """
    Plota curva ROC
    
    Args:
        fpr: False Positive Rate
        tpr: True Positive Rate  
        auc_score: AUC score
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # Curva ROC
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Linha diagonal (classificador aleatório)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Curva ROC - Modelo ML',
        xaxis_title='Taxa de Falsos Positivos',
        yaxis_title='Taxa de Verdadeiros Positivos',
        template="plotly_white",
        width=500,
        height=500
    )
    
    return fig

def plot_monthly_returns(equity_curve: pd.Series) -> go.Figure:
    """
    Plota retornos mensais em heatmap
    
    Args:
        equity_curve: Série com evolução do capital
        
    Returns:
        Figura Plotly
    """
    if equity_curve.empty:
        return go.Figure()
    
    # Calcular retornos mensais
    monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
    
    # Criar matriz para heatmap
    years = monthly_returns.index.year.unique()
    months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
              'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    
    heatmap_data = []
    for year in years:
        year_data = []
        for month in range(1, 13):
            try:
                ret = monthly_returns[
                    (monthly_returns.index.year == year) & 
                    (monthly_returns.index.month == month)
                ].iloc[0] * 100
                year_data.append(ret)
            except:
                year_data.append(np.nan)
        heatmap_data.append(year_data)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=months,
        y=years,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f'{val:.1f}%' if not np.isnan(val) else '' 
               for val in row] for row in heatmap_data],
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Ano: %{y}<br>Mês: %{x}<br>Retorno: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Retornos Mensais (%)',
        template="plotly_white",
        height=max(300, len(years) * 50)
    )
    
    return fig

