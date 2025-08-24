# src/backtest/plotting.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class TradingDashboard:
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c', 
            'danger': '#d62728',
            'warning': '#ff7f0e'
        }
    
    def create_equity_curve(self, equity_data):
        """Cria gráfico da curva de equity"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_data['Date'],
            y=equity_data['Equity'],
            mode='lines',
            name='Equity',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        fig.update_layout(
            title='Curva de Equity',
            xaxis_title='Data',
            yaxis_title='Valor da Carteira ($)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_drawdown_chart(self, equity_data):
        """Cria gráfico de drawdown"""
        running_max = equity_data['Equity'].cummax()
        drawdown = (equity_data['Equity'] - running_max) / running_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_data['Date'],
            y=drawdown,
            mode='lines',
            fill='tonexty',
            name='Drawdown',
            line=dict(color=self.colors['danger']),
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(self.colors['danger'])) + [0.3])}"
        ))
        
        fig.update_layout(
            title='Drawdown',
            xaxis_title='Data',
            yaxis_title='Drawdown (%)',
            template='plotly_white'
        )
        
        return fig
    
    def create_performance_dashboard(self, results, metrics):
        """Cria dashboard completo"""
        # Layout com subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Equity Curve', 'Drawdown', 'Monthly Returns', 
                          'Trade Distribution', 'Model Predictions', 'Risk Metrics'],
            specs=[[{"colspan": 2}, None],
                   [{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Equity curve
        equity_data = results['combined_equity']
        fig.add_trace(
            go.Scatter(x=equity_data['Date'], y=equity_data['Equity'],
                      mode='lines', name='Equity'),
            row=1, col=1
        )
        
        # Adiciona mais gráficos...
        
        fig.update_layout(
            height=1200,
            title_text="Dashboard de Performance",
            template='plotly_white'
        )
        
        return fig
