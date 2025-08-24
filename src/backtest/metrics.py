# src/backtest/metrics.py
import numpy as np
import pandas as pd

def calculate_performance_metrics(equity_curve, trades_df):
    """Calcula métricas de performance"""
    returns = equity_curve['Equity'].pct_change().dropna()
    
    # Métricas básicas
    total_return = (equity_curve['Equity'].iloc[-1] / equity_curve['Equity'].iloc[0]) - 1
    cagr = (1 + total_return) ** (252 / len(equity_curve)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (returns.mean() * 252) / (volatility + 1e-10)
    
    # Drawdown
    running_max = equity_curve['Equity'].cummax()
    drawdown = (equity_curve['Equity'] - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Métricas de trades
    if not trades_df.empty:
        win_rate = (trades_df['PnL'] > 0).mean()
        avg_win = trades_df[trades_df['PnL'] > 0]['PnL'].mean()
        avg_loss = trades_df[trades_df['PnL'] < 0]['PnL'].mean()
        profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf
    else:
        win_rate = 0
        profit_factor = 0
    
    return {
        'Total Return': f"{total_return:.2%}",
        'CAGR': f"{cagr:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Win Rate': f"{win_rate:.2%}",
        'Profit Factor': f"{profit_factor:.2f}",
        'Total Trades': len(trades_df)
    }
