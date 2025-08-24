"""
Cálculo de métricas de performance para estratégias de trading
"""
"""
Cálculo de métricas de performance para estratégias de trading
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple  # Adicione Tuple aqui
from datetime import datetime

# ... resto do código permanece igual


def calculate_performance_metrics(trades: List, equity_curve: pd.Series, 
                                initial_capital: float) -> Dict[str, Any]:
    """
    Calcula métricas completas de performance
    
    Args:
        trades: Lista de trades executados
        equity_curve: Série temporal do capital
        initial_capital: Capital inicial
        
    Returns:
        Dict com todas as métricas calculadas
    """
    if not trades or equity_curve.empty:
        return _empty_metrics()
    
    # Métricas básicas
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.net_pnl > 0]
    losing_trades = [t for t in trades if t.net_pnl <= 0]
    
    # Retornos
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    
    # CAGR
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    years = (end_date - start_date).days / 365.25
    cagr = (equity_curve.iloc[-1] / initial_capital) ** (1/years) - 1 if years > 0 else 0
    
    # Drawdown
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = abs(drawdown.min())
    
    # Retornos diários para Sharpe/Sortino
    daily_returns = equity_curve.pct_change().dropna()
    
    # Sharpe Ratio
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
    else:
        sharpe = 0
    
    # Sortino Ratio
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) > 1 and negative_returns.std() > 0:
        sortino = (daily_returns.mean() * 252) / (negative_returns.std() * np.sqrt(252))
    else:
        sortino = 0
    
    # Calmar Ratio
    calmar = cagr / max_drawdown if max_drawdown > 0 else 0
    
    # Hit Rate
    hit_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    # Profit Factor
    gross_profit = sum(t.net_pnl for t in winning_trades)
    gross_loss = abs(sum(t.net_pnl for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Expectancy
    avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0
    expectancy = (hit_rate * avg_win) + ((1 - hit_rate) * avg_loss)
    
    # R médio
    valid_r = [t.r_multiple for t in trades if not np.isnan(t.r_multiple)]
    avg_r = np.mean(valid_r) if valid_r else 0
    
    # Exposição (tempo em posição)
    total_bars = len(equity_curve)
    bars_in_position = sum(t.bars_held for t in trades)
    exposure = bars_in_position / total_bars if total_bars > 0 else 0
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar,
        'Hit Rate': hit_rate,
        'Profit Factor': profit_factor,
        'Expectancy': expectancy,
        'Average R': avg_r,
        'Num Trades': total_trades,
        'Exposure': exposure,
        'Equity Curve': equity_curve,
        'Drawdown Series': drawdown
    }

def _empty_metrics() -> Dict[str, Any]:
    """Retorna métricas vazias para casos sem trades"""
    return {
        'Total Return': 0.0,
        'CAGR': 0.0,
        'Sharpe Ratio': 0.0,
        'Sortino Ratio': 0.0,
        'Max Drawdown': 0.0,
        'Calmar Ratio': 0.0,
        'Hit Rate': 0.0,
        'Profit Factor': 0.0,
        'Expectancy': 0.0,
        'Average R': 0.0,
        'Num Trades': 0,
        'Exposure': 0.0,
        'Equity Curve': pd.Series(),
        'Drawdown Series': pd.Series()
    }

def rank_strategies(strategy_metrics: Dict[str, Dict], 
                   criterion: str = 'Calmar Ratio') -> List[Tuple[str, float]]:
    """
    Rankeia estratégias por critério específico
    
    Args:
        strategy_metrics: Dict com métricas por estratégia
        criterion: Critério de ranking
        
    Returns:
        Lista de tuplas (nome_estratégia, valor_métrica) ordenada
    """
    rankings = []
    
    for name, metrics in strategy_metrics.items():
        if metrics and criterion in metrics:
            value = metrics[criterion]
            # Tratar valores infinitos
            if np.isinf(value):
                value = 999 if value > 0 else -999
            rankings.append((name, value))
    
    # Ordenar (maior é melhor, exceto para Max Drawdown)
    reverse = criterion != 'Max Drawdown'
    rankings.sort(key=lambda x: x[1], reverse=reverse)
    
    return rankings

def create_metrics_table(strategy_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """Cria tabela formatada com métricas de todas as estratégias"""
    
    if not strategy_metrics:
        return pd.DataFrame()
    
    rows = []
    for name, metrics in strategy_metrics.items():
        if metrics:
            row = {
                'Estratégia': name,
                'Retorno Total': f"{metrics['Total Return']:.2%}",
                'CAGR': f"{metrics['CAGR']:.2%}",
                'Sharpe': f"{metrics['Sharpe Ratio']:.2f}",
                'Sortino': f"{metrics['Sortino Ratio']:.2f}",
                'Max DD': f"{metrics['Max Drawdown']:.2%}",
                'Calmar': f"{metrics['Calmar Ratio']:.2f}",
                'Hit Rate': f"{metrics['Hit Rate']:.2%}",
                'Profit Factor': f"{metrics['Profit Factor']:.2f}",
                'Expectancy': f"{metrics['Expectancy']:.2f}",
                'Avg R': f"{metrics['Average R']:.2f}",
                'Trades': int(metrics['Num Trades']),
                'Exposição': f"{metrics['Exposure']:.2%}"
            }
            rows.append(row)
    
    return pd.DataFrame(rows)

