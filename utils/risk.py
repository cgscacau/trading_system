"""
Gestão de risco e cálculo de posição BRL→USD
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class PositionSizing:
    """Resultado do cálculo de posição"""
    capital_brl: float
    capital_usd: float
    risk_brl: float
    risk_usd: float
    position_size: float
    stop_distance: float
    exposure_brl: float
    exposure_usd: float
    r_ratio: float
    leverage: float
    fees_total: float

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcula Average True Range
    
    Args:
        df: DataFrame com OHLC
        period: Período para cálculo
        
    Returns:
        Série com valores ATR
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    
    return atr

def calculate_position_size(
    capital_brl: float,
    risk_percent: float,
    fx_rate: float,
    asset_price: float,
    stop_distance: float,
    fee_percent: float = 0.001,
    slippage_percent: float = 0.0005,
    asset_currency: str = 'USD',
    min_lot_size: float = 1.0
) -> PositionSizing:
    """
    Calcula tamanho de posição com conversão BRL→USD
    
    Args:
        capital_brl: Capital disponível em BRL
        risk_percent: Percentual de risco por trade (0.01 = 1%)
        fx_rate: Taxa USDBRL (quantos BRL por 1 USD)
        asset_price: Preço atual do ativo
        stop_distance: Distância do stop em preço
        fee_percent: Taxa por lado (0.001 = 0.1%)
        slippage_percent: Slippage por lado (0.0005 = 0.05%)
        asset_currency: Moeda do ativo ('USD' ou 'BRL')
        min_lot_size: Tamanho mínimo do lote
        
    Returns:
        PositionSizing com todos os cálculos
    """
    # Capital em USD
    capital_usd = capital_brl / fx_rate if asset_currency == 'USD' else capital_brl
    
    # Risco monetário
    risk_brl = capital_brl * risk_percent
    risk_usd = risk_brl / fx_rate if asset_currency == 'USD' else risk_brl
    
    # Custos totais por unidade (entrada + saída)
    cost_per_unit = asset_price * (fee_percent + slippage_percent) * 2
    
    # Risco efetivo por unidade (stop + custos)
    effective_risk_per_unit = stop_distance + cost_per_unit
    
    # Tamanho da posição
    if effective_risk_per_unit > 0:
        raw_position_size = risk_usd / effective_risk_per_unit
        position_size = max(0, np.floor(raw_position_size / min_lot_size) * min_lot_size)
    else:
        position_size = 0
    
    # Exposição
    exposure_usd = position_size * asset_price
    exposure_brl = exposure_usd * fx_rate if asset_currency == 'USD' else exposure_usd
    
    # R:R ratio (assumindo 2R como target padrão)
    target_distance = stop_distance * 2  # 2R
    r_ratio = target_distance / stop_distance if stop_distance > 0 else 0
    
    # Alavancagem
    leverage = exposure_brl / capital_brl if capital_brl > 0 else 0
    
    # Fees totais estimados
    fees_total = position_size * cost_per_unit
    
    return PositionSizing(
        capital_brl=capital_brl,
        capital_usd=capital_usd,
        risk_brl=risk_brl,
        risk_usd=risk_usd,
        position_size=position_size,
        stop_distance=stop_distance,
        exposure_brl=exposure_brl,
        exposure_usd=exposure_usd,
        r_ratio=r_ratio,
        leverage=leverage,
        fees_total=fees_total
    )

def validate_risk_parameters(capital: float, risk_percent: float, 
                           fx_rate: float, price: float) -> Dict[str, str]:
    """
    Valida parâmetros de risco
    
    Returns:
        Dict com erros encontrados (vazio se tudo OK)
    """
    errors = {}
    
    if capital <= 0:
        errors['capital'] = "Capital deve ser maior que zero"
    
    if not (0 < risk_percent <= 0.1):  # Max 10%
        errors['risk'] = "Risco deve estar entre 0.1% e 10%"
    
    if fx_rate <= 0:
        errors['fx_rate'] = "Taxa de câmbio deve ser maior que zero"
    
    if price <= 0:
        errors['price'] = "Preço do ativo deve ser maior que zero"
    
    return errors

def get_risk_checklist(df: pd.DataFrame, current_price: float, 
                      atr_value: float) -> Dict[str, str]:
    """
    Gera checklist rápido de risco
    
    Returns:
        Dict com itens do checklist
    """
    checklist = {}
    
    # Tendência
    sma_200 = df['Close'].rolling(200).mean().iloc[-1]
    if current_price > sma_200:
        checklist['Tendência'] = f"Alta (preço {((current_price/sma_200-1)*100):.1f}% acima SMA200)"
    else:
        checklist['Tendência'] = f"Baixa (preço {((1-current_price/sma_200)*100):.1f}% abaixo SMA200)"
    
    # Volatilidade
    atr_avg = df['Close'].rolling(50).apply(lambda x: calculate_atr(df.loc[x.index]).iloc[-1]).mean()
    if atr_value > atr_avg * 1.2:
        checklist['Volatilidade'] = "Alta (ATR acima da média)"
    elif atr_value < atr_avg * 0.8:
        checklist['Volatilidade'] = "Baixa (ATR abaixo da média)"
    else:
        checklist['Volatilidade'] = "Normal"
    
    # Volume (se disponível)
    if 'Volume' in df.columns:
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        vol_current = df['Volume'].iloc[-1]
        if vol_current > vol_avg * 1.5:
            checklist['Volume'] = "Alto (acima da média)"
        elif vol_current < vol_avg * 0.5:
            checklist['Volume'] = "Baixo (abaixo da média)"
        else:
            checklist['Volume'] = "Normal"
    
    return checklist

