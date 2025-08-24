# src/data/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, config):
        self.config = config
    
    def download_data(self, tickers, start, end, interval="1d"):
        """Coleta dados do Yahoo Finance com tratamento de erros"""
        try:
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                threads=True,
                group_by='ticker'
            )
            
            datasets = {}
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        df = data.copy()
                    else:
                        df = data[ticker].copy()
                    
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                    df['Symbol'] = ticker
                    df.reset_index(inplace=True)
                    datasets[ticker] = df
                    print(f"✓ {ticker}: {len(df)} registros coletados")
                    
                except Exception as e:
                    print(f"✗ Erro ao processar {ticker}: {e}")
                    continue
                    
            return datasets
            
        except Exception as e:
            print(f"Erro na coleta de dados: {e}")
            return {}
