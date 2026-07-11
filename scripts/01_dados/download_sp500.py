"""Etapa 1 — Aquisição do histórico OHLCV do S&P 500 (^GSPC) via yfinance (2005-2025).
"""
import yfinance as yf
import pandas as pd

def extrair_dados():
    ticker = "^GSPC" # S&P 500
    nome_ficheiro = "data/SP500_Bruto.csv"
    
    print(f"A ligar ao Yahoo Finance para extrair dados do {ticker}...")
    
    # Download de 20 anos de dados
    dados = yf.download(ticker, start="2005-01-01", end="2026-01-01", auto_adjust=False)
    
    # Limpeza de colunas do novo yfinance
    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = dados.columns.droplevel(1)
        
    # Manter apenas as colunas essenciais
    dados = dados[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Guardar no disco
    dados.to_csv(nome_ficheiro)
    print(f"SUCESSO! O ficheiro '{nome_ficheiro}' foi criado com {len(dados)} dias de histórico.")

if __name__ == "__main__":
    extrair_dados()