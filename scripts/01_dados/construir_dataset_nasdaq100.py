"""Etapa 1-2 — Construção integral do dataset mestre do NASDAQ-100 (^NDX, 2005-2025): OHLCV, indicadores técnicos e macroeconomia.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import ta

# 1. EXTRAÇÃO DE DADOS NASDAQ-100
print("1. A gerar o dataset do NASDAQ-100...")
tickers_macro = {
    'VIX': '^VIX',              
    'Petroleo_WTI': 'CL=F',     
    'Ouro': 'GC=F',             
    'Dolar_Index': 'DX-Y.NYB',  
    'TBill_3M': '^IRX',         
    'Treasury_10Y': '^TNX'
}
data_inicio = '2005-01-01'
data_fim = '2026-01-01'

dados_macro = pd.DataFrame()
for nome_coluna, ticker in tickers_macro.items():
    temp = yf.download(ticker, start=data_inicio, end=data_fim, auto_adjust=False, progress=False)
    if isinstance(temp.columns, pd.MultiIndex): temp.columns = temp.columns.droplevel(1)
    dados_macro[nome_coluna] = temp['Close']

dados_macro['Spread_10Y_3M'] = dados_macro['Treasury_10Y'] - dados_macro['TBill_3M']

dados = yf.download('^NDX', start=data_inicio, end=data_fim, auto_adjust=False, progress=False)
if isinstance(dados.columns, pd.MultiIndex): dados.columns = dados.columns.droplevel(1)
dados = dados[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

dados['SMA_20'] = ta.trend.SMAIndicator(close=dados['Close'], window=20).sma_indicator()
dados['SMA_50'] = ta.trend.SMAIndicator(close=dados['Close'], window=50).sma_indicator()
dados['EMA_20'] = ta.trend.EMAIndicator(close=dados['Close'], window=20).ema_indicator()
dados['MACD'] = ta.trend.MACD(close=dados['Close']).macd()
dados['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=dados['Close'], volume=dados['Volume']).on_balance_volume()

dados['RSI'] = ta.momentum.RSIIndicator(close=dados['Close'], window=14).rsi()
dados['ROC_10'] = ta.momentum.ROCIndicator(close=dados['Close'], window=10).roc()
dados['ROC_20'] = ta.momentum.ROCIndicator(close=dados['Close'], window=20).roc()
dados['MOM_10'] = dados['Close'].diff(periods=10)
dados['Stoch_K'] = ta.momentum.StochasticOscillator(high=dados['High'], low=dados['Low'], close=dados['Close']).stoch()

bollinger = ta.volatility.BollingerBands(close=dados['Close'], window=20, window_dev=2)
dados['BB_High'] = bollinger.bollinger_hband()
dados['BB_Low'] = bollinger.bollinger_lband()
dados['ATR'] = ta.volatility.AverageTrueRange(high=dados['High'], low=dados['Low'], close=dados['Close']).average_true_range()

janelas = [5, 10, 14, 20, 50, 100, 200]
for j in janelas:
    dados[f'SMA_{j}'] = dados['Close'].rolling(window=j).mean()
    dados[f'EMA_{j}'] = dados['Close'].ewm(span=j, adjust=False).mean()
    if j <= 50: dados[f'Return_{j}d'] = dados['Close'].pct_change(j)
    dados[f'RSI_{j}'] = ta.momentum.RSIIndicator(dados['Close'], window=j).rsi()

dados = dados.join(dados_macro).ffill()

dados['Ouro_Ret'] = dados['Ouro'].pct_change()
dados['Petroleo_Ret'] = dados['Petroleo_WTI'].pct_change()
dados['Dolar_Ret'] = dados['Dolar_Index'].pct_change()
dados['Volume_Ret'] = dados['Volume'].pct_change()
dados['Dist_SMA_200'] = (dados['Close'] / dados['SMA_200']) - 1
dados['Treasury_10Y_Diff'] = dados['Treasury_10Y'].diff()
dados['TBill_3M_Diff'] = dados['TBill_3M'].diff()
dados['Spread_Diff'] = dados['Spread_10Y_3M'].diff()
dados['Vol_Zscore'] = (dados['Volume'] - dados['Volume'].rolling(20, min_periods=1).mean()) / dados['Volume'].rolling(20, min_periods=1).std()
dados['ATR_Zscore'] = (dados['ATR'] - dados['ATR'].rolling(20, min_periods=1).mean()) / dados['ATR'].rolling(20, min_periods=1).std()
dados['Target'] = np.where(dados['Close'].shift(-1) > dados['Close'], 1, 0)
dados.replace([np.inf, -np.inf], np.nan, inplace=True)
dados = dados.dropna()
dados.to_csv("data/NASDAQ100_Master_Dataset.csv")
print("-> Dataset NASDAQ100_Master_Dataset.csv criado com sucesso!")
