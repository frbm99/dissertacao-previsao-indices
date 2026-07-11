"""Etapa 2 — Cálculo dos indicadores técnicos do S&P 500 e criação do alvo binário t+1.
"""
import pandas as pd
import numpy as np
import ta

print("A carregar os dados brutos do S&P 500...")
dados = pd.read_csv("data/SP500_Bruto.csv", index_col='Date', parse_dates=True)

print("A calcular o Super Esquadrão de Variáveis Técnicas...")
# --- Família 2: Seguidores de Tendência ---
dados['SMA_20'] = ta.trend.SMAIndicator(close=dados['Close'], window=20).sma_indicator()
dados['SMA_50'] = ta.trend.SMAIndicator(close=dados['Close'], window=50).sma_indicator()
dados['EMA_20'] = ta.trend.EMAIndicator(close=dados['Close'], window=20).ema_indicator()
dados['MACD'] = ta.trend.MACD(close=dados['Close']).macd()
dados['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=dados['Close'], volume=dados['Volume']).on_balance_volume()

# --- Família 3: Osciladores de Momento ---
dados['RSI'] = ta.momentum.RSIIndicator(close=dados['Close'], window=14).rsi()
dados['ROC_10'] = ta.momentum.ROCIndicator(close=dados['Close'], window=10).roc()
dados['ROC_20'] = ta.momentum.ROCIndicator(close=dados['Close'], window=20).roc()
dados['MOM_10'] = dados['Close'].diff(periods=10) # Momentum
dados['Stoch_K'] = ta.momentum.StochasticOscillator(high=dados['High'], low=dados['Low'], close=dados['Close']).stoch()

# --- Família 4: Volatilidade ---
bollinger = ta.volatility.BollingerBands(close=dados['Close'], window=20, window_dev=2)
dados['BB_High'] = bollinger.bollinger_hband()
dados['BB_Low'] = bollinger.bollinger_lband()
dados['ATR'] = ta.volatility.AverageTrueRange(high=dados['High'], low=dados['Low'], close=dados['Close']).average_true_range()

# --- Criar o Target ---
dados['Target'] = np.where(dados['Close'].shift(-1) > dados['Close'], 1, 0)

# Limpeza das linhas vazias iniciais
dados.dropna(inplace=True)
dados = dados.iloc[:-1]

dados.to_csv("data/SP500_Pronto_Modelos.csv")
print(f"SUCESSO! O teu ficheiro tem agora {len(dados.columns)} colunas técnicas (Base + Indicadores + Target).")