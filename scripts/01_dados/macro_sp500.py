"""Etapa 2 — Integração das variáveis macroeconómicas no S&P 500; produz o dataset mestre.
"""
import pandas as pd
import yfinance as yf
import ta  # Importar a biblioteca de Análise Técnica


print("A carregar os dados técnicos base do S&P 500...")
dados = pd.read_csv("data/SP500_Pronto_Modelos.csv", index_col='Date', parse_dates=True)

# ====================================================================
# A FÁBRICA DE VARIÁVEIS (TESTE EXAUSTIVO PARA A ORIENTAÇÃO)
# ====================================================================
print("A gerar múltiplas janelas temporais de indicadores (SMAs, EMAs, RSIs, Returns)...")
janelas = [5, 10, 14, 20, 50, 100, 200]

for j in janelas:
    # 1. Médias Móveis (Simples e Exponenciais)
    dados[f'SMA_{j}'] = dados['Close'].rolling(window=j).mean()
    dados[f'EMA_{j}'] = dados['Close'].ewm(span=j, adjust=False).mean()
    
    # 2. Retornos Percentuais (Volatilidade e Momento)
    # Não faz sentido calcular o retorno de 200 dias para prever amanhã, logo limitamos aos 50 dias
    if j <= 50:
        dados[f'Return_{j}d'] = dados['Close'].pct_change(j)
        
    # 3. Índice de Força Relativa (RSI) para várias velocidades
    dados[f'RSI_{j}'] = ta.momentum.RSIIndicator(dados['Close'], window=j).rsi()

# ====================================================================

print("\nA ligar à base de dados para extrair a Macroeconomia global...")
tickers_macro = {
    'VIX': '^VIX',              # Índice do Medo
    'Petroleo_WTI': 'CL=F',     # Petróleo Bruto
    'Ouro': 'GC=F',             # Ouro
    'Dolar_Index': 'DX-Y.NYB',  # Força do Dólar
    'TBill_3M': '^IRX',         # Taxa de Juro 3 Meses
    'Treasury_10Y': '^TNX',      # Taxa de Juro 10 Anos
    'NASDAQ': '^IXIC', 
    'Dow_Jones': '^DJI'          # Para a correlação com o NASDAQ
}

data_inicio = dados.index.min()
data_fim = dados.index.max() + pd.Timedelta(days=5) 

dados_macro = pd.DataFrame(index=dados.index)

for nome_coluna, ticker in tickers_macro.items():
    print(f" -> A extrair {nome_coluna}...")
    temp = yf.download(ticker, start=data_inicio, end=data_fim, auto_adjust=False)
    
    if isinstance(temp.columns, pd.MultiIndex):
        temp.columns = temp.columns.droplevel(1)
        
    dados_macro[nome_coluna] = temp['Close']

print("\nA processar a Curva de Rendimentos...")
dados_macro['Spread_10Y_3M'] = dados_macro['Treasury_10Y'] - dados_macro['TBill_3M']

print("A fundir a Macroeconomia com a Análise Técnica...")
# Juntar tudo e limpar as linhas vazias (incluindo as geradas pelas médias de 200 dias)
dados_finais = dados.join(dados_macro).ffill().dropna()

print(f"Linhas iniciais (apenas preços): {len(dados)}")
print(f"Linhas após fusão (limpas): {len(dados_finais)}")
print(f"Total de Variáveis (Features + Target): {len(dados_finais.columns)}")

nome_final = "data/SP500_Master_Dataset.csv"
dados_finais.to_csv(nome_final)
print(f"\nSUCESSO! O ficheiro '{nome_final}' está gordo de features e pronto para a prova de fogo.")