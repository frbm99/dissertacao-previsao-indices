"""Figura — Evolução histórica do preço de fecho do S&P 500 (estatísticas descritivas incluídas).
"""
import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar os dados que acabaste de descarregar
dados = pd.read_csv("data/SP500_Bruto.csv", index_col='Date', parse_dates=True)

# 2. Informações vitais para escreveres na Tese
print(f"Índice Analisado: S&P 500")
print(f"Data do primeiro registo: {dados.index.min().date()}")
print(f"Data do último registo: {dados.index.max().date()}")
print(f"Total de dias de negociação (linhas): {len(dados)}")
print(f"Total de variáveis (colunas): {len(dados.columns)} {list(dados.columns)}")

# O comando describe() calcula a média, mínimo, máximo e desvio padrão automaticamente
print(dados.describe().round(2)) 

# 3. Gerar um Gráfico Histórico para o teu documento
print("\nA gerar o gráfico da evolução do preço...")
plt.figure(figsize=(12, 6))
plt.plot(dados.index, dados['Close'], color='#1f77b4', linewidth=1)
plt.title("Evolução Histórica do Preço de Fecho - S&P 500 (2005 - Presente)", fontsize=14)
plt.xlabel("Ano", fontsize=12)
plt.ylabel("Preço de Fecho (USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Guardar o gráfico como imagem com alta qualidade (para o Word/LaTeX)
nome_imagem = "figures/Grafico_SP500_Bruto.png"
plt.tight_layout()
plt.savefig(nome_imagem, dpi=300)
print(f"SUCESSO! O gráfico foi guardado na tua pasta como '{nome_imagem}'.")

# Mostrar o gráfico no ecrã do Antigravity
plt.show()