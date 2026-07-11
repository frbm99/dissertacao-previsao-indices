"""Figura — Evolução histórica do preço de fecho do DJIA e do NASDAQ-100.
"""
import pandas as pd
import matplotlib.pyplot as plt

def plot_index(nome_indice, dataset_path, color_hex, output_image_name):
    print(f"A gerar o gráfico da evolução do preço para {nome_indice}...")
    dados = pd.read_csv(dataset_path, index_col='Date', parse_dates=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dados.index, dados['Close'], color=color_hex, linewidth=1)
    plt.title(f"Evolução Histórica do Preço de Fecho - {nome_indice} (2005 - Presente)", fontsize=14)
    plt.xlabel("Ano", fontsize=12)
    plt.ylabel("Preço de Fecho (USD)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_image_name, dpi=300)
    print(f"SUCESSO! O gráfico foi guardado na tua pasta como '{output_image_name}'.")
    plt.close()

# Gerar para DJIA (dataset reconstruido com história completa desde 2005)
plot_index("DJIA", "data/DJIA_Master_Dataset.csv", "#ff7f0e", "figures/Grafico_DJIA_Bruto.png")

# Gerar para NASDAQ-100 (índice oficial; o NASDAQ composto curto foi abandonado)
plot_index("NASDAQ-100", "data/NASDAQ100_Master_Dataset.csv", "#2ca02c", "figures/Grafico_NASDAQ100_Bruto.png")
