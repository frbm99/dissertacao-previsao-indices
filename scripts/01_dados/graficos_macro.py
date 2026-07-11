"""Figura — Evolução das variáveis macroeconómicas (VIX, taxas de juro) usadas na dissertação.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

print("A preparar os gráficos macroeconómicos...")

# 1. Garantir que a pasta resultados existe
if not os.path.exists("results"):
    os.makedirs("results")

# 2. Carregar os dados que já tens
dados = pd.read_csv("data/SP500_Master_Dataset.csv", index_col='Date', parse_dates=True)

# 3. Criar uma figura com 2 gráficos (um em cima do outro)
fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=("Índice de Volatilidade CBOE (VIX) - O 'Índice do Medo'", 
                                    "Taxa de Juro a 3 Meses (T-Bill 3M) - O Custo do Dinheiro"),
                    vertical_spacing=0.15)

# Gráfico 1: VIX (Cor vermelha para representar risco)
fig.add_trace(go.Scatter(x=dados.index, y=dados['VIX'], 
                         mode='lines', name='VIX', line=dict(color='#e74c3c', width=1.5)), 
              row=1, col=1)

# Gráfico 2: T-Bill 3M (Cor azul para representar a política monetária)
fig.add_trace(go.Scatter(x=dados.index, y=dados['TBill_3M'], 
                         mode='lines', name='T-Bill 3M', line=dict(color='#2980b9', width=1.5)), 
              row=2, col=1)

# 4. Adicionar anotações das crises no gráfico do VIX (Para impressionar na tese)
crise_2008 = '2008-10-24'
covid_2020 = '2020-03-16'

if crise_2008 in dados.index:
    fig.add_annotation(x=crise_2008, y=dados.loc[crise_2008, 'VIX'],
                       text="Crise Financeira (2008)", showarrow=True, arrowhead=1, row=1, col=1)

if covid_2020 in dados.index:
    fig.add_annotation(x=covid_2020, y=dados.loc[covid_2020, 'VIX'],
                       text="Pandemia COVID-19", showarrow=True, arrowhead=1, row=1, col=1)

# 5. Formatação de estilo profissional
fig.update_layout(
    title_text="Evolução Macroeconómica Global (2005 - 2025)",
    title_x=0.5,
    height=800,
    template="plotly_white",
    showlegend=False
)

# Eixos Y
fig.update_yaxes(title_text="Pontos", row=1, col=1)
fig.update_yaxes(title_text="Taxa (%)", row=2, col=1)

# 6. Guardar em HTML Interativo e numa Imagem Estática (para o PDF da Tese)
ficheiro_html = "results/Evolucao_Macro.html"

fig.write_html(ficheiro_html)

print(f" SUCESSO! Abre o ficheiro '{ficheiro_html}' no teu browser.")
print("Dica: No browser, podes usar a máquina fotográfica no canto superior direito do gráfico para fazer download da imagem com qualidade perfeita para a tua tese.")