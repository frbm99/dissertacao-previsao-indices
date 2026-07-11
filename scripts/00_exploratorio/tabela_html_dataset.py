"""Análise exploratória: exporta as últimas linhas do dataset mestre para uma tabela HTML.
"""
import pandas as pd

print("A gerar o visualizador do Master Dataset...")
dados = pd.read_csv("data/SP500_Master_Dataset.csv", index_col='Date', parse_dates=True)

# Vamos ver apenas os últimos 100 dias para ser super rápido
html_table = dados.tail(100).to_html(classes='dataframe')

# Adicionar um bocado de estilo (CSS) para ficar com aspeto profissional
html_completo = f"""
<html>
<head>
    <title>S&P 500 Master Dataset</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
        th, td {{ border: 1px solid #dddddd; text-align: right; padding: 8px; }}
        th {{ background-color: #2c3e50; color: white; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
    </style>
</head>
<body>
    <h2>S&P 500 - Últimos 100 Dias de Negociação</h2>
    <p>Total de variáveis: {len(dados.columns)}</p>
    {html_table}
</body>
</html>
"""

with open("results/S&P500_Tabela.html", "w", encoding="utf-8") as file:
    file.write(html_completo)

print("SUCESSO! Abre o ficheiro 'S&P500_Tabela.html' no teu browser (Chrome/Edge).")