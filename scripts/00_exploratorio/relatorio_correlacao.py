"""Análise exploratória: relatório HTML com as correlações das variáveis com o alvo.
"""
import pandas as pd
import os

print("A processar a matemática para o Relatório Web Atualizado...")

# Garantir que a pasta resultados existe
if not os.path.exists("results"):
    os.makedirs("results")

# 1. Carregar os dados (que agora já têm NASDAQ e Dow Jones se correstes o passo 2.5)
dados = pd.read_csv("data/SP500_Master_Dataset.csv", index_col='Date', parse_dates=True)

# 2. Calcular a correlação e arredondar para 4 casas decimais
matriz_corr = dados.corr().round(4)

# 3. Preparar a Tabela do Target
tabela_target = matriz_corr[['Target']].drop('Target').sort_values(by='Target', ascending=False)

# 4. Gerar o código HTML para as tabelas
html_target = tabela_target.to_html(classes='dataframe')
html_matriz = matriz_corr.to_html(classes='dataframe')

# 5. Criar o visual da página (CSS)
html_completo = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Relatório de Correlação Atualizado - S&P 500</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            padding: 40px;
            max-width: 1200px;
            margin: auto;
        }}
        h1, h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .tabela-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 40px;
            overflow-x: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 14px;
        }}
        th, td {{
            border: 1px solid #e0e0e0;
            text-align: center;
            padding: 12px;
        }}
        th {{
            background-color: #2c3e50;
            color: white;
            position: sticky;
            top: 0;
        }}
        tr:nth-child(even) {{ background-color: #fbfbfb; }}
        tr:hover {{ background-color: #f1f4f8; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Relatório de Correlação de Variáveis (Com Novas Features)</h1>
    <p>Este relatório analisa o impacto das 27 variáveis técnicas, macroeconómicas e de intermercado na direção (Target) do índice S&P 500.</p>
    
    <h2>1. Top Variáveis que afetam o Target (Subida/Descida)</h2>
    <div class="tabela-container">
        <p><i>Valores próximos de 1 significam forte probabilidade de subida. Valores próximos de -1 significam forte probabilidade de descida.</i></p>
        {html_target}
    </div>

    <h2>2. Matriz de Correlação Completa (Todas contra Todas)</h2>
    <div class="tabela-container">
        <p><i>Usa esta tabela para detetar variáveis redundantes (multicolinearidade).</i></p>
        {html_matriz}
    </div>
</body>
</html>
"""

# 6. Guardar o ficheiro HTML especificamente dentro da pasta 'resultados'
nome_ficheiro = "results/Relatorio_Correlacao.html"
with open(nome_ficheiro, "w", encoding="utf-8") as file:
    file.write(html_completo)

print(f" SUCESSO! O relatório atualizado foi guardado em '{nome_ficheiro}'.")