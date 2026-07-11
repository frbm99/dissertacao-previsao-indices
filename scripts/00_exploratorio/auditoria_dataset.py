"""Análise exploratória: auditoria de integridade do dataset mestre (nulos, balanceamento do alvo, períodos).
"""
import pandas as pd

print("A INICIAR A AUDITORIA DE DADOS DA TESE ")

# Carregar os dados finais
dados = pd.read_csv("data/SP500_Master_Dataset.csv", index_col='Date', parse_dates=True)

# --- TESTE 1: Valores Nulos (Os buracos) ---
print("\n1. TESTE DE INTEGRIDADE (Valores Nulos):")
total_nulos = dados.isnull().sum().sum()
if total_nulos == 0:
    print("Aprovado: A tabela está 100% preenchida. Zero valores nulos.")
else:
    print(f" Chumbo: Encontrados {total_nulos} valores vazios. Precisamos de rever a limpeza.")

# --- TESTE 2: Distribuição do Target ---
print("\n2. TESTE DE BALANCEAMENTO (Distribuição do Alvo):")
contagem = dados['Target'].value_counts(normalize=True) * 100
print(f"Dias de Subida (1): {contagem.get(1, 0):.1f}%")
print(f"Dias de Descida (0): {contagem.get(0, 0):.1f}%")
print("Nota: É normal haver ligeiramente mais dias de subida (ex: 53%) porque o S&P 500 tende a crescer a longo prazo.")

# --- TESTE 3: Prova Lógica do Target (Verificação Manual) ---
print("\n3. PROVA LÓGICA DO TARGET (Amostra de 3 dias):")
# Vamos buscar apenas a coluna Close e Target para os 3 primeiros dias
amostra = dados[['Close', 'Target']].head(3)
print(amostra)
print("Nota: Verifica tu mesmo. Se o 'Close' do 2º dia for maior que o do 1º dia, o 'Target' do 1º dia TEM de ser 1.0.")

# --- TESTE 4: Teste de Sanidade Macroeconómica (O Pânico do Covid) ---
print("\n4. TESTE DE SANIDADE MACRO (O Crash de Março de 2020):")
try:
    # 16 de Março de 2020 foi um dos piores dias da história da bolsa
    dia_covid = dados.loc['2020-03-16']
    print(f"Preço de Fecho (S&P 500): {dia_covid['Close']:.2f} (Uma queda brutal)")
    print(f"Índice do Medo (VIX): {dia_covid['VIX']:.2f}")
    
    if dia_covid['VIX'] > 40:
         print("Aprovado: O VIX está altíssimo, o que reflete perfeitamente o pânico mundial desse dia.")
    else:
         print("Chumbo: O VIX está demasiado baixo para uma crise, os dados macro podem estar desalinhados.")
except KeyError:
    print("Dia 16-03-2020 não encontrado na tabela.")

print("\n" + "-" * 50)
print("FIM DA AUDITORIA ")