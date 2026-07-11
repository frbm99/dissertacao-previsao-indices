"""Etapa 3 — Modelo XGBoost para previsão da direção diária (t+1) do DJIA.

Seleção de variáveis por permutation importance, validação walk-forward
(TimeSeriesSplit) dentro do período de treino e otimização de hiperparâmetros
por RandomizedSearchCV; avaliação final no teste cego (2022-2025).
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, cohen_kappa_score, make_scorer)
import xgboost as xgb

# Parâmetros do índice (única secção que difere entre os três índices).
DATASET = "data/DJIA_Master_Dataset.csv"
NOME_MODELO = "XGBoost - DJIA"
CAMINHO_HTML = "results/XGB_DJIA.html"
MODEL_PATH = "models/campeao_xgb_djia.pkl"

mcc_scorer = make_scorer(matthews_corrcoef)


# 1. Carregar o Dataset Completo
print(f"A carregar o {DATASET}...")
dados = pd.read_csv(DATASET, index_col='Date', parse_dates=True)

# Alvo binário t+1: 1 se o fecho do dia seguinte for superior ao do próprio dia.
# A última linha não tem futuro observável (shift(-1) = NaN); com np.where, NaN > x avalia a
# False e criava-se um rótulo fabricado que o dropna não removeria. Força-se NaN para garantir a remoção.
dados['Target'] = (dados['Close'].shift(-1) > dados['Close']).astype(float)
dados.loc[dados.index[-1], 'Target'] = np.nan

dados.replace([np.inf, -np.inf], np.nan, inplace=True)
dados = dados.dropna()
dados['Target'] = dados['Target'].astype(int)

print("\nA aplicar Engenharia de Dados (Estacionaridade) para o XGBoost...")
dados['Ouro_Ret'] = dados['Ouro'].pct_change()
dados['Petroleo_Ret'] = dados['Petroleo_WTI'].pct_change()
dados['Dolar_Ret'] = dados['Dolar_Index'].pct_change()
dados['Volume_Ret'] = dados['Volume'].pct_change()
dados['Treasury_10Y_Diff'] = dados['Treasury_10Y'].diff()
dados['TBill_3M_Diff'] = dados['TBill_3M'].diff()
dados['Spread_Diff'] = dados['Spread_10Y_3M'].diff()

# Converte as médias móveis em distâncias relativas ao fecho (estacionárias).
colunas_medias = [col for col in dados.columns if col.startswith('SMA_') or col.startswith('EMA_')]
for col in colunas_medias:
    dados[f"Dist_{col}"] = (dados['Close'] / dados[col]) - 1

# Versões estacionárias dos indicadores em nível (árvores e redes não extrapolam para fora do
# intervalo observado no treino). ATR e MOM_10 são excluídos adiante (existem ATR_Zscore e Return_10d).
if 'MACD' in dados.columns:
    dados['MACD_Norm'] = dados['MACD'] / dados['Close']
if 'BB_High' in dados.columns:
    dados['Dist_BB_High'] = (dados['Close'] / dados['BB_High']) - 1
if 'BB_Low' in dados.columns:
    dados['Dist_BB_Low'] = (dados['Close'] / dados['BB_Low']) - 1
if 'OBV' in dados.columns:
    dados['OBV_Zscore'] = (dados['OBV'] - dados['OBV'].rolling(20, min_periods=1).mean()) / dados['OBV'].rolling(20, min_periods=1).std()

dados.replace([np.inf, -np.inf], np.nan, inplace=True)
dados = dados.dropna()

# Exclui o rótulo, os níveis brutos (preço, macro e índices) e os indicadores não estacionários.
# Os níveis de outros índices são sempre excluídos, para evitar fuga de informação entre mercados.
colunas_excluir = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume',
                   'Ouro', 'Petroleo_WTI', 'Dolar_Index', 'Treasury_10Y', 'TBill_3M', 'Spread_10Y_3M',
                   'SP500', 'NASDAQ', 'NASDAQ100', 'DJIA', 'Dow_Jones',
                   'Sentiment_Score', 'Sentiment_Score_MA3',
                   'MACD', 'OBV', 'BB_High', 'BB_Low', 'ATR', 'MOM_10'] + colunas_medias
banco_features = [col for col in dados.columns if col not in colunas_excluir]

X_raw = dados[banco_features]
y_raw = dados['Target']

# scale_pos_weight (equivalente do class_weight='balanced' no XGBoost) calculado no treino <=2021
y_tr_global = y_raw.loc[:'2021-12-31']
spw = (y_tr_global == 0).sum() / max((y_tr_global == 1).sum(), 1)
print(f"scale_pos_weight (neg/pos) no treino: {spw:.3f}")

# ====================================================================
# FASE 1: TRIAGEM DE FEATURE SETS (sem nunca tocar no teste 2022+)
#   <=2019 : treino do seletor de importancias
#   2020   : avaliação OUT-OF-SAMPLE da permutation importance
#   <=2021 : validação WALK-FORWARD (TimeSeriesSplit) para escolher o feature set
#   2022+  : teste cego
# ====================================================================
print("\n--- FASE 1: TRIAGEM DE FEATURE SETS (Permutation Importance + Walk-Forward) ---")

X_fit = X_raw.loc[:'2019-12-31']
y_fit = y_raw.loc[:'2019-12-31']
X_imp = X_raw.loc['2020-01-01':'2020-12-31']
y_imp = y_raw.loc['2020-01-01':'2020-12-31']
spw_fit = (y_fit == 0).sum() / max((y_fit == 1).sum(), 1)

print("A treinar o seletor (<=2019) e a medir Permutation Importance (2020)...")
seletor_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                random_state=42, n_jobs=-1, tree_method='hist', device='cpu',
                                objective='binary:logistic', eval_metric='logloss',
                                scale_pos_weight=spw_fit)
seletor_xgb.fit(X_fit, y_fit)

# Permutation importance out-of-sample: mede a queda de MCC ao permutar cada variável em 2020.
# Substitui a importância interna do XGBoost (ganho, in-sample) por uma medida out-of-sample.
perm = permutation_importance(seletor_xgb, X_imp, y_imp, scoring=mcc_scorer,
                              n_repeats=10, random_state=42, n_jobs=-1)
importancias = pd.Series(perm.importances_mean, index=banco_features).sort_values(ascending=False)

top_5_xgb = importancias.nlargest(5).index.tolist()
top_10_xgb = importancias.nlargest(10).index.tolist()
top_15_xgb = importancias.nlargest(15).index.tolist()
top_20_xgb = importancias.nlargest(20).index.tolist()

# Escudo Macro + filtro de Pearson (correlacao calculada em <=2020)
X_corr = X_raw.loc[:'2020-12-31']
macro_obrigatorias = ['VIX', 'Ouro_Ret', 'Treasury_10Y_Diff', 'Spread_Diff', 'Dolar_Ret']
features_pearson = [var for var in macro_obrigatorias if var in X_corr.columns]

LIMIAR_CORRELACAO = 0.75
LIMIAR_IMPORTANCIA = 0.0  # com permutation importance, so mantemos features uteis (queda de MCC > 0)

print(f"A gerar Conjunto_Dinamico_Pearson (Filtro < {LIMIAR_CORRELACAO*100}% e importancia > 0)...")
for candidato in importancias.index:
    if importancias[candidato] <= LIMIAR_IMPORTANCIA: break
    if candidato in features_pearson: continue
    too_correlated = False
    for ja_selecionado in features_pearson:
        if abs(X_corr[candidato].corr(X_corr[ja_selecionado])) > LIMIAR_CORRELACAO:
            too_correlated = True
            break
    if not too_correlated: features_pearson.append(candidato)

feature_sets = {
    'Top_5_XGBoost': top_5_xgb,
    'Top_10_XGBoost': top_10_xgb,
    'Top_15_XGBoost': top_15_xgb,
    'Top_20_XGBoost': top_20_xgb,
    'Conjunto_Dinamico_Pearson': features_pearson
}

resultados_fase1 = []
tscv_f1 = TimeSeriesSplit(n_splits=5)
print("\nA avaliar cada feature set por WALK-FORWARD (media de MCC em 5 janelas <=2021)...")
for nome_set, features_do_set in feature_sets.items():
    X_set = dados.loc[:'2021-12-31', features_do_set]
    y_set = dados.loc[:'2021-12-31', 'Target']

    modelo_estatico = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                        random_state=42, n_jobs=-1, tree_method='hist', device='cpu',
                                        objective='binary:logistic', eval_metric='logloss',
                                        scale_pos_weight=spw)
    scores = cross_val_score(modelo_estatico, X_set, y_set, cv=tscv_f1, scoring=mcc_scorer)
    mcc_val = scores.mean()

    resultados_fase1.append({'Feature_Set': nome_set, 'Features': features_do_set, 'MCC_Val': mcc_val})
    print(f"[{nome_set}] MCC walk-forward medio: {mcc_val:.4f}")

df_res_f1 = pd.DataFrame(resultados_fase1).sort_values(by='MCC_Val', ascending=False)
campeao_fase1 = df_res_f1.iloc[0]

features_vencedoras = campeao_fase1['Features']
nome_campeao = campeao_fase1['Feature_Set']

print(f"\n[CAMPEAO] da Fase 1: {nome_campeao} (MCC walk-forward: {campeao_fase1['MCC_Val']:.4f})")

# ====================================================================
# FASE 2: OTIMIZAÇÃO SUPREMA DO XGBOOST FINAL
# ====================================================================
X = dados[features_vencedoras]
y = dados['Target']

X_treino = X.loc[:'2021-12-31']
y_treino = y.loc[:'2021-12-31']
X_teste = X.loc['2022-01-01':'2025-12-31']
y_teste = y.loc['2022-01-01':'2025-12-31']

print("\nA otimizar hiperparâmetros (Randomized Search com Regularização, CPU)...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [1, 2, 3],             # Árvores rasas protegem contra overfitting
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'gamma': [0.1, 1, 5],               # Poda rigorosa
    'reg_alpha': [0, 0.1, 1],           # Regularização L1
    'reg_lambda': [1, 10]               # Regularização L2
}

tscv = TimeSeriesSplit(n_splits=5)

modelo_xgb = xgb.XGBClassifier(
    random_state=42,
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    device='cpu',
    scale_pos_weight=spw
)

grid_search = RandomizedSearchCV(
    estimator=modelo_xgb,
    param_distributions=param_grid,
    n_iter=100,
    cv=tscv,
    scoring=mcc_scorer,
    n_jobs=-1,
    random_state=42
)
grid_search.fit(X_treino, y_treino)

melhor_xgb = grid_search.best_estimator_
print(f"\n[MELHORES PARÂMETROS] Encontrados: {grid_search.best_params_}")

# Persistência do modelo final.
joblib.dump(melhor_xgb, MODEL_PATH)
print(f"[OK] Modelo guardado em {MODEL_PATH}")

# ====================================================================
# AVALIAÇÃO FINAL NO MERCADO PÓS-2022
# ====================================================================
print("\nA avaliar o modelo cego no período de teste...")
previsoes = melhor_xgb.predict(X_teste)
probabilidades = melhor_xgb.predict_proba(X_teste)[:, 1]

acc = accuracy_score(y_teste, previsoes) * 100
prec = precision_score(y_teste, previsoes, zero_division=0) * 100
rec = recall_score(y_teste, previsoes, zero_division=0) * 100
f1 = f1_score(y_teste, previsoes, zero_division=0) * 100

auc_roc = roc_auc_score(y_teste, probabilidades)
mcc = matthews_corrcoef(y_teste, previsoes)
kappa = cohen_kappa_score(y_teste, previsoes)

cm = confusion_matrix(y_teste, previsoes)
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

print(f"Accuracy (Taxa de Acerto):    {acc:.2f}%")
print(f"Precision (Precisão):         {prec:.2f}%")
print(f"Recall (Sensibilidade):       {rec:.2f}%")
print(f"F1-Score:                     {f1:.2f}%")
print(f"AUC-ROC (Capacidade):         {auc_roc:.4f}")
print(f"MCC (Correlação de Matthews): {mcc:.4f}")

print("Matriz de Confusão:")
print(cm)

# ====================================================================
# GERAÇÃO DO HTML (tabela de uma linha)
# ====================================================================
os.makedirs("results", exist_ok=True)

best_params_str = ", ".join([f"{k}={v}" for k, v in grid_search.best_params_.items()])
features_str = ", ".join(features_vencedoras)

html = f"""<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<title>{NOME_MODELO}</title>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background:#0f172a; color:#f8fafc; margin:30px; }}
    h1 {{ color:#38bdf8; font-size:1.4rem; }}
    .wrap {{ overflow-x:auto; }}
    table {{ border-collapse: collapse; width:100%; font-size:0.85rem; }}
    th, td {{ border:1px solid #334155; padding:8px 10px; text-align:center; white-space:nowrap; }}
    th {{ background:#1e293b; color:#eab308; text-transform:uppercase; font-size:0.72rem; letter-spacing:0.5px; }}
    td {{ background:#0b1220; }}
    td.txt {{ text-align:left; white-space:normal; max-width:340px; font-size:0.78rem; color:#cbd5e1; }}
</style>
</head>
<body>
<h1>{NOME_MODELO} (t+1)</h1>
<div class="wrap">
<table>
<tr>
    <th>Modelo</th><th>Accuracy (%)</th><th>AUC ROC</th><th>Precision (%)</th>
    <th>Recall (%)</th><th>F1-Score (%)</th><th>MCC</th><th>Kappa</th>
    <th>Feature Set</th><th>Features por Extenso</th><th>Parâmetros Vencedores</th>
    <th>TN</th><th>FP</th><th>FN</th><th>TP</th>
</tr>
<tr>
    <td>{NOME_MODELO}</td><td>{acc:.2f}</td><td>{auc_roc:.4f}</td><td>{prec:.2f}</td>
    <td>{rec:.2f}</td><td>{f1:.2f}</td><td>{mcc:.4f}</td><td>{kappa:.4f}</td>
    <td>{nome_campeao}</td><td class="txt">{features_str}</td><td class="txt">{best_params_str}</td>
    <td>{TN}</td><td>{FP}</td><td>{FN}</td><td>{TP}</td>
</tr>
</table>
</div>
</body>
</html>
"""
with open(CAMINHO_HTML, "w", encoding="utf-8") as file:
    file.write(html)

print(f"\n[OK] Ficheiro {CAMINHO_HTML} gerado com sucesso!")
