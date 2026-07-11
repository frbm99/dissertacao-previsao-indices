"""Etapa 5 — Ensembles com as redes otimizadas por HPO.

Carrega o modelo de rede otimizado (models/campeao_hpo_*.keras), treina o ramo de
árvore correspondente e ajusta o peso de fusão na validação de 2021 (teste cego).
Parametrizável por ENS_TYPE e ENS_INDICE.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, cohen_kappa_score, make_scorer)
import xgboost as xgb
import tensorflow as tf

# ====================================================================
# CONFIG (via ambiente):
#   ENS_TYPE in {lstm_xgb, lstm_rf, gru_xgb, gru_rf, deeptransf_rf, deeptransf_xgb, deeptransf_lstm}
#   ENS_INDICE in {sp500, djia, nasdaq100}
# Reutiliza os modelos HPO já guardados (campeao_hpo_{rede}_{índice}.keras) -> sem re-treino das redes.
# ====================================================================
ENS_TYPE = os.environ.get('ENS_TYPE', 'lstm_rf')
INDICE = os.environ.get('ENS_INDICE', 'sp500')
dl_family, partner = ENS_TYPE.rsplit('_', 1)   # ex: ('lstm','rf') ; ('deeptransf','lstm')

DATASETS = {'sp500': 'data/SP500_Master_Dataset.csv', 'djia': 'data/DJIA_Master_Dataset.csv',
            'nasdaq100': 'data/NASDAQ100_Master_Dataset.csv'}
NOMES_IDX = {'sp500': 'SP500', 'djia': 'DJIA', 'nasdaq100': 'NASDAQ-100'}
ETIQ = {'lstm': 'LSTM', 'gru': 'GRU', 'deeptransf': 'DeepTransformer', 'rf': 'RandomForest', 'xgb': 'XGBoost'}
DATASET = DATASETS[INDICE]
NOME_MODELO = f"Ensemble HPO {ETIQ[dl_family]}+{ETIQ[partner]} - {NOMES_IDX[INDICE]}"
CAMINHO_HTML = f"results/HPO_ENS_{ENS_TYPE}_{INDICE}.html"
N_FEATURES_SEL = 12
mcc_scorer = make_scorer(matthews_corrcoef)


with open('results/hpo_best_configs.json') as f:
    HPO = json.load(f)
cfg_dl = HPO[f"{dl_family}_{INDICE}"]
cfg_dl2 = HPO[f"lstm_{INDICE}"] if partner == 'lstm' else None
print(f"Config DL ({dl_family}): {cfg_dl}")
if cfg_dl2:
    print(f"Config DL2 (lstm): {cfg_dl2}")

# 1. Carregar + engenharia (idêntica a hpo_runner / todos os modelos)
dados = pd.read_csv(DATASET, index_col='Date', parse_dates=True)
dados['Target'] = (dados['Close'].shift(-1) > dados['Close']).astype(float)
dados.loc[dados.index[-1], 'Target'] = np.nan
dados.replace([np.inf, -np.inf], np.nan, inplace=True)
dados = dados.dropna()
dados['Target'] = dados['Target'].astype(int)
dados['Ouro_Ret'] = dados['Ouro'].pct_change()
dados['Petroleo_Ret'] = dados['Petroleo_WTI'].pct_change()
dados['Dolar_Ret'] = dados['Dolar_Index'].pct_change()
dados['Volume_Ret'] = dados['Volume'].pct_change()
dados['Treasury_10Y_Diff'] = dados['Treasury_10Y'].diff()
dados['TBill_3M_Diff'] = dados['TBill_3M'].diff()
dados['Spread_Diff'] = dados['Spread_10Y_3M'].diff()
colunas_medias = [col for col in dados.columns if col.startswith('SMA_') or col.startswith('EMA_')]
for col in colunas_medias:
    dados[f"Dist_{col}"] = (dados['Close'] / dados[col]) - 1
if 'MACD' in dados.columns: dados['MACD_Norm'] = dados['MACD'] / dados['Close']
if 'BB_High' in dados.columns: dados['Dist_BB_High'] = (dados['Close'] / dados['BB_High']) - 1
if 'BB_Low' in dados.columns: dados['Dist_BB_Low'] = (dados['Close'] / dados['BB_Low']) - 1
if 'OBV' in dados.columns:
    dados['OBV_Zscore'] = (dados['OBV'] - dados['OBV'].rolling(20, min_periods=1).mean()) / dados['OBV'].rolling(20, min_periods=1).std()
dados.replace([np.inf, -np.inf], np.nan, inplace=True)
dados = dados.dropna()
colunas_excluir = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume',
                   'Ouro', 'Petroleo_WTI', 'Dolar_Index', 'Treasury_10Y', 'TBill_3M', 'Spread_10Y_3M',
                   'SP500', 'NASDAQ', 'NASDAQ100', 'DJIA', 'Dow_Jones',
                   'Sentiment_Score', 'Sentiment_Score_MA3',
                   'MACD', 'OBV', 'BB_High', 'BB_Low', 'ATR', 'MOM_10'] + colunas_medias
banco_features = [col for col in dados.columns if col not in colunas_excluir]
X_raw = dados[banco_features]; y = dados['Target']

# 2. Seleção de features (determinística -> reproduz a mesma Top-12 do HPO)
X_fit = X_raw.loc[:'2019-12-31']; y_fit = y.loc[:'2019-12-31']
X_imp = X_raw.loc['2020-01-01':'2020-12-31']; y_imp = y.loc['2020-01-01':'2020-12-31']
seletor = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1, class_weight='balanced')
seletor.fit(X_fit, y_fit)
perm = permutation_importance(seletor, X_imp, y_imp, scoring=mcc_scorer, n_repeats=10, random_state=42, n_jobs=-1)
importancias = pd.Series(perm.importances_mean, index=banco_features).sort_values(ascending=False)
features_sel = importancias.nlargest(N_FEATURES_SEL).index.tolist()

# 3. Split + scaler (fit só no treino)
X = dados[features_sel]
X_train = X.loc[:'2020-12-31']; y_train = y.loc[:'2020-12-31']
X_val = X.loc['2021-01-01':'2021-12-31']; y_val = y.loc['2021-01-01':'2021-12-31']
X_test = X.loc['2022-01-01':'2025-12-31']; y_test = y.loc['2022-01-01':'2025-12-31']
scaler = Pipeline([('robust', RobustScaler()), ('minmax', MinMaxScaler(feature_range=(-1, 1)))])
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
yv = y_val.values; yt = y_test.values   # alvos canonicos (1 por dia), alinhados entre ramos

def make_sequences(cur_scaled, y_cur, lookback, ctx_scaled, y_ctx):
    ctx = ctx_scaled[-(lookback - 1):]
    X_comb = np.vstack([ctx, cur_scaled])
    y_comb = pd.concat([y_ctx.iloc[-(lookback - 1):], y_cur])
    X_seq = [X_comb[i:i + lookback] for i in range(len(X_comb) - lookback + 1)]
    return np.array(X_seq)

def prob_rede_hpo(family, cfg):
    """Carrega o modelo HPO ja guardado e devolve prob de validacao e teste (1 por dia)."""
    modelo = tf.keras.models.load_model(f"models/campeao_hpo_{family}_{INDICE}.keras")
    lb = cfg['lookback']
    Xv = make_sequences(X_val_s, y_val, lb, X_train_s, y_train)
    Xt = make_sequences(X_test_s, y_test, lb, X_val_s, y_val)
    return modelo.predict(Xv, verbose=0).flatten(), modelo.predict(Xt, verbose=0).flatten()

# ====================================================================
# RAMO A (parceiro) e RAMO B (rede DL principal)
# ====================================================================
nomeB = ETIQ[dl_family]
probB_val, probB_test = prob_rede_hpo(dl_family, cfg_dl)

if partner == 'rf':
    nomeA = 'RF'
    modeloA = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5,
                                     max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
    modeloA.fit(X_train, y_train)
    probA_val = modeloA.predict_proba(X_val)[:, 1]; probA_test = modeloA.predict_proba(X_test)[:, 1]
    joblib.dump(modeloA, f"models/campeao_hpoens_{ENS_TYPE}_{INDICE}_rf.pkl")
elif partner == 'xgb':
    nomeA = 'XGB'
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    modeloA = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, gamma=1, reg_alpha=0.1, reg_lambda=10, scale_pos_weight=spw,
                                objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                                device='cpu', random_state=42, n_jobs=-1)
    modeloA.fit(X_train, y_train)
    probA_val = modeloA.predict_proba(X_val)[:, 1]; probA_test = modeloA.predict_proba(X_test)[:, 1]
    joblib.dump(modeloA, f"models/campeao_hpoens_{ENS_TYPE}_{INDICE}_xgb.pkl")
else:  # partner == 'lstm' -> segundo ramo de rede (reusa modelo HPO do LSTM)
    nomeA = 'LSTM'
    probA_val, probA_test = prob_rede_hpo('lstm', cfg_dl2)

print(f"Alinhamento val: A({nomeA})={len(probA_val)} B({nomeB})={len(probB_val)} y={len(yv)} | "
      f"teste: A={len(probA_test)} B={len(probB_test)} y={len(yt)}")

# ====================================================================
# FUSÃO: peso afinado na VALIDAÇÃO 2021 (nunca no teste). Limiar fixo 0.5.
# ====================================================================
melhor_peso_a, melhor_mcc_val = 0.5, -2.0
for wa in np.round(np.arange(0.0, 1.01, 0.1), 2):
    m = matthews_corrcoef(yv, ((wa * probA_val + (1 - wa) * probB_val) > 0.5).astype(int))
    if m > melhor_mcc_val:
        melhor_mcc_val, melhor_peso_a = m, wa
wb = 1 - melhor_peso_a
print(f"-> Peso ideal (validacao): {nomeA}={melhor_peso_a*100:.0f}% | {nomeB}={wb*100:.0f}% (MCC val {melhor_mcc_val:.4f})")

probabilidades = melhor_peso_a * probA_test + wb * probB_test
previsoes = (probabilidades > 0.5).astype(int)
acc = accuracy_score(yt, previsoes) * 100
prec = precision_score(yt, previsoes, zero_division=0) * 100
rec = recall_score(yt, previsoes, zero_division=0) * 100
f1 = f1_score(yt, previsoes, zero_division=0) * 100
auc_roc = roc_auc_score(yt, probabilidades)
mcc = matthews_corrcoef(yt, previsoes)
kappa = cohen_kappa_score(yt, previsoes)
cm = confusion_matrix(yt, previsoes); TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
mcc_a = matthews_corrcoef(yt, (probA_test > 0.5).astype(int))
mcc_b = matthews_corrcoef(yt, (probB_test > 0.5).astype(int))
print(f"\nAccuracy: {acc:.2f}%  AUC: {auc_roc:.4f}  MCC: {mcc:.4f}  Kappa: {kappa:.4f}")
print(f"(referencia teste: MCC {nomeA}={mcc_a:.4f} | MCC {nomeB}={mcc_b:.4f})")
print(cm)

# ====================================================================
# Guardar consolidado: resultados/resultados_hpo_ensembles.csv (+ .html)
# ====================================================================
os.makedirs("results", exist_ok=True)
CSV = "results/resultados_hpo_ensembles.csv"
linha = {
    'Ensemble': f"{ETIQ[dl_family]}+{ETIQ[partner]}", 'Indice': NOMES_IDX[INDICE],
    'Fusao': f"{nomeA} {melhor_peso_a*100:.0f}% / {nomeB} {wb*100:.0f}%",
    'Lookback_DL': cfg_dl['lookback'], 'Units_DL': cfg_dl['units'], 'Dropout_DL': cfg_dl['dropout'],
    'Accuracy_%': round(acc, 2), 'AUC_ROC': round(auc_roc, 4), 'MCC': round(mcc, 4), 'Kappa': round(kappa, 4),
    f'MCC_{nomeA}_iso': round(mcc_a, 4), f'MCC_{nomeB}_iso': round(mcc_b, 4),
    'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP)
}
chave_ens = f"{ETIQ[dl_family]}+{ETIQ[partner]}"
if os.path.exists(CSV):
    tab = pd.read_csv(CSV)
    tab = tab[~((tab['Ensemble'] == chave_ens) & (tab['Indice'] == NOMES_IDX[INDICE]))]
    tab = pd.concat([tab, pd.DataFrame([linha])], ignore_index=True)
else:
    tab = pd.DataFrame([linha])
tab = tab.sort_values(['Ensemble', 'Indice'])
tab.to_csv(CSV, index=False)

ths = "".join(f"<th>{c}</th>" for c in tab.columns)
trs = "".join("<tr>" + "".join(f"<td>{r[c]}</td>" for c in tab.columns) + "</tr>" for _, r in tab.iterrows())
with open("results/resultados_hpo_ensembles.html", "w", encoding="utf-8") as fh:
    fh.write(f"""<!DOCTYPE html><html lang="pt"><head><meta charset="UTF-8"><title>Ensembles HPO</title>
<style>body{{font-family:'Segoe UI',Tahoma,sans-serif;background:#0f172a;color:#f8fafc;margin:30px;}}
h1{{color:#38bdf8;font-size:1.4rem;}} .wrap{{overflow-x:auto;}} table{{border-collapse:collapse;width:100%;font-size:0.8rem;}}
th,td{{border:1px solid #334155;padding:6px 8px;text-align:center;white-space:nowrap;}}
th{{background:#1e293b;color:#eab308;text-transform:uppercase;font-size:0.66rem;}} td{{background:#0b1220;}}</style></head>
<body><h1>Ensembles com redes HPO — teste 2022-2025</h1><div class="wrap"><table><tr>{ths}</tr>{trs}</table></div></body></html>""")

# HTML individual de uma linha
features_str = ", ".join(features_sel)
params_str = (f"fusao {nomeA}={melhor_peso_a*100:.0f}%/{nomeB}={wb*100:.0f}%, "
              f"DL={dl_family} lookback={cfg_dl['lookback']} units={cfg_dl['units']} dropout={cfg_dl['dropout']} "
              f"lr={cfg_dl['learning_rate']} batch={cfg_dl['batch_size']} (config HPO), thr=0.50")
with open(CAMINHO_HTML, "w", encoding="utf-8") as fh:
    fh.write(f"""<!DOCTYPE html><html lang="pt"><head><meta charset="UTF-8"><title>{NOME_MODELO}</title>
<style>body{{font-family:'Segoe UI',Tahoma,sans-serif;background:#0f172a;color:#f8fafc;margin:30px;}}
h1{{color:#38bdf8;font-size:1.4rem;}} .wrap{{overflow-x:auto;}} table{{border-collapse:collapse;width:100%;font-size:0.85rem;}}
th,td{{border:1px solid #334155;padding:8px 10px;text-align:center;white-space:nowrap;}}
th{{background:#1e293b;color:#eab308;text-transform:uppercase;font-size:0.72rem;}} td{{background:#0b1220;}}
td.txt{{text-align:left;white-space:normal;max-width:340px;font-size:0.78rem;color:#cbd5e1;}}</style></head>
<body><h1>{NOME_MODELO} (t+1)</h1><div class="wrap"><table>
<tr><th>Modelo</th><th>Accuracy (%)</th><th>AUC ROC</th><th>Precision (%)</th><th>Recall (%)</th>
<th>F1-Score (%)</th><th>MCC</th><th>Kappa</th><th>Feature Set</th><th>Features por Extenso</th>
<th>Parâmetros Vencedores</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th></tr>
<tr><td>{NOME_MODELO}</td><td>{acc:.2f}</td><td>{auc_roc:.4f}</td><td>{prec:.2f}</td><td>{rec:.2f}</td>
<td>{f1:.2f}</td><td>{mcc:.4f}</td><td>{kappa:.4f}</td><td>Permutation Top {N_FEATURES_SEL}</td>
<td class="txt">{features_str}</td><td class="txt">{params_str}</td>
<td>{TN}</td><td>{FP}</td><td>{FN}</td><td>{TP}</td></tr></table></div></body></html>""")
print(f"[OK] {CAMINHO_HTML} + resultados/resultados_hpo_ensembles.csv atualizados.")
