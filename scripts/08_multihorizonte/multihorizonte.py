"""Fase 4 (Etapa 9) — Previsão direcional a múltiplos horizontes (t+h) com os campeões.

Reutiliza EXATAMENTE o pipeline dos campeões de validação (ensemble rede+RF), alterando apenas
a definição do alvo: Target_h = 1 se Close(t+h) > Close(t). Tudo o resto é idêntico às fases
anteriores (Top-12, RobustScaler->MinMax só no treino, split treino<=2020 / val 2021 / teste
2022-2025, limiar 0,5, seeds fixas). Não importa nem altera scripts existentes — a lógica é copiada.

Anti-leakage do alvo: as últimas h linhas não têm Close(t+h) e são removidas (tal como a última
linha no caso t+1), para não entrarem no treino nem na avaliação.

Parametrizável por MH_INDICE (sp500/djia/nasdaq100) e MH_HORIZON (1/3/5/7).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import time
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, cohen_kappa_score, make_scorer)
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D, Embedding)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

INDICE = os.environ.get('MH_INDICE', 'sp500')
H = int(os.environ.get('MH_HORIZON', 3))
N_SEEDS = int(os.environ.get('MH_SEEDS', 8)); EPOCHS = int(os.environ.get('MH_EPOCHS', 60))
N_FEATURES_SEL = 12
DATASETS = {'sp500': 'data/SP500_Master_Dataset.csv', 'djia': 'data/DJIA_Master_Dataset.csv', 'nasdaq100': 'data/NASDAQ100_Master_Dataset.csv'}
NOMES = {'sp500': 'SP500', 'djia': 'DJIA', 'nasdaq100': 'NASDAQ-100'}
ETIQ = {'deeptransf': 'DeepTransformer', 'lstm': 'LSTM', 'gru': 'GRU'}
CAMPEOES = {'sp500': ('deeptransf', True), 'djia': ('lstm', False), 'nasdaq100': ('gru', False)}
NN, USE_HPO = CAMPEOES[INDICE]
mcc_scorer = make_scorer(matthews_corrcoef)

if USE_HPO:
    with open('results/hpo_best_configs.json') as f:
        cfg = json.load(f)[f"{NN}_{INDICE}"]
    LB = cfg['lookback']; UNITS = cfg['units']; DROP = cfg['dropout']; LR = cfg['learning_rate']; BS = cfg['batch_size']
    cen = 'HPO'
else:
    LB = 20; UNITS = 32; DROP = 0.2; LR = 1e-3; BS = (32 if NN == 'deeptransf' else 16)
    cen = 'sem-HPO'
CAMPEAO = f"{ETIQ[NN]}+RF ({cen})"
print(f"=== MULTIHORIZONTE {NOMES[INDICE]} — {CAMPEAO} | horizonte t+{H} | lookback={LB} ===\n")

t_ini = time.perf_counter()
dados = pd.read_csv(DATASETS[INDICE], index_col='Date', parse_dates=True)
# Alvo a h dias: 1 se Close(t+h) > Close(t). As últimas h linhas não têm futuro -> NaN -> removidas.
dados['Target'] = (dados['Close'].shift(-H) > dados['Close']).astype(float)
dados.loc[dados.index[-H:], 'Target'] = np.nan
dados.replace([np.inf, -np.inf], np.nan, inplace=True); dados = dados.dropna(); dados['Target'] = dados['Target'].astype(int)
dados['Ouro_Ret'] = dados['Ouro'].pct_change(); dados['Petroleo_Ret'] = dados['Petroleo_WTI'].pct_change()
dados['Dolar_Ret'] = dados['Dolar_Index'].pct_change(); dados['Volume_Ret'] = dados['Volume'].pct_change()
dados['Treasury_10Y_Diff'] = dados['Treasury_10Y'].diff(); dados['TBill_3M_Diff'] = dados['TBill_3M'].diff()
dados['Spread_Diff'] = dados['Spread_10Y_3M'].diff()
cm = [c for c in dados.columns if c.startswith('SMA_') or c.startswith('EMA_')]
for col in cm: dados[f"Dist_{col}"] = (dados['Close'] / dados[col]) - 1
if 'MACD' in dados.columns: dados['MACD_Norm'] = dados['MACD'] / dados['Close']
if 'BB_High' in dados.columns: dados['Dist_BB_High'] = (dados['Close'] / dados['BB_High']) - 1
if 'BB_Low' in dados.columns: dados['Dist_BB_Low'] = (dados['Close'] / dados['BB_Low']) - 1
if 'OBV' in dados.columns:
    dados['OBV_Zscore'] = (dados['OBV'] - dados['OBV'].rolling(20, min_periods=1).mean()) / dados['OBV'].rolling(20, min_periods=1).std()
dados.replace([np.inf, -np.inf], np.nan, inplace=True); dados = dados.dropna()
excl = ['Target','Open','High','Low','Close','Volume','Ouro','Petroleo_WTI','Dolar_Index','Treasury_10Y','TBill_3M',
        'Spread_10Y_3M','SP500','NASDAQ','NASDAQ100','DJIA','Dow_Jones','Sentiment_Score','Sentiment_Score_MA3',
        'MACD','OBV','BB_High','BB_Low','ATR','MOM_10'] + cm
banco = [c for c in dados.columns if c not in excl]; y = dados['Target']
Xf = dados.loc[:'2019-12-31', banco]; yf = y.loc[:'2019-12-31']
Xi = dados.loc['2020-01-01':'2020-12-31', banco]; yi = y.loc['2020-01-01':'2020-12-31']
sel = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1, class_weight='balanced').fit(Xf, yf)
perm = permutation_importance(sel, Xi, yi, scoring=mcc_scorer, n_repeats=10, random_state=42, n_jobs=-1)
top12 = pd.Series(perm.importances_mean, index=banco).sort_values(ascending=False).nlargest(N_FEATURES_SEL).index.tolist()

def seq(cur, ycur, ctx, yctx):
    cur = np.vstack([ctx[-(LB-1):], cur]); ycur = pd.concat([yctx.iloc[-(LB-1):], ycur])
    return (np.array([cur[i:i+LB] for i in range(len(cur)-LB+1)]),
            np.array([ycur.iloc[i+LB-1] for i in range(len(cur)-LB+1)]))

def construir(nf):
    if NN in ('lstm', 'gru'):
        cam = LSTM if NN == 'lstm' else GRU
        m = Sequential([Input(shape=(LB, nf)), cam(UNITS, activation='tanh'), Dropout(DROP), Dense(1, activation='sigmoid')])
    else:
        inp = Input(shape=(LB, nf)); x = Dense(UNITS, activation="relu", kernel_regularizer=l2(1e-4))(inp)
        x = x + Embedding(input_dim=LB, output_dim=UNITS)(tf.range(start=0, limit=LB, delta=1))
        for _ in range(2):
            a = Dropout(DROP)(MultiHeadAttention(num_heads=2, key_dim=UNITS)(x, x))
            o = LayerNormalization(epsilon=1e-6)(x + a)
            ff = Dense(UNITS, activation="relu", kernel_regularizer=l2(1e-4))(o); ff = Dropout(DROP)(ff)
            ff = Dense(UNITS, kernel_regularizer=l2(1e-4))(ff); x = LayerNormalization(epsilon=1e-6)(o + ff)
        m = Model(inp, Dense(1, activation='sigmoid')(Dropout(DROP)(GlobalAveragePooling1D()(x))))
    m.compile(Adam(LR), 'binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')]); return m

X = dados[top12]
Xtr, ytr = X.loc[:'2020-12-31'], y.loc[:'2020-12-31']
Xv, yv = X.loc['2021-01-01':'2021-12-31'], y.loc['2021-01-01':'2021-12-31']
Xte, yte = X.loc['2022-01-01':'2025-12-31'], y.loc['2022-01-01':'2025-12-31']
rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, max_features='sqrt',
                            class_weight='balanced', random_state=42, n_jobs=-1).fit(Xtr, ytr)
prv = rf.predict_proba(Xv)[:, 1]; prt = rf.predict_proba(Xte)[:, 1]
sc = Pipeline([('r', RobustScaler()), ('m', MinMaxScaler((-1, 1)))])
Xtr_s = sc.fit_transform(Xtr); Xv_s = sc.transform(Xv); Xte_s = sc.transform(Xte)
Xtr3 = np.array([Xtr_s[i:i+LB] for i in range(len(Xtr_s)-LB+1)]); ytr3 = np.array([ytr.iloc[i+LB-1] for i in range(len(Xtr_s)-LB+1)])
Xv3, yv3 = seq(Xv_s, yv, Xtr_s, ytr); Xte3, _ = seq(Xte_s, yte, Xv_s, yv)
pesos = compute_class_weight('balanced', classes=np.unique(ytr3), y=ytr3); cw = {0: pesos[0], 1: pesos[1]}
bv, best = -2, None
for s in range(1, N_SEEDS+1):
    tf.keras.backend.clear_session(); os.environ['PYTHONHASHSEED'] = str(s); random.seed(s); np.random.seed(s); tf.random.set_seed(s)
    mdl = construir(len(top12)); es = EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True)
    mdl.fit(Xtr3, ytr3, epochs=EPOCHS, batch_size=BS, validation_data=(Xv3, yv3), shuffle=False, callbacks=[es], class_weight=cw, verbose=0)
    vm = matthews_corrcoef(yv3, (mdl.predict(Xv3, verbose=0).flatten() > 0.5).astype(int))
    if vm > bv: bv, best = vm, mdl
pnv = best.predict(Xv3, verbose=0).flatten(); pnt = best.predict(Xte3, verbose=0).flatten()
yv_a = yv.values; bw, bm = 0.5, -2
for w in np.round(np.arange(0, 1.01, 0.1), 2):
    m = matthews_corrcoef(yv_a, ((w*prv + (1-w)*pnv) > 0.5).astype(int))
    if m > bm: bm, bw = m, w
proba = bw*prt + (1-bw)*pnt; yte_a = yte.values; p = (proba > 0.5).astype(int)
dt = time.perf_counter() - t_ini

res = {'Indice': NOMES[INDICE], 'Campeao': CAMPEAO, 'Horizonte': H,
       'Accuracy': round(accuracy_score(yte_a, p), 4), 'Precision': round(precision_score(yte_a, p, zero_division=0), 4),
       'Recall': round(recall_score(yte_a, p, zero_division=0), 4), 'F1': round(f1_score(yte_a, p, zero_division=0), 4),
       'AUC_ROC': round(roc_auc_score(yte_a, proba), 4), 'MCC': round(matthews_corrcoef(yte_a, p), 4),
       'Kappa': round(cohen_kappa_score(yte_a, p), 4), 'Peso_RF': bw, 'N_teste': len(yte_a), 'Subida_%': round(100*yte_a.mean(), 1)}
print(f"[t+{H}] Acc {res['Accuracy']:.4f} | AUC {res['AUC_ROC']:.4f} | MCC {res['MCC']:+.4f} | Kappa {res['Kappa']:+.4f} | RF {bw*100:.0f}% | {dt:.1f}s", flush=True)

os.makedirs("results/multihorizonte", exist_ok=True)
CSV = "results/multihorizonte/resultados_multihorizonte.csv"
if os.path.exists(CSV):
    tab = pd.read_csv(CSV)
    tab = tab[~((tab['Indice'] == NOMES[INDICE]) & (tab['Horizonte'] == H))]
    tab = pd.concat([tab, pd.DataFrame([res])], ignore_index=True)
else:
    tab = pd.DataFrame([res])
tab = tab.sort_values(['Indice', 'Horizonte']).reset_index(drop=True)
tab.to_csv(CSV, index=False)
print(f"[OK] {CSV} atualizado.")
