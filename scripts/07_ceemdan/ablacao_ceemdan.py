"""Fase 3 (Etapa 8) — Ablação controlada da decomposição CEEMDAN nos modelos campeões.

Espelha a ablação de sentimento (Fase 2): treina o ensemble campeão de cada índice a acrescentar
as 10 features de IMF causais (geradas por scripts/07_ceemdan/gerar_imfs_ceemdan.py) e mede o
efeito por MCC no teste. As IMFs são ANEXADAS ao Top-12 (não competem na seleção): entram como
colunas extra no ramo Random Forest e como canais extra na sequência 3D do ramo rede.

Quatro variantes por campeão (as duas primeiras reaproveitadas da Fase 2):
  base            -> Top-12
  sentimento      -> Top-12 + sentimento
  ceemdan         -> Top-12 + 10 IMFs
  ceemdan+sent    -> Top-12 + 10 IMFs + sentimento
Escreve results/resultados_ceemdan.csv. Parametrizável por CE_INDICE (sp500/djia/nasdaq100).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, matthews_corrcoef, make_scorer
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D, Embedding)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

INDICE = os.environ.get('CE_INDICE', 'sp500')
N_SEEDS = int(os.environ.get('CE_SEEDS', 8)); EPOCHS = int(os.environ.get('CE_EPOCHS', 60))
N_FEATURES_SEL = 12; SENT = ['Sentiment_Score', 'Sentiment_Score_MA3']
K_IMF = 10; IMF_COLS = [f"IMF_{k + 1}" for k in range(K_IMF)]
DATASETS = {'sp500': 'data/SP500_Multimodal_Dataset.csv', 'djia': 'data/DJIA_Multimodal_Dataset.csv', 'nasdaq100': 'data/NASDAQ100_Multimodal_Dataset.csv'}
NOMES = {'sp500': 'SP500', 'djia': 'DJIA', 'nasdaq100': 'NASDAQ-100'}
ETIQ = {'deeptransf': 'DeepTransformer', 'lstm': 'LSTM', 'gru': 'GRU'}
# Campeões de validação (fixos, iguais à Fase 2): (rede, usa_HPO)
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
print(f"=== CEEMDAN {NOMES[INDICE]} — {CAMPEAO} | lookback={LB} units={UNITS} dropout={DROP} lr={LR} batch={BS} ===\n")

dados = pd.read_csv(DATASETS[INDICE], index_col='Date', parse_dates=True)
dados['Target'] = (dados['Close'].shift(-1) > dados['Close']).astype(float)
dados.loc[dados.index[-1], 'Target'] = np.nan
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
# Top-12 selecionado APENAS das features originais (as IMFs são anexadas, não competem)
Xf = dados.loc[:'2019-12-31', banco]; yf = y.loc[:'2019-12-31']
Xi = dados.loc['2020-01-01':'2020-12-31', banco]; yi = y.loc['2020-01-01':'2020-12-31']
sel = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1, class_weight='balanced').fit(Xf, yf)
perm = permutation_importance(sel, Xi, yi, scoring=mcc_scorer, n_repeats=10, random_state=42, n_jobs=-1)
top12 = pd.Series(perm.importances_mean, index=banco).sort_values(ascending=False).nlargest(N_FEATURES_SEL).index.tolist()

# Anexar as IMFs causais (alinhadas por data)
imfs = pd.read_csv(f"data/ceemdan_imfs_{INDICE}.csv", index_col=0, parse_dates=True)
dados = dados.join(imfs[IMF_COLS])
if dados[IMF_COLS].isna().any().any():
    faltam = int(dados[IMF_COLS].isna().any(axis=1).sum())
    raise SystemExit(f"ERRO: {faltam} linhas sem IMFs (geração CEEMDAN incompleta para {INDICE}).")

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

def avaliar(feats, tag):
    X = dados[feats]
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
        mdl = construir(len(feats)); es = EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True)
        mdl.fit(Xtr3, ytr3, epochs=EPOCHS, batch_size=BS, validation_data=(Xv3, yv3), shuffle=False, callbacks=[es], class_weight=cw, verbose=0)
        vm = matthews_corrcoef(yv3, (mdl.predict(Xv3, verbose=0).flatten() > 0.5).astype(int))
        if vm > bv: bv, best = vm, mdl
    pnv = best.predict(Xv3, verbose=0).flatten(); pnt = best.predict(Xte3, verbose=0).flatten()
    yv_a = yv.values; bw, bm = 0.5, -2
    for w in np.round(np.arange(0, 1.01, 0.1), 2):
        m = matthews_corrcoef(yv_a, ((w*prv + (1-w)*pnv) > 0.5).astype(int))
        if m > bm: bm, bw = m, w
    proba = bw*prt + (1-bw)*pnt; yte_a = yte.values; mask = np.asarray(Xte.index <= pd.Timestamp('2024-03-04'))
    def met(mk): p = (proba[mk] > 0.5).astype(int); return matthews_corrcoef(yte_a[mk], p), roc_auc_score(yte_a[mk], proba[mk])
    mf, ms = met(np.ones(len(proba), bool)), met(mask)
    print(f"[{tag}] (RF {bw*100:.0f}%) FULL MCC {mf[0]:+.4f} AUC {mf[1]:.4f} | SUB MCC {ms[0]:+.4f}", flush=True)
    return mf[0], ms[0]

# Reaproveitar base e sentimento da Fase 2
FASE2 = "results/resultados_multimodal_campeoes.csv"
f2 = pd.read_csv(FASE2); f2 = f2[f2['Indice'] == NOMES[INDICE]].iloc[0]
base_full, base_sub = float(f2['MCC_uni_full']), float(f2['MCC_uni_sub'])
sent_full, sent_sub = float(f2['MCC_multi_full']), float(f2['MCC_multi_sub'])

ce_full, ce_sub = avaliar(top12 + IMF_COLS, "CEEMDAN")
ces_full, ces_sub = avaliar(top12 + IMF_COLS + SENT, "CEEMDAN+SENT")

print(f"\n  {NOMES[INDICE]} — {CAMPEAO} — efeito da CEEMDAN (MCC teste completo)")
print(f"  base {base_full:+.4f} | sent {sent_full:+.4f} | ceemdan {ce_full:+.4f} (delta {ce_full-base_full:+.4f}) | ceemdan+sent {ces_full:+.4f} (delta {ces_full-base_full:+.4f})")

os.makedirs("results", exist_ok=True)
CSV = "results/resultados_ceemdan.csv"
linha = {'Indice': NOMES[INDICE], 'Campeao': CAMPEAO,
         'MCC_base': round(base_full, 4), 'MCC_sent': round(sent_full, 4),
         'MCC_ceemdan': round(ce_full, 4), 'MCC_ceemdan_sent': round(ces_full, 4),
         'Delta_sent': round(sent_full - base_full, 4),
         'Delta_ceemdan': round(ce_full - base_full, 4),
         'Delta_ceemdan_sent': round(ces_full - base_full, 4),
         'MCC_base_sub': round(base_sub, 4), 'MCC_sent_sub': round(sent_sub, 4),
         'MCC_ceemdan_sub': round(ce_sub, 4), 'MCC_ceemdan_sent_sub': round(ces_sub, 4),
         'Delta_ceemdan_sub': round(ce_sub - base_sub, 4),
         'Delta_ceemdan_sent_sub': round(ces_sub - base_sub, 4)}
if os.path.exists(CSV):
    tab = pd.read_csv(CSV); tab = tab[tab['Indice'] != NOMES[INDICE]]; tab = pd.concat([tab, pd.DataFrame([linha])], ignore_index=True)
else:
    tab = pd.DataFrame([linha])
tab.to_csv(CSV, index=False); print(f"\n[OK] {CSV} atualizado.")
