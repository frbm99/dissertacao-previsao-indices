"""Etapa 6 — Custo computacional (RQ3): tempos de treino e inferência por família de
modelo, medidos em CPU num índice representativo.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, matthews_corrcoef
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D, Embedding)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Benchmark de CUSTO COMPUTACIONAL (RQ3): tempo de treino e inferencia por familia de modelo.
# Indice representativo (custo ~igual entre índices: datasets do mesmo tamanho). Tudo em CPU.
DATASET = "data/SP500_Master_Dataset.csv"
LOOKBACK = 20; N_FEATURES_SEL = 12; EPOCHS = 60
mcc_scorer = make_scorer(matthews_corrcoef)
print(f"Benchmark de tempos | dataset={DATASET} | device=CPU (TF GPUs={tf.config.list_physical_devices('GPU')})\n")

# --- pré-processamento padrao (igual a todos) ---
dados = pd.read_csv(DATASET, index_col='Date', parse_dates=True)
dados['Target'] = (dados['Close'].shift(-1) > dados['Close']).astype(float)
dados.loc[dados.index[-1], 'Target'] = np.nan
dados.replace([np.inf, -np.inf], np.nan, inplace=True); dados = dados.dropna(); dados['Target'] = dados['Target'].astype(int)
dados['Ouro_Ret'] = dados['Ouro'].pct_change(); dados['Petroleo_Ret'] = dados['Petroleo_WTI'].pct_change()
dados['Dolar_Ret'] = dados['Dolar_Index'].pct_change(); dados['Volume_Ret'] = dados['Volume'].pct_change()
dados['Treasury_10Y_Diff'] = dados['Treasury_10Y'].diff(); dados['TBill_3M_Diff'] = dados['TBill_3M'].diff()
dados['Spread_Diff'] = dados['Spread_10Y_3M'].diff()
colunas_medias = [c for c in dados.columns if c.startswith('SMA_') or c.startswith('EMA_')]
for col in colunas_medias: dados[f"Dist_{col}"] = (dados['Close'] / dados[col]) - 1
if 'MACD' in dados.columns: dados['MACD_Norm'] = dados['MACD'] / dados['Close']
if 'BB_High' in dados.columns: dados['Dist_BB_High'] = (dados['Close'] / dados['BB_High']) - 1
if 'BB_Low' in dados.columns: dados['Dist_BB_Low'] = (dados['Close'] / dados['BB_Low']) - 1
if 'OBV' in dados.columns:
    dados['OBV_Zscore'] = (dados['OBV'] - dados['OBV'].rolling(20, min_periods=1).mean()) / dados['OBV'].rolling(20, min_periods=1).std()
dados.replace([np.inf, -np.inf], np.nan, inplace=True); dados = dados.dropna()
excl = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ouro', 'Petroleo_WTI', 'Dolar_Index',
        'Treasury_10Y', 'TBill_3M', 'Spread_10Y_3M', 'SP500', 'NASDAQ', 'NASDAQ100', 'DJIA', 'Dow_Jones',
        'Sentiment_Score', 'Sentiment_Score_MA3', 'MACD', 'OBV', 'BB_High', 'BB_Low', 'ATR', 'MOM_10'] + colunas_medias
banco = [c for c in dados.columns if c not in excl]; y = dados['Target']
Xfit = dados.loc[:'2019-12-31', banco]; yfit = y.loc[:'2019-12-31']
Ximp = dados.loc['2020-01-01':'2020-12-31', banco]; yimp = y.loc['2020-01-01':'2020-12-31']
selr = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1, class_weight='balanced')
selr.fit(Xfit, yfit)
perm = permutation_importance(selr, Ximp, yimp, scoring=mcc_scorer, n_repeats=10, random_state=42, n_jobs=-1)
top12 = pd.Series(perm.importances_mean, index=banco).sort_values(ascending=False).nlargest(N_FEATURES_SEL).index.tolist()
X = dados[top12]
Xtr, ytr = X.loc[:'2021-12-31'], y.loc[:'2021-12-31']           # arvores: treino <=2021
Xtr2, ytr2 = X.loc[:'2020-12-31'], y.loc[:'2020-12-31']         # DL: treino <=2020
Xv, yv = X.loc['2021-01-01':'2021-12-31'], y.loc['2021-01-01':'2021-12-31']
Xte, yte = X.loc['2022-01-01':'2025-12-31'], y.loc['2022-01-01':'2025-12-31']
sc = Pipeline([('r', RobustScaler()), ('m', MinMaxScaler((-1, 1)))])
Xtr2_s = sc.fit_transform(Xtr2); Xv_s = sc.transform(Xv); Xte_s = sc.transform(Xte)
def seq(cur, ycur, ctx=None, yctx=None):
    if ctx is not None:
        cur = np.vstack([ctx[-(LOOKBACK-1):], cur]); ycur = pd.concat([yctx.iloc[-(LOOKBACK-1):], ycur])
    return np.array([cur[i:i+LOOKBACK] for i in range(len(cur)-LOOKBACK+1)]), \
           np.array([ycur.iloc[i+LOOKBACK-1] for i in range(len(cur)-LOOKBACK+1)])
Xtr3, ytr3 = seq(Xtr2_s, ytr2); Xv3, yv3 = seq(Xv_s, yv, Xtr2_s, ytr2); Xte3, _ = seq(Xte_s, yte, Xv_s, yv)
pesos = compute_class_weight('balanced', classes=np.unique(ytr3), y=ytr3); cw = {0: pesos[0], 1: pesos[1]}
n_te = len(Xte)
print(f"Treino arvores: {len(Xtr)} linhas | Treino DL: {Xtr3.shape} | Teste: {n_te} dias\n")

resultados = []
def reg(nome, tipo, t_train, t_infer, epocas=None):
    resultados.append({'Modelo': nome, 'Tipo': tipo, 't_treino_s': round(t_train, 2),
                       'Epocas': epocas if epocas else '-',
                       't_infer_total_s': round(t_infer, 3),
                       't_infer_por_amostra_ms': round(1000 * t_infer / n_te, 3)})
    print(f"  {nome:24s} treino {t_train:7.2f}s | inferencia {t_infer*1000:7.1f}ms ({1000*t_infer/n_te:.3f} ms/amostra)"
          + (f" | {epocas} epocas" if epocas else ""))

# --- Árvores ---
print("Arvores (1 fit; sem procura):")
m = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, max_features='sqrt',
                           class_weight='balanced', random_state=42, n_jobs=-1)
t0 = time.perf_counter(); m.fit(Xtr, ytr); tt = time.perf_counter() - t0
t0 = time.perf_counter(); m.predict_proba(Xte); ti = time.perf_counter() - t0
reg("Random Forest", "Arvore", tt, ti)

spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      gamma=1, reg_alpha=0.1, reg_lambda=10, scale_pos_weight=spw, objective='binary:logistic',
                      eval_metric='logloss', tree_method='hist', device='cpu', random_state=42, n_jobs=-1)
t0 = time.perf_counter(); m.fit(Xtr, ytr); tt = time.perf_counter() - t0
t0 = time.perf_counter(); m.predict_proba(Xte); ti = time.perf_counter() - t0
reg("XGBoost", "Arvore", tt, ti)

# --- Redes (1 treino com early stopping) ---
def build(family):
    if family in ('LSTM', 'GRU'):
        cam = LSTM if family == 'LSTM' else GRU
        mdl = Sequential([Input(shape=(LOOKBACK, len(top12))), cam(32, activation='tanh'),
                          Dropout(0.2), Dense(1, activation='sigmoid')])
    else:
        nb = 1 if family == 'Transformer (1 bloco)' else 2
        inp = Input(shape=(LOOKBACK, len(top12))); x = Dense(32, activation="relu", kernel_regularizer=l2(1e-4))(inp)
        x = x + Embedding(input_dim=LOOKBACK, output_dim=32)(tf.range(start=0, limit=LOOKBACK, delta=1))
        for _ in range(nb):
            a = Dropout(0.2)(MultiHeadAttention(num_heads=2, key_dim=32)(x, x))
            o = LayerNormalization(epsilon=1e-6)(x + a)
            f = Dense(32, activation="relu", kernel_regularizer=l2(1e-4))(o); f = Dropout(0.2)(f)
            f = Dense(32, kernel_regularizer=l2(1e-4))(f); x = LayerNormalization(epsilon=1e-6)(o + f)
        out = Dense(1, activation='sigmoid')(Dropout(0.2)(GlobalAveragePooling1D()(x)))
        mdl = Model(inp, out)
    mdl.compile(Adam(1e-3), 'binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    return mdl

print("\nRedes (1 treino, early stopping; batch 16 rec. / 32 transf.):")
for fam, bs in [('LSTM', 16), ('GRU', 16), ('Transformer (1 bloco)', 32), ('Deep Transformer (2 blocos)', 32)]:
    tf.keras.backend.clear_session(); random.seed(42); np.random.seed(42); tf.random.set_seed(42)
    mdl = build(fam)
    es = EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True)
    t0 = time.perf_counter()
    h = mdl.fit(Xtr3, ytr3, epochs=EPOCHS, batch_size=bs, validation_data=(Xv3, yv3),
                shuffle=False, callbacks=[es], class_weight=cw, verbose=0)
    tt = time.perf_counter() - t0
    t0 = time.perf_counter(); mdl.predict(Xte3, verbose=0); ti = time.perf_counter() - t0
    reg(fam, "Rede", tt, ti, epocas=len(h.history['loss']))

os.makedirs("results", exist_ok=True)
tab = pd.DataFrame(resultados)
tab.to_csv("results/resultados_tempos.csv", index=False)
print("\n" + tab.to_string(index=False))
print("\n[OK] resultados/resultados_tempos.csv guardado.")
