"""Etapa 4 — Ensemble híbrido Deep Transformer+Random Forest (soft-voting) para o NASDAQ-100.

Combina as probabilidades dos dois ramos por média ponderada; o peso de fusão é
ajustado pelo MCC na validação de 2021 e o teste (2022-2025) permanece cego.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, cohen_kappa_score, make_scorer)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D, Embedding)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Parâmetros do índice (única secção que difere entre os três índices).
DATASET = "data/NASDAQ100_Master_Dataset.csv"
NOME_MODELO = "Ensemble DeepTransformer+RandomForest - NASDAQ-100"
CAMINHO_HTML = "results/Ensemble_DeepTransf_RF_NASDAQ100.html"
DT_PATH = "models/campeao_ensDTR_deeptransf_nasdaq100.keras"
RF_PATH = "models/campeao_ensDTR_rf_nasdaq100.pkl"

N_SEEDS = int(os.environ.get('ENS_SEEDS', 10))
EPOCHS = int(os.environ.get('ENS_EPOCHS', 60))
LOOKBACK = 20
N_FEATURES_SEL = 12
NUM_BLOCKS = 2

mcc_scorer = make_scorer(matthews_corrcoef)


# 1. Carregar + engenharia de estacionaridade (igual a todos os outros modelos)
print(f"A carregar o {DATASET}...")
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

colunas_excluir = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume',
                   'Ouro', 'Petroleo_WTI', 'Dolar_Index', 'Treasury_10Y', 'TBill_3M', 'Spread_10Y_3M',
                   'SP500', 'NASDAQ', 'NASDAQ100', 'DJIA', 'Dow_Jones',
                   'Sentiment_Score', 'Sentiment_Score_MA3',
                   'MACD', 'OBV', 'BB_High', 'BB_Low', 'ATR', 'MOM_10'] + colunas_medias
banco_features = [col for col in dados.columns if col not in colunas_excluir]

X_raw = dados[banco_features]
y = dados['Target']

# 2. Seleção de features por Permutation Importance (seletor <=2019, avaliado em 2020)
print("\nFase 1: selecao de features (Permutation Importance, sem espreitar 2021)...")
X_fit = X_raw.loc[:'2019-12-31']; y_fit = y.loc[:'2019-12-31']
X_imp = X_raw.loc['2020-01-01':'2020-12-31']; y_imp = y.loc['2020-01-01':'2020-12-31']
seletor = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42,
                                 n_jobs=-1, class_weight='balanced')
seletor.fit(X_fit, y_fit)
perm = permutation_importance(seletor, X_imp, y_imp, scoring=mcc_scorer,
                              n_repeats=10, random_state=42, n_jobs=-1)
importancias = pd.Series(perm.importances_mean, index=banco_features).sort_values(ascending=False)
features_sel = importancias.nlargest(N_FEATURES_SEL).index.tolist()
print(f"Top {N_FEATURES_SEL} features: {features_sel}")

# 3. Divisão temporal de 3 vias: treino<=2020 | validação 2021 | teste 2022+
X = dados[features_sel]
X_train = X.loc[:'2020-12-31']; y_train = y.loc[:'2020-12-31']
X_val = X.loc['2021-01-01':'2021-12-31']; y_val = y.loc['2021-01-01':'2021-12-31']
X_test = X.loc['2022-01-01':'2025-12-31']; y_test = y.loc['2022-01-01':'2025-12-31']

# ====================================================================
# RAMO A: Random Forest (tabular, sem escala, class_weight balanced). Treina <=2020.
# ====================================================================
print("\nRamo Random Forest (class_weight='balanced')...")
modelo_rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5,
                                   max_features='sqrt', class_weight='balanced',
                                   random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)
prob_rf_val = modelo_rf.predict_proba(X_val)[:, 1]
prob_rf_test = modelo_rf.predict_proba(X_test)[:, 1]

# ====================================================================
# RAMO B: Deep Transformer (sequências escaladas). Scaler só no treino; seed search pela validação.
# ====================================================================
scaler = Pipeline([('robust', RobustScaler()), ('minmax', MinMaxScaler(feature_range=(-1, 1)))])
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

def make_sequences(cur_scaled, y_cur, lookback, ctx_scaled=None, y_ctx=None):
    if ctx_scaled is not None:
        ctx = ctx_scaled[-(lookback - 1):]
        X_comb = np.vstack([ctx, cur_scaled])
        y_comb = pd.concat([y_ctx.iloc[-(lookback - 1):], y_cur])
    else:
        X_comb, y_comb = cur_scaled, y_cur
    X_seq, y_seq = [], []
    for i in range(len(X_comb) - lookback + 1):
        X_seq.append(X_comb[i:i + lookback])
        y_seq.append(y_comb.iloc[i + lookback - 1])
    return np.array(X_seq), np.array(y_seq)

X_tr3D, y_tr3D = make_sequences(X_train_s, y_train, LOOKBACK)
X_val3D, y_val3D = make_sequences(X_val_s, y_val, LOOKBACK, X_train_s, y_train)
X_te3D, y_te3D = make_sequences(X_test_s, y_test, LOOKBACK, X_val_s, y_val)
print(f"Sequencias DT -> treino:{X_tr3D.shape} val:{X_val3D.shape} teste:{X_te3D.shape}")
print(f"Alinhamento val: rf={len(prob_rf_val)} dt={len(X_val3D)} | teste: rf={len(prob_rf_test)} dt={len(X_te3D)}")

pesos = compute_class_weight('balanced', classes=np.unique(y_tr3D), y=y_tr3D)
pesos_das_classes = {0: pesos[0], 1: pesos[1]}

def bloco_transformer(x, embed_dim, num_heads, dropout):
    atencao = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    atencao = Dropout(dropout)(atencao)
    out = LayerNormalization(epsilon=1e-6)(x + atencao)
    ffn = Dense(embed_dim, activation="relu", kernel_regularizer=l2(1e-4))(out)
    ffn = Dropout(dropout)(ffn)
    ffn = Dense(embed_dim, kernel_regularizer=l2(1e-4))(ffn)
    return LayerNormalization(epsilon=1e-6)(out + ffn)

def construir_deep_transformer(lookback, num_features, num_blocks=2, embed_dim=32, num_heads=2, dropout=0.2):
    inputs = Input(shape=(lookback, num_features))
    x = Dense(embed_dim, activation="relu", kernel_regularizer=l2(1e-4))(inputs)
    posicoes = tf.range(start=0, limit=lookback, delta=1)
    pos_embedding = Embedding(input_dim=lookback, output_dim=embed_dim)(posicoes)
    x = x + pos_embedding
    for _ in range(num_blocks):
        x = bloco_transformer(x, embed_dim, num_heads, dropout)
    flat = GlobalAveragePooling1D()(x)
    flat = Dropout(dropout)(flat)
    outputs = Dense(1, activation="sigmoid")(flat)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

print(f"\nRamo Deep Transformer ({NUM_BLOCKS} blocos): a treinar {N_SEEDS} seeds; melhor pela validacao 2021...")
melhor_val_mcc = -2.0
melhor_dt = None
melhor_seed = None
for semente in range(1, N_SEEDS + 1):
    tf.keras.backend.clear_session()
    os.environ['PYTHONHASHSEED'] = str(semente)
    random.seed(semente); np.random.seed(semente); tf.random.set_seed(semente)
    modelo_dt = construir_deep_transformer(LOOKBACK, len(features_sel), NUM_BLOCKS)
    paragem = EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True)
    modelo_dt.fit(X_tr3D, y_tr3D, epochs=EPOCHS, batch_size=32,
                  validation_data=(X_val3D, y_val3D), shuffle=False,
                  callbacks=[paragem], class_weight=pesos_das_classes, verbose=0)
    vp = modelo_dt.predict(X_val3D, verbose=0).flatten()
    vm = matthews_corrcoef(y_val3D, (vp > 0.5).astype(int))
    print(f"  [DT seed {semente}] MCC validacao: {vm:.4f}")
    if vm > melhor_val_mcc:
        melhor_val_mcc, melhor_dt, melhor_seed = vm, modelo_dt, semente

prob_dt_val = melhor_dt.predict(X_val3D, verbose=0).flatten()
prob_dt_test = melhor_dt.predict(X_te3D, verbose=0).flatten()
print(f"DeepTransformer campeao: seed {melhor_seed} (MCC val {melhor_val_mcc:.4f})")

# ====================================================================
# FUSÃO: peso afinado na VALIDAÇÃO 2021 (NUNCA no teste). Limiar fixo 0.5.
# ====================================================================
print("\nA afinar o peso da fusao na validacao 2021...")
melhor_peso_rf, melhor_fus_mcc = 0.5, -2.0
for peso_rf in np.round(np.arange(0.0, 1.01, 0.1), 2):
    fus_val = peso_rf * prob_rf_val + (1 - peso_rf) * prob_dt_val
    m = matthews_corrcoef(y_val3D, (fus_val > 0.5).astype(int))
    if m > melhor_fus_mcc:
        melhor_fus_mcc, melhor_peso_rf = m, peso_rf
peso_dt = 1 - melhor_peso_rf
print(f"-> Peso ideal (validacao): RandomForest={melhor_peso_rf*100:.0f}% | DeepTransformer={peso_dt*100:.0f}% (MCC val {melhor_fus_mcc:.4f})")

melhor_dt.save(DT_PATH); joblib.dump(modelo_rf, RF_PATH)
print(f"[OK] Modelos guardados: {DT_PATH} + {RF_PATH}")

# 7. Avaliação final CEGA no teste 2022+
probabilidades = melhor_peso_rf * prob_rf_test + peso_dt * prob_dt_test
previsoes = (probabilidades > 0.5).astype(int)

acc = accuracy_score(y_te3D, previsoes) * 100
prec = precision_score(y_te3D, previsoes, zero_division=0) * 100
rec = recall_score(y_te3D, previsoes, zero_division=0) * 100
f1 = f1_score(y_te3D, previsoes, zero_division=0) * 100
auc_roc = roc_auc_score(y_te3D, probabilidades)
mcc = matthews_corrcoef(y_te3D, previsoes)
kappa = cohen_kappa_score(y_te3D, previsoes)
cm = confusion_matrix(y_te3D, previsoes)
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

mcc_rf = matthews_corrcoef(y_te3D, (prob_rf_test > 0.5).astype(int))
mcc_dt = matthews_corrcoef(y_te3D, (prob_dt_test > 0.5).astype(int))

print(f"Accuracy:  {acc:.2f}%   AUC-ROC: {auc_roc:.4f}   MCC: {mcc:.4f}   Kappa: {kappa:.4f}")
print(f"Precision: {prec:.2f}%  Recall: {rec:.2f}%  F1: {f1:.2f}%")
print(f"(referencia teste: MCC RandomForest={mcc_rf:.4f} | MCC DeepTransformer={mcc_dt:.4f})")
print("Matriz de Confusao:"); print(cm)

# 8. HTML de uma linha (igual aos outros modelos)
os.makedirs("results", exist_ok=True)
params_str = (f"fusao RF={melhor_peso_rf*100:.0f}%/DT={peso_dt*100:.0f}%, "
              f"dt_seed={melhor_seed}, blocos={NUM_BLOCKS}, lookback={LOOKBACK}, rf=depth10/300, threshold=0.50")
features_str = ", ".join(features_sel)
nome_campeao = f"Permutation Top {N_FEATURES_SEL}"

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
    <td>{nome_campeao}</td><td class="txt">{features_str}</td><td class="txt">{params_str}</td>
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
