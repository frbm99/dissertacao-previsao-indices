"""Etapa 3 — Modelo LSTM para previsão da direção diária (t+1) do DJIA.

Divisão cronológica (treino até 2020, validação 2021, teste cego 2022-2025);
arquitetura fixa treinada com múltiplas sementes, selecionando a melhor pelo
MCC da validação; limiar de decisão fixo em 0,5.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Parâmetros do índice (única secção que difere entre os três índices).
DATASET = "data/DJIA_Master_Dataset.csv"
NOME_MODELO = "LSTM - DJIA"
CAMINHO_HTML = "results/LSTM_DJIA.html"
MODEL_PATH = "models/campeao_lstm_djia.keras"

# Orçamento de treino (configurável por ambiente para validar barato)
N_SEEDS = int(os.environ.get('LSTM_SEEDS', 8))
EPOCHS = int(os.environ.get('LSTM_EPOCHS', 60))
LOOKBACK = 20
N_FEATURES_SEL = 12

mcc_scorer = make_scorer(matthews_corrcoef)


# 1. Carregar + engenharia de estacionaridade (igual a RF/XGBoost)
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

# 4. Scaler ajustado SÓ no treino (sem fuga para validação/teste)
scaler = Pipeline([('robust', RobustScaler()), ('minmax', MinMaxScaler(feature_range=(-1, 1)))])
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# 5. Sequências com herança de contexto (val/teste reaproveitam os últimos lookback-1 dias
#    do split anterior -> não perdem amostras e usam história real, sem acesso a informação futura).
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
print(f"Sequencias -> treino:{X_tr3D.shape} val:{X_val3D.shape} teste:{X_te3D.shape}")

pesos = compute_class_weight('balanced', classes=np.unique(y_tr3D), y=y_tr3D)
pesos_das_classes = {0: pesos[0], 1: pesos[1]}

def melhor_limiar_mcc(probs, y_true):
    # Limiar de decisão fixo em 0,5 (comum a todos os modelos): o ajuste do limiar na validação
    # degenerava no teste, por deriva da distribuição de probabilidades entre períodos.
    return 0.5, matthews_corrcoef(y_true, (probs > 0.5).astype(int))

# 6. Procura de seed: seleciona pela VALIDAÇÃO 2021 (nunca pelo teste!)
print(f"\nFase 2: a treinar {N_SEEDS} seeds; selecao pela validacao 2021 (teste intocado)...")
melhor_val_mcc = -2.0
melhor_modelo = None
melhor_seed = None
melhor_thr = 0.5

for semente in range(1, N_SEEDS + 1):
    tf.keras.backend.clear_session()
    os.environ['PYTHONHASHSEED'] = str(semente)
    random.seed(semente); np.random.seed(semente); tf.random.set_seed(semente)

    modelo = Sequential([
        Input(shape=(LOOKBACK, len(features_sel))),
        LSTM(units=32, activation='tanh'),
        Dropout(0.2),
        Dense(units=1, activation='sigmoid')
    ])
    modelo.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy',
                   metrics=[tf.keras.metrics.AUC(name='auc')])
    paragem = EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True)
    modelo.fit(X_tr3D, y_tr3D, epochs=EPOCHS, batch_size=16,
               validation_data=(X_val3D, y_val3D), shuffle=False,
               callbacks=[paragem], class_weight=pesos_das_classes, verbose=0)

    val_probs = modelo.predict(X_val3D, verbose=0).flatten()
    thr, val_mcc = melhor_limiar_mcc(val_probs, y_val3D)
    print(f"  [seed {semente}] MCC validacao: {val_mcc:.4f} (thr={thr:.3f})")
    if val_mcc > melhor_val_mcc:
        melhor_val_mcc, melhor_modelo, melhor_seed, melhor_thr = val_mcc, modelo, semente, thr

print(f"\n[CAMPEAO] seed {melhor_seed} (MCC validacao {melhor_val_mcc:.4f}, thr={melhor_thr:.3f})")
melhor_modelo.save(MODEL_PATH)
print(f"[OK] Modelo guardado em {MODEL_PATH}")

# 7. Avaliação final CEGA no teste 2022+ (limiar vindo da validação)
probabilidades = melhor_modelo.predict(X_te3D, verbose=0).flatten()
previsoes = (probabilidades > melhor_thr).astype(int)

acc = accuracy_score(y_te3D, previsoes) * 100
prec = precision_score(y_te3D, previsoes, zero_division=0) * 100
rec = recall_score(y_te3D, previsoes, zero_division=0) * 100
f1 = f1_score(y_te3D, previsoes, zero_division=0) * 100
auc_roc = roc_auc_score(y_te3D, probabilidades)
mcc = matthews_corrcoef(y_te3D, previsoes)
kappa = cohen_kappa_score(y_te3D, previsoes)
cm = confusion_matrix(y_te3D, previsoes)
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

print(f"Accuracy:  {acc:.2f}%   AUC-ROC: {auc_roc:.4f}   MCC: {mcc:.4f}   Kappa: {kappa:.4f}")
print(f"Precision: {prec:.2f}%  Recall: {rec:.2f}%  F1: {f1:.2f}%")
print("Matriz de Confusao:"); print(cm)

# 8. HTML de uma linha (igual aos outros modelos)
os.makedirs("results", exist_ok=True)
params_str = f"seed={melhor_seed}, lookback={LOOKBACK}, units=32, dropout=0.2, batch=16, threshold={melhor_thr:.3f}"
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
