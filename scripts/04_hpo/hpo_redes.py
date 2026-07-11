"""Etapa 5 — Otimização de hiperparâmetros das redes neuronais (Keras Tuner / Hyperband).

Procura lookback (loop externo, por reconstrução das sequências), units, dropout,
learning rate e batch size, com objetivo val_auc na validação de 2021; o teste
(2022-2025) nunca participa na procura. Parametrizável por variáveis de ambiente:
HPO_FAMILY (lstm|gru|transformer|deeptransf) e HPO_INDICE (sp500|djia|nasdaq100).
A configuração vencedora é registada em results/hpo_best_configs.json.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import shutil
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D, Embedding)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# ====================================================================
# CONFIG (via ambiente): HPO_FAMILY in {lstm,gru,transformer,deeptransf}; HPO_INDICE in {sp500,djia,nasdaq100}
# ====================================================================
FAMILY = os.environ.get('HPO_FAMILY', 'lstm')
INDICE = os.environ.get('HPO_INDICE', 'sp500')

DATASETS = {'sp500': 'data/SP500_Master_Dataset.csv', 'djia': 'data/DJIA_Master_Dataset.csv',
            'nasdaq100': 'data/NASDAQ100_Master_Dataset.csv'}
NOMES_IDX = {'sp500': 'SP500', 'djia': 'DJIA', 'nasdaq100': 'NASDAQ-100'}
FAM_NOME = {'lstm': 'LSTM', 'gru': 'GRU', 'transformer': 'Transformer 1-bloco',
            'deeptransf': 'Deep Transformer 2-blocos'}
NUM_BLOCKS = {'transformer': 1, 'deeptransf': 2}

DATASET = DATASETS[INDICE]
NOME_MODELO = f"{FAM_NOME[FAMILY]} (HPO) - {NOMES_IDX[INDICE]}"
CAMINHO_HTML = f"results/HPO_{FAMILY}_{INDICE}.html"
MODEL_PATH = f"models/campeao_hpo_{FAMILY}_{INDICE}.keras"
HPO_DIR = f"hpo_tuner/{FAMILY}_{INDICE}"
CONFIG_JSON = "results/hpo_best_configs.json"

N_FEATURES_SEL = 12
LOOKBACKS = [int(x) for x in os.environ.get('HPO_LOOKBACKS', '5,10,20,40').split(',')]
HPO_MAX_EPOCHS = int(os.environ.get('HPO_MAX_EPOCHS', 40))
HPO_FACTOR = int(os.environ.get('HPO_FACTOR', 3))
FINAL_EPOCHS = int(os.environ.get('HPO_FINAL_EPOCHS', 80))

mcc_scorer = make_scorer(matthews_corrcoef)

print(f"  family={FAMILY} indice={INDICE} lookbacks={LOOKBACKS} max_epochs={HPO_MAX_EPOCHS}")
print(f"  GPUs visiveis ao TF: {tf.config.list_physical_devices('GPU')}")

# 1. Carregar + engenharia de estacionaridade (igual a todos os outros modelos)
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

X_raw = dados[banco_features]; y = dados['Target']

# 2. Seleção de features por Permutation Importance (seletor <=2019, avaliado em 2020)
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

# 3. Split temporal de 3 vias + scaler ajustado SÓ no treino
X = dados[features_sel]
X_train = X.loc[:'2020-12-31']; y_train = y.loc[:'2020-12-31']
X_val = X.loc['2021-01-01':'2021-12-31']; y_val = y.loc['2021-01-01':'2021-12-31']
X_test = X.loc['2022-01-01':'2025-12-31']; y_test = y.loc['2022-01-01':'2025-12-31']
scaler = Pipeline([('robust', RobustScaler()), ('minmax', MinMaxScaler(feature_range=(-1, 1)))])
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
N_FEAT = X_train_s.shape[1]

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

# ====================================================================
# Construtores de arquitetura (units=unidades recorrentes OU embed_dim do transformer)
# ====================================================================
def _bloco_transformer(x, embed_dim, num_heads, dropout):
    at = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    at = Dropout(dropout)(at)
    out = LayerNormalization(epsilon=1e-6)(x + at)
    ffn = Dense(embed_dim, activation="relu", kernel_regularizer=l2(1e-4))(out)
    ffn = Dropout(dropout)(ffn)
    ffn = Dense(embed_dim, kernel_regularizer=l2(1e-4))(ffn)
    return LayerNormalization(epsilon=1e-6)(out + ffn)

def construir_modelo(family, lookback, n_features, units, dropout, lr):
    if family in ('lstm', 'gru'):
        camada = LSTM if family == 'lstm' else GRU
        model = Sequential([
            Input(shape=(lookback, n_features)),
            camada(units=units, activation='tanh'),
            Dropout(dropout),
            Dense(1, activation='sigmoid')
        ])
    else:  # transformer (1 bloco) / deeptransf (2 blocos); units = embed_dim
        embed_dim = units
        inputs = Input(shape=(lookback, n_features))
        x = Dense(embed_dim, activation="relu", kernel_regularizer=l2(1e-4))(inputs)
        pos = Embedding(input_dim=lookback, output_dim=embed_dim)(tf.range(start=0, limit=lookback, delta=1))
        x = x + pos
        for _ in range(NUM_BLOCKS[family]):
            x = _bloco_transformer(x, embed_dim, num_heads=2, dropout=dropout)
        flat = GlobalAveragePooling1D()(x)
        flat = Dropout(dropout)(flat)
        outputs = Dense(1, activation='sigmoid')(flat)
        model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

class HyperSeq(kt.HyperModel):
    def __init__(self, family, lookback, n_features):
        self.family = family; self.lookback = lookback; self.n_features = n_features
    def build(self, hp):
        units = hp.Choice('units', [32, 64])
        dropout = hp.Choice('dropout', [0.2, 0.3])
        lr = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
        return construir_modelo(self.family, self.lookback, self.n_features, units, dropout, lr)
    def fit(self, hp, model, *args, **kwargs):
        kwargs['batch_size'] = hp.Choice('batch_size', [16, 32, 64])
        return model.fit(*args, **kwargs)

# 4. Procura: um Hyperband por lookback; comparam-se pelo melhor val_auc (validação 2021).
if os.path.isdir(HPO_DIR):
    shutil.rmtree(HPO_DIR)

resultados_lb = []
for lb in LOOKBACKS:
    tf.keras.backend.clear_session()
    random.seed(42); np.random.seed(42); tf.random.set_seed(42)
    X_tr3D, y_tr3D = make_sequences(X_train_s, y_train, lb)
    X_val3D, y_val3D = make_sequences(X_val_s, y_val, lb, X_train_s, y_train)
    pesos = compute_class_weight('balanced', classes=np.unique(y_tr3D), y=y_tr3D)
    cw = {0: pesos[0], 1: pesos[1]}
    tuner = kt.Hyperband(
        HyperSeq(FAMILY, lb, N_FEAT),
        objective=kt.Objective('val_auc', direction='max'),
        max_epochs=HPO_MAX_EPOCHS, factor=HPO_FACTOR,
        directory=HPO_DIR, project_name=f'lb{lb}', overwrite=True, seed=42)
    es = EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True)
    tuner.search(X_tr3D, y_tr3D, validation_data=(X_val3D, y_val3D),
                 epochs=HPO_MAX_EPOCHS, class_weight=cw, shuffle=False, callbacks=[es], verbose=0)
    best_trial = tuner.oracle.get_best_trials(1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]
    resultados_lb.append({'lookback': lb, 'val_auc': float(best_trial.score), 'hp': best_hp.values})
    print(f"[lookback {lb:>2}] melhor val_auc={best_trial.score:.4f}  hp={best_hp.values}")

df_lb = pd.DataFrame(resultados_lb).sort_values('val_auc', ascending=False)
campeao = df_lb.iloc[0]
best_lb = int(campeao['lookback']); best = campeao['hp']
print(f"\n[CONFIG VENCEDORA] lookback={best_lb} | units={best['units']} dropout={best['dropout']} "
      f"lr={best['learning_rate']} batch={best['batch_size']} | val_auc={campeao['val_auc']:.4f}")

# 5. Re-treino final com a config vencedora + avaliação CEGA no teste 2022+
tf.keras.backend.clear_session()
random.seed(42); np.random.seed(42); tf.random.set_seed(42)
X_tr3D, y_tr3D = make_sequences(X_train_s, y_train, best_lb)
X_val3D, y_val3D = make_sequences(X_val_s, y_val, best_lb, X_train_s, y_train)
X_te3D, y_te3D = make_sequences(X_test_s, y_test, best_lb, X_val_s, y_val)
pesos = compute_class_weight('balanced', classes=np.unique(y_tr3D), y=y_tr3D)
cw = {0: pesos[0], 1: pesos[1]}
modelo = construir_modelo(FAMILY, best_lb, N_FEAT, best['units'], best['dropout'], best['learning_rate'])
es = EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True)
modelo.fit(X_tr3D, y_tr3D, epochs=FINAL_EPOCHS, batch_size=best['batch_size'],
           validation_data=(X_val3D, y_val3D), shuffle=False, callbacks=[es], class_weight=cw, verbose=0)
modelo.save(MODEL_PATH)

probabilidades = modelo.predict(X_te3D, verbose=0).flatten()
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
print(f"Accuracy:  {acc:.2f}%   AUC-ROC: {auc_roc:.4f}   MCC: {mcc:.4f}   Kappa: {kappa:.4f}")
print(f"Precision: {prec:.2f}%  Recall: {rec:.2f}%  F1: {f1:.2f}%")
print("Matriz de Confusao:"); print(cm)

# 6. Guardar config vencedora no JSON partilhado (para os ensembles reutilizarem)
configs = {}
if os.path.exists(CONFIG_JSON):
    try:
        with open(CONFIG_JSON) as f:
            configs = json.load(f)
    except Exception:
        configs = {}
configs[f"{FAMILY}_{INDICE}"] = {
    'family': FAMILY, 'indice': INDICE, 'lookback': best_lb,
    'units': int(best['units']), 'dropout': float(best['dropout']),
    'learning_rate': float(best['learning_rate']), 'batch_size': int(best['batch_size']),
    'val_auc': float(campeao['val_auc']), 'test_mcc': float(mcc), 'test_auc': float(auc_roc)
}
with open(CONFIG_JSON, 'w') as f:
    json.dump(configs, f, indent=2)
print(f"[OK] Config guardada em {CONFIG_JSON} | modelo em {MODEL_PATH}")

# 6b. Consolidado em resultados/resultados_hpo.csv (+ .html): 1 linha por familia+índice
os.makedirs("results", exist_ok=True)
CSV_HPO = "results/resultados_hpo.csv"
linha = {
    'Modelo': FAM_NOME[FAMILY], 'Indice': NOMES_IDX[INDICE], 'Lookback': best_lb,
    'Units': int(best['units']), 'Dropout': float(best['dropout']),
    'LearningRate': float(best['learning_rate']), 'BatchSize': int(best['batch_size']),
    'Val_AUC': round(float(campeao['val_auc']), 4),
    'Accuracy_%': round(acc, 2), 'AUC_ROC': round(auc_roc, 4), 'Precision_%': round(prec, 2),
    'Recall_%': round(rec, 2), 'F1_%': round(f1, 2), 'MCC': round(mcc, 4), 'Kappa': round(kappa, 4),
    'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP)
}
if os.path.exists(CSV_HPO):
    tab = pd.read_csv(CSV_HPO)
    tab = tab[~((tab['Modelo'] == FAM_NOME[FAMILY]) & (tab['Indice'] == NOMES_IDX[INDICE]))]
    tab = pd.concat([tab, pd.DataFrame([linha])], ignore_index=True)
else:
    tab = pd.DataFrame([linha])
ordem_fam = {'LSTM': 0, 'GRU': 1, 'Transformer 1-bloco': 2, 'Deep Transformer 2-blocos': 3}
tab['_ord'] = tab['Modelo'].map(ordem_fam).fillna(99)
tab = tab.sort_values(['_ord', 'Indice']).drop(columns='_ord')
tab.to_csv(CSV_HPO, index=False)

ths = "".join(f"<th>{c}</th>" for c in tab.columns)
trs = "".join("<tr>" + "".join(f"<td>{r[c]}</td>" for c in tab.columns) + "</tr>" for _, r in tab.iterrows())
html_cons = f"""<!DOCTYPE html><html lang="pt"><head><meta charset="UTF-8"><title>Resultados HPO</title>
<style> body {{ font-family:'Segoe UI',Tahoma,sans-serif; background:#0f172a; color:#f8fafc; margin:30px; }}
 h1 {{ color:#38bdf8; font-size:1.4rem; }} .wrap {{ overflow-x:auto; }}
 table {{ border-collapse:collapse; width:100%; font-size:0.82rem; }}
 th,td {{ border:1px solid #334155; padding:6px 8px; text-align:center; white-space:nowrap; }}
 th {{ background:#1e293b; color:#eab308; text-transform:uppercase; font-size:0.68rem; }} td {{ background:#0b1220; }}
</style></head><body><h1>Resultados HPO (Keras Tuner / Hyperband) — teste 2022-2025</h1>
<div class="wrap"><table><tr>{ths}</tr>{trs}</table></div></body></html>"""
with open("results/resultados_hpo.html", "w", encoding="utf-8") as f:
    f.write(html_cons)
print("[OK] resultados/resultados_hpo.csv + .html atualizados.")

# 7. HTML de uma linha
os.makedirs("results", exist_ok=True)
params_str = (f"lookback={best_lb}, units={best['units']}, dropout={best['dropout']}, "
              f"lr={best['learning_rate']}, batch={best['batch_size']}, thr=0.50 (HPO Hyperband/val_auc)")
features_str = ", ".join(features_sel)
nome_campeao = f"Permutation Top {N_FEATURES_SEL}"
html = f"""<!DOCTYPE html>
<html lang="pt"><head><meta charset="UTF-8"><title>{NOME_MODELO}</title>
<style>
 body {{ font-family:'Segoe UI',Tahoma,sans-serif; background:#0f172a; color:#f8fafc; margin:30px; }}
 h1 {{ color:#38bdf8; font-size:1.4rem; }} .wrap {{ overflow-x:auto; }}
 table {{ border-collapse:collapse; width:100%; font-size:0.85rem; }}
 th,td {{ border:1px solid #334155; padding:8px 10px; text-align:center; white-space:nowrap; }}
 th {{ background:#1e293b; color:#eab308; text-transform:uppercase; font-size:0.72rem; }}
 td {{ background:#0b1220; }} td.txt {{ text-align:left; white-space:normal; max-width:340px; font-size:0.78rem; color:#cbd5e1; }}
</style></head><body>
<h1>{NOME_MODELO} (t+1)</h1><div class="wrap"><table>
<tr><th>Modelo</th><th>Accuracy (%)</th><th>AUC ROC</th><th>Precision (%)</th><th>Recall (%)</th>
<th>F1-Score (%)</th><th>MCC</th><th>Kappa</th><th>Feature Set</th><th>Features por Extenso</th>
<th>Parâmetros Vencedores</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th></tr>
<tr><td>{NOME_MODELO}</td><td>{acc:.2f}</td><td>{auc_roc:.4f}</td><td>{prec:.2f}</td><td>{rec:.2f}</td>
<td>{f1:.2f}</td><td>{mcc:.4f}</td><td>{kappa:.4f}</td><td>{nome_campeao}</td>
<td class="txt">{features_str}</td><td class="txt">{params_str}</td>
<td>{TN}</td><td>{FP}</td><td>{FN}</td><td>{TP}</td></tr>
</table></div></body></html>
"""
with open(CAMINHO_HTML, "w", encoding="utf-8") as file:
    file.write(html)
print(f"[OK] {CAMINHO_HTML} gerado.")
