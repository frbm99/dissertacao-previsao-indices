"""Etapa 6 — Seleção do modelo campeão por índice pelo MCC de VALIDAÇÃO (2021).

Avalia todos os candidatos (individuais e ensembles, com e sem HPO) num protocolo
uniforme e sem fuga de informação: a escolha nunca observa o conjunto de teste.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef, make_scorer
import xgboost as xgb
import tensorflow as tf

# Seleciona o campeao por índice pelo MCC de VALIDAÇÃO 2021 (sem leakage), protocolo uniforme:
# Top-12, treino<=2020, val=2021, teste=2022+, limiar 0.5. Carrega modelos guardados (sem re-treino,
# exceto RF/XGB individuais que são re-treinados em <=2020). Ensembles: fusão re-afinada na validação.
INDICES = {'sp500': 'data/SP500_Master_Dataset.csv', 'djia': 'data/DJIA_Master_Dataset.csv', 'nasdaq100': 'data/NASDAQ100_Master_Dataset.csv'}
N_FEATURES_SEL = 12
mcc_scorer = make_scorer(matthews_corrcoef)
print(f"Selecao de campeoes por MCC de VALIDACAO 2021 | TF GPUs={tf.config.list_physical_devices('GPU')}\n")

def preprocess(fmaster):
    d = pd.read_csv(fmaster, index_col='Date', parse_dates=True)
    d['Target'] = (d['Close'].shift(-1) > d['Close']).astype(float)
    d.loc[d.index[-1], 'Target'] = np.nan
    d.replace([np.inf, -np.inf], np.nan, inplace=True); d = d.dropna(); d['Target'] = d['Target'].astype(int)
    d['Ouro_Ret'] = d['Ouro'].pct_change(); d['Petroleo_Ret'] = d['Petroleo_WTI'].pct_change()
    d['Dolar_Ret'] = d['Dolar_Index'].pct_change(); d['Volume_Ret'] = d['Volume'].pct_change()
    d['Treasury_10Y_Diff'] = d['Treasury_10Y'].diff(); d['TBill_3M_Diff'] = d['TBill_3M'].diff()
    d['Spread_Diff'] = d['Spread_10Y_3M'].diff()
    cm = [c for c in d.columns if c.startswith('SMA_') or c.startswith('EMA_')]
    for col in cm: d[f"Dist_{col}"] = (d['Close'] / d[col]) - 1
    if 'MACD' in d.columns: d['MACD_Norm'] = d['MACD'] / d['Close']
    if 'BB_High' in d.columns: d['Dist_BB_High'] = (d['Close'] / d['BB_High']) - 1
    if 'BB_Low' in d.columns: d['Dist_BB_Low'] = (d['Close'] / d['BB_Low']) - 1
    if 'OBV' in d.columns:
        d['OBV_Zscore'] = (d['OBV'] - d['OBV'].rolling(20, min_periods=1).mean()) / d['OBV'].rolling(20, min_periods=1).std()
    d.replace([np.inf, -np.inf], np.nan, inplace=True); d = d.dropna()
    excl = ['Target','Open','High','Low','Close','Volume','Ouro','Petroleo_WTI','Dolar_Index','Treasury_10Y',
            'TBill_3M','Spread_10Y_3M','SP500','NASDAQ','NASDAQ100','DJIA','Dow_Jones','Sentiment_Score',
            'Sentiment_Score_MA3','MACD','OBV','BB_High','BB_Low','ATR','MOM_10'] + cm
    banco = [c for c in d.columns if c not in excl]; y = d['Target']
    Xf = d.loc[:'2019-12-31', banco]; yf = y.loc[:'2019-12-31']
    Xi = d.loc['2020-01-01':'2020-12-31', banco]; yi = y.loc['2020-01-01':'2020-12-31']
    s = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1, class_weight='balanced')
    s.fit(Xf, yf)
    pm = permutation_importance(s, Xi, yi, scoring=mcc_scorer, n_repeats=10, random_state=42, n_jobs=-1)
    top12 = pd.Series(pm.importances_mean, index=banco).sort_values(ascending=False).nlargest(N_FEATURES_SEL).index.tolist()
    X = d[top12]
    Xtr, ytr = X.loc[:'2020-12-31'], y.loc[:'2020-12-31']
    Xv, yv = X.loc['2021-01-01':'2021-12-31'], y.loc['2021-01-01':'2021-12-31']
    Xte, yte = X.loc['2022-01-01':'2025-12-31'], y.loc['2022-01-01':'2025-12-31']
    sc = Pipeline([('r', RobustScaler()), ('m', MinMaxScaler((-1, 1)))])
    Xtr_s = sc.fit_transform(Xtr); Xv_s = sc.transform(Xv); Xte_s = sc.transform(Xte)
    return dict(top12=top12, Xtr=Xtr, ytr=ytr, Xv=Xv, yv=yv.values, Xte=Xte, yte=yte.values,
                Xtr_s=Xtr_s, Xv_s=Xv_s, Xte_s=Xte_s)

def seq(cur, lb, ctx):
    cur = np.vstack([ctx[-(lb-1):], cur])
    return np.array([cur[i:i+lb] for i in range(len(cur)-lb+1)])

def prob_keras(path, P):
    m = tf.keras.models.load_model(path)
    lb = m.input_shape[1]
    pv = m.predict(seq(P['Xv_s'], lb, P['Xtr_s']), verbose=0).flatten()
    pt = m.predict(seq(P['Xte_s'], lb, P['Xv_s']), verbose=0).flatten()
    tf.keras.backend.clear_session()
    return pv, pt

def prob_tree(model, P):
    return model.predict_proba(P['Xv'])[:, 1], model.predict_proba(P['Xte'])[:, 1]

def mcc(y, p): return matthews_corrcoef(y, (p > 0.5).astype(int))

def fundir(pa_v, pa_t, pb_v, pb_t, yv):
    bw, bm = 0.5, -2
    for w in np.round(np.arange(0, 1.01, 0.1), 2):
        m = mcc(yv, w*pa_v + (1-w)*pb_v)
        if m > bm: bm, bw = m, w
    return bm, bw*pa_t + (1-bw)*pb_t

resultados = []
for idx, fmaster in INDICES.items():
    P = preprocess(fmaster)
    yv, yte = P['yv'], P['yte']

    def add(nome, cenario, pv, pt):
        resultados.append({'Indice': idx, 'Modelo': nome, 'Cenario': cenario,
                           'MCC_val': round(mcc(yv, pv), 4), 'MCC_test': round(mcc(yte, pt), 4)})

    # RF / XGBoost individuais (re-treino <=2020 / Top-12)
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, max_features='sqrt',
                                class_weight='balanced', random_state=42, n_jobs=-1).fit(P['Xtr'], P['ytr'])
    add('Random Forest', 'sem-HPO', *prob_tree(rf, P))
    spw = (P['ytr'] == 0).sum() / max((P['ytr'] == 1).sum(), 1)
    xg = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                           gamma=1, reg_alpha=0.1, reg_lambda=10, scale_pos_weight=spw, objective='binary:logistic',
                           eval_metric='logloss', tree_method='hist', device='cpu', random_state=42, n_jobs=-1).fit(P['Xtr'], P['ytr'])
    add('XGBoost', 'sem-HPO', *prob_tree(xg, P))

    # Redes individuais (sem-HPO e HPO) - carregar guardados
    redes = {'LSTM': 'models/campeao_lstm_{}.keras', 'GRU': 'models/campeao_gru_{}.keras',
             'Transformer-1bl': 'models/campeao_transformer_{}.keras', 'DeepTransformer-2bl': 'models/campeao_deep_transformer_{}.keras'}
    for nome, pat in redes.items():
        try:
            add(nome, 'sem-HPO', *prob_keras(pat.format(idx), P))
        except Exception as e:
            print(f"  [skip] {nome} sem-HPO {idx}: {e}")
    hpo = {'LSTM': 'models/campeao_hpo_lstm_{}.keras', 'GRU': 'models/campeao_hpo_gru_{}.keras',
           'Transformer-1bl': 'models/campeao_hpo_transformer_{}.keras', 'DeepTransformer-2bl': 'models/campeao_hpo_deeptransf_{}.keras'}
    for nome, pat in hpo.items():
        try:
            add(nome, 'HPO', *prob_keras(pat.format(idx), P))
        except Exception as e:
            print(f"  [skip] {nome} HPO {idx}: {e}")

    # Ensembles sem-HPO (carregar 2 ramos; fusão re-afinada na validação)
    ens = [
        ('LSTM+XGB', 'models/campeao_ensLX_lstm_{}.keras', 'models/campeao_ensLX_xgb_{}.pkl', 'keras'),
        ('LSTM+RF', 'models/campeao_ensLR_lstm_{}.keras', 'models/campeao_ensLR_rf_{}.pkl', 'keras'),
        ('GRU+XGB', 'models/campeao_ensGX_gru_{}.keras', 'models/campeao_ensGX_xgb_{}.pkl', 'keras'),
        ('GRU+RF', 'models/campeao_ensGR_gru_{}.keras', 'models/campeao_ensGR_rf_{}.pkl', 'keras'),
        ('DeepTransf+LSTM', 'models/campeao_ensDTL_deeptransf_{}.keras', 'models/campeao_ensDTL_lstm_{}.keras', 'keras2'),
        ('DeepTransf+RF', 'models/campeao_ensDTR_deeptransf_{}.keras', 'models/campeao_ensDTR_rf_{}.pkl', 'keras'),
        ('DeepTransf+XGB', 'models/campeao_ensDTX_deeptransf_{}.keras', 'models/campeao_ensDTX_xgb_{}.pkl', 'keras'),
    ]
    for nome, pa, pb, kind in ens:
        try:
            pav, pat = prob_keras(pa.format(idx), P)
            if kind == 'keras2':
                pbv, pbt = prob_keras(pb.format(idx), P)
            else:
                pbv, pbt = prob_tree(joblib.load(pb.format(idx)), P)
            mv, pt = fundir(pav, pat, pbv, pbt, yv)
            resultados.append({'Indice': idx, 'Modelo': nome, 'Cenario': 'sem-HPO',
                               'MCC_val': round(mv, 4), 'MCC_test': round(mcc(yte, pt), 4)})
        except Exception as e:
            print(f"  [skip] {nome} sem-HPO {idx}: {e}")

    # Ensembles HPO (ramo-rede = modelo HPO; ramo-árvore = hpoens .pkl)
    ensh = [
        ('LSTM+XGB', 'models/campeao_hpo_lstm_{}.keras', 'models/campeao_hpoens_lstm_xgb_{}_xgb.pkl', 'tree'),
        ('LSTM+RF', 'models/campeao_hpo_lstm_{}.keras', 'models/campeao_hpoens_lstm_rf_{}_rf.pkl', 'tree'),
        ('GRU+XGB', 'models/campeao_hpo_gru_{}.keras', 'models/campeao_hpoens_gru_xgb_{}_xgb.pkl', 'tree'),
        ('GRU+RF', 'models/campeao_hpo_gru_{}.keras', 'models/campeao_hpoens_gru_rf_{}_rf.pkl', 'tree'),
        ('DeepTransf+RF', 'models/campeao_hpo_deeptransf_{}.keras', 'models/campeao_hpoens_deeptransf_rf_{}_rf.pkl', 'tree'),
        ('DeepTransf+XGB', 'models/campeao_hpo_deeptransf_{}.keras', 'models/campeao_hpoens_deeptransf_xgb_{}_xgb.pkl', 'tree'),
        ('DeepTransf+LSTM', 'models/campeao_hpo_deeptransf_{}.keras', 'models/campeao_hpo_lstm_{}.keras', 'keras2'),
    ]
    for nome, pa, pb, kind in ensh:
        try:
            pav, pat = prob_keras(pa.format(idx), P)
            if kind == 'keras2':
                pbv, pbt = prob_keras(pb.format(idx), P)
            else:
                pbv, pbt = prob_tree(joblib.load(pb.format(idx)), P)
            mv, pt = fundir(pav, pat, pbv, pbt, yv)
            resultados.append({'Indice': idx, 'Modelo': nome, 'Cenario': 'HPO',
                               'MCC_val': round(mv, 4), 'MCC_test': round(mcc(yte, pt), 4)})
        except Exception as e:
            print(f"  [skip] {nome} HPO {idx}: {e}")

    print(f"[{idx}] {len([r for r in resultados if r['Indice']==idx])} candidatos avaliados")

df = pd.DataFrame(resultados)
df.to_csv("results/campeoes_validacao_todos.csv", index=False)
print("\n=========== CAMPEAO POR INDICE (max MCC de VALIDACAO) ===========")
camp = []
for idx in INDICES:
    sub = df[df['Indice'] == idx].sort_values('MCC_val', ascending=False)
    top = sub.iloc[0]
    print(f"\n### {idx.upper()} ###")
    print(sub.head(5).to_string(index=False))
    print(f"  -> CAMPEAO (val): {top['Modelo']} [{top['Cenario']}]  MCC_val={top['MCC_val']:+.4f}  MCC_test={top['MCC_test']:+.4f}")
    camp.append({'Indice': idx, 'Campeao': top['Modelo'], 'Cenario': top['Cenario'],
                 'MCC_val': top['MCC_val'], 'MCC_test': top['MCC_test']})
pd.DataFrame(camp).to_csv("results/campeoes_validacao.csv", index=False)
print("\n[OK] resultados/campeoes_validacao.csv (+ _todos.csv) guardados.")
