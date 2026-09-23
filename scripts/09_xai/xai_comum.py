"""XAI — módulo comum: reconstrói o pipeline dos campeões e CARREGA os modelos guardados.

Não re-treina nada: reproduz o pré-processamento determinístico (features, split, scaler,
sequências) idêntico aos scripts de ensemble, carrega o ramo RF (.pkl) e o ramo rede (.keras)
de models/, e devolve probabilidades de validação/teste e a fusão (peso re-derivado na validação,
tal como na seleção). As features vêm de rf.feature_names_in_ (fonte definitiva do Top-12).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef
import tensorflow as tf

DATASETS = {'sp500': 'data/SP500_Master_Dataset.csv', 'djia': 'data/DJIA_Master_Dataset.csv',
            'nasdaq100': 'data/NASDAQ100_Master_Dataset.csv'}
CAMP = {
    'sp500': dict(nome='SP500', etiq='DeepTransformer+RF (HPO)', nn='deeptransf',
                  rf='models/campeao_hpoens_deeptransf_rf_sp500_rf.pkl', net='models/campeao_hpo_deeptransf_sp500.keras'),
    'djia': dict(nome='DJIA', etiq='LSTM+RF (sem-HPO)', nn='lstm',
                 rf='models/campeao_ensLR_rf_djia.pkl', net='models/campeao_ensLR_lstm_djia.keras'),
    'nasdaq100': dict(nome='NASDAQ-100', etiq='GRU+RF (sem-HPO)', nn='gru',
                      rf='models/campeao_ensGR_rf_nasdaq100.pkl', net='models/campeao_ensGR_gru_nasdaq100.keras'),
}

def _engenharia(fmaster):
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
    return d

def _seq(cur, lb, ctx):
    cur = np.vstack([ctx[-(lb - 1):], cur])
    return np.array([cur[i:i + lb] for i in range(len(cur) - lb + 1)])

def preparar(indice, carregar_rede=True):
    cfg = CAMP[indice]
    d = _engenharia(DATASETS[indice]); y = d['Target']
    rf = joblib.load(cfg['rf'])
    feats = list(rf.feature_names_in_)                 # Top-12 definitivo (o que a RF viu)
    assert all(f in d.columns for f in feats), "features do modelo ausentes na engenharia"
    X = d[feats]
    Xtr = X.loc[:'2020-12-31']                          # ramo treinado <=2020 (ensemble)
    Xv, yv = X.loc['2021-01-01':'2021-12-31'], y.loc['2021-01-01':'2021-12-31']
    Xte, yte = X.loc['2022-01-01':'2025-12-31'], y.loc['2022-01-01':'2025-12-31']

    prf_v = rf.predict_proba(Xv)[:, 1]; prf_t = rf.predict_proba(Xte)[:, 1]
    out = dict(nome=cfg['nome'], etiq=cfg['etiq'], nn=cfg['nn'], feats=feats,
               rf=rf, X_test=Xte, y_test=yte.values, X_val=Xv, y_val=yv.values,
               prob_rf_val=prf_v, prob_rf_test=prf_t)

    if carregar_rede:
        net = tf.keras.models.load_model(cfg['net'])
        lb = net.input_shape[1]
        sc = Pipeline([('r', RobustScaler()), ('m', MinMaxScaler((-1, 1)))])
        Xtr_s = sc.fit_transform(Xtr); Xv_s = sc.transform(Xv); Xte_s = sc.transform(Xte)
        Xv3 = _seq(Xv_s, lb, Xtr_s); Xte3 = _seq(Xte_s, lb, Xv_s)
        pnv = net.predict(Xv3, verbose=0).flatten(); pnt = net.predict(Xte3, verbose=0).flatten()
        # peso de fusão re-derivado na validação (igual à seleção): max MCC val @0.5
        bw, bm = 0.5, -2
        for w in np.round(np.arange(0, 1.01, 0.1), 2):
            m = matthews_corrcoef(yv.values, ((w * prf_v + (1 - w) * pnv) > 0.5).astype(int))
            if m > bm: bm, bw = m, w
        out.update(net=net, lookback=lb, Xte3=Xte3, Xv3=Xv3, Xtr3=_seq(Xtr_s, lb, Xtr_s),
                   prob_net_val=pnv, prob_net_test=pnt, peso_rf=bw,
                   prob_fused_test=bw * prf_t + (1 - bw) * pnt)
    return out
