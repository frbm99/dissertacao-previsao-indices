"""XAI Tarefa 2 — SHAP no ramo REDE (janelas 3D). Só corre se for viável em tempo razoável.

Usa GradientExplainer (compatível com TF2/Keras 3). A importância SHAP é agregada por feature
fazendo média de |SHAP| sobre amostras e sobre a dimensão temporal do lookback -> importância
interpretável por feature. Não re-treina: usa os modelos guardados via xai_comum.

Env XAI_MODE: 'test' (1 índice, N amostras, para cronometrar) ou 'full' (3 índices, teste completo).
Env XAI_N: nº de amostras no modo test (default 100).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from xai_comum import preparar, CAMP

MODE = os.environ.get('XAI_MODE', 'test')
N = int(os.environ.get('XAI_N', 100))
BG = int(os.environ.get('XAI_BG', 100))

def shap_rede(P, n_amostras=None):
    net, Xtr3, Xte3, feats = P['net'], P['Xtr3'], P['Xte3'], P['feats']
    rng = np.random.RandomState(42)
    bg = Xtr3[rng.choice(len(Xtr3), size=min(BG, len(Xtr3)), replace=False)]
    alvo = Xte3 if n_amostras is None else Xte3[:n_amostras]
    expl = shap.GradientExplainer(net, bg)
    sv = expl.shap_values(alvo)
    if isinstance(sv, list): sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 4: sv = sv[..., 0]              # (n, lb, feat, 1) -> (n, lb, feat)
    meanabs = np.abs(sv).mean(axis=(0, 1))         # média sobre amostras e tempo -> por feature
    return meanabs, alvo.shape[0]

if MODE == 'test':
    idx = os.environ.get('XAI_INDICE', 'nasdaq100')
    P = preparar(idx, carregar_rede=True)
    print(f"[teste] {P['nome']} rede={P['nn']} lookback={P['lookback']} | {N} amostras, background {BG}...", flush=True)
    t0 = time.perf_counter(); meanabs, n = shap_rede(P, N); dt = time.perf_counter() - t0
    por_amostra = dt / n
    total_amostras = sum(preparar(i, carregar_rede=False)['X_test'].shape[0] for i in CAMP)
    est = por_amostra * total_amostras + 3 * 60  # + overhead de carregamento/explainer por índice
    print(f"[teste] {n} amostras em {dt:.1f}s ({por_amostra:.2f}s/amostra)", flush=True)
    print(f"[extrapolacao] teste completo 3 indices (~{total_amostras} amostras) ~= {est/60:.1f} min", flush=True)
    top3 = pd.Series(meanabs, index=P['feats']).sort_values(ascending=False).head(3)
    print(f"[teste] top-3 features (rede): {', '.join(f'{k} ({v:.4f})' for k,v in top3.items())}", flush=True)
else:
    os.makedirs('figures/xai', exist_ok=True); os.makedirs('results/xai', exist_ok=True)
    linhas = []
    for idx in CAMP:
        P = preparar(idx, carregar_rede=True)
        print(f"[{P['nome']}] SHAP rede sobre teste completo ({P['Xte3'].shape[0]} amostras)...", flush=True)
        t0 = time.perf_counter(); meanabs, n = shap_rede(P, None); dt = time.perf_counter() - t0
        s = pd.Series(meanabs, index=P['feats']).sort_values(ascending=False)
        plt.figure(figsize=(7, 4)); s[::-1].plot.barh()
        plt.title(f"Importância |SHAP| média (rede {P['nn'].upper()}) — {P['nome']}", fontsize=10)
        plt.xlabel('mean|SHAP| (média sobre tempo)'); plt.tight_layout()
        plt.savefig(f"figures/xai/shap_rede_barras_{idx}.png", dpi=150, bbox_inches='tight'); plt.close()
        for f, v in zip(P['feats'], meanabs):
            linhas.append({'Indice': P['nome'], 'Campeao': P['etiq'], 'Ramo': 'rede', 'Feature': f, 'MeanAbsSHAP': round(float(v), 6)})
        print(f"   {dt:.1f}s | top-3: {', '.join(f'{k} ({v:.4f})' for k,v in s.head(3).items())}", flush=True)
    CSV = 'results/xai/importancia_features.csv'
    novo = pd.DataFrame(linhas)
    if os.path.exists(CSV):
        antigo = pd.read_csv(CSV); antigo = antigo[antigo['Ramo'] != 'rede']
        novo = pd.concat([antigo, novo], ignore_index=True)
    novo.sort_values(['Indice', 'Ramo', 'MeanAbsSHAP'], ascending=[True, True, False]).to_csv(CSV, index=False)
    print(f"\n[OK] importância da rede acrescentada a {CSV} + figuras.")
