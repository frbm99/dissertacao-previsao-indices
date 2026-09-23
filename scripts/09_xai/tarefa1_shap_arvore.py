"""XAI Tarefa 1 — SHAP TreeExplainer no ramo Random Forest de cada campeão (teste 2022-2025).

Gera, por índice: summary plot (beeswarm) e gráfico de barras de mean|SHAP|.
Escreve results/xai/importancia_features.csv (importância por feature, ramo=RF).
Não re-treina: usa os modelos guardados via xai_comum.preparar.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xai_comum import preparar, CAMP

os.makedirs('figures/xai', exist_ok=True); os.makedirs('results/xai', exist_ok=True)
linhas = []
for idx in CAMP:
    P = preparar(idx, carregar_rede=False)
    Xte, feats, nome = P['X_test'], P['feats'], P['nome']
    print(f"[{nome}] SHAP TreeExplainer sobre {Xte.shape[0]} amostras de teste, {len(feats)} features...", flush=True)
    expl = shap.TreeExplainer(P['rf'])
    sv = expl.shap_values(Xte)
    # normalizar para a classe positiva (subida)
    if isinstance(sv, list):
        sv1 = sv[1]
    elif getattr(sv, 'ndim', 2) == 3:
        sv1 = sv[:, :, 1]
    else:
        sv1 = sv

    # beeswarm
    plt.figure()
    shap.summary_plot(sv1, Xte, feature_names=feats, show=False, max_display=len(feats))
    plt.title(f"SHAP (RF) — {nome}", fontsize=11); plt.tight_layout()
    plt.savefig(f"figures/xai/shap_beeswarm_{idx}.png", dpi=150, bbox_inches='tight'); plt.close()

    # barras mean|SHAP|
    meanabs = np.abs(sv1).mean(axis=0)
    plt.figure()
    shap.summary_plot(sv1, Xte, feature_names=feats, plot_type='bar', show=False, max_display=len(feats))
    plt.title(f"Importância média |SHAP| (RF) — {nome}", fontsize=11); plt.tight_layout()
    plt.savefig(f"figures/xai/shap_barras_{idx}.png", dpi=150, bbox_inches='tight'); plt.close()

    for f, v in zip(feats, meanabs):
        linhas.append({'Indice': nome, 'Campeao': P['etiq'], 'Ramo': 'RF', 'Feature': f, 'MeanAbsSHAP': round(float(v), 6)})
    top3 = pd.Series(meanabs, index=feats).sort_values(ascending=False).head(3)
    print(f"   top-3: {', '.join(f'{k} ({v:.4f})' for k, v in top3.items())}", flush=True)

CSV = 'results/xai/importancia_features.csv'
tab = pd.DataFrame(linhas).sort_values(['Indice', 'MeanAbsSHAP'], ascending=[True, False])
tab.to_csv(CSV, index=False)
print(f"\n[OK] {CSV} + figuras beeswarm/barras guardadas.")
