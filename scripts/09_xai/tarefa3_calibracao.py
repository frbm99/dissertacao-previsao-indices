"""XAI Tarefa 3 — Calibração dos campeões (teste 2022-2025).

Para cada campeão calcula o Brier score (probabilidade fundida do ensemble, e por ramo) e gera
um reliability diagram (probabilidade prevista vs frequência observada). Não re-treina.
Escreve results/xai/calibracao.csv e figures/xai/calibracao_{indice}.png.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from xai_comum import preparar, CAMP

os.makedirs('figures/xai', exist_ok=True); os.makedirs('results/xai', exist_ok=True)
linhas = []
for idx in CAMP:
    P = preparar(idx, carregar_rede=True)
    y = P['y_test']; pf = P['prob_fused_test']; prf = P['prob_rf_test']; pnn = P['prob_net_test']
    b_f = brier_score_loss(y, pf); b_rf = brier_score_loss(y, prf); b_nn = brier_score_loss(y, pnn)
    # referência: Brier de um previsor constante = taxa-base (p = média de y)
    base = y.mean(); b_base = brier_score_loss(y, np.full_like(pf, base))
    print(f"[{P['nome']}] Brier fundido={b_f:.4f} | RF={b_rf:.4f} | rede={b_nn:.4f} | base(p={base:.3f})={b_base:.4f} | peso_RF={P['peso_rf']*100:.0f}%", flush=True)

    frac_pos, mean_pred = calibration_curve(y, pf, n_bins=10, strategy='uniform')
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Calibração perfeita')
    plt.plot(mean_pred, frac_pos, 'o-', label=f"{P['etiq']}")
    plt.xlabel('Probabilidade prevista'); plt.ylabel('Frequência observada de subida')
    plt.title(f"Reliability diagram — {P['nome']}\nBrier={b_f:.4f} (base={b_base:.4f})", fontsize=10)
    plt.legend(loc='upper left', fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"figures/xai/calibracao_{idx}.png", dpi=150, bbox_inches='tight'); plt.close()

    linhas.append({'Indice': P['nome'], 'Campeao': P['etiq'], 'N_teste': len(y),
                   'Brier_fundido': round(b_f, 4), 'Brier_RF': round(b_rf, 4), 'Brier_rede': round(b_nn, 4),
                   'Brier_base': round(b_base, 4), 'Taxa_base': round(float(base), 4), 'Peso_RF': P['peso_rf']})

CSV = 'results/xai/calibracao.csv'
pd.DataFrame(linhas).to_csv(CSV, index=False)
print(f"\n[OK] {CSV} + reliability diagrams guardados.")
