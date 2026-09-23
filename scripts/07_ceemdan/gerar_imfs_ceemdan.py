"""Fase 3 (Etapa 8) — Geração das features CEEMDAN causais (=<t) para os três índices.

Decompõe a série do fecho (Close) em IMFs de forma ESTRITAMENTE CAUSAL: para cada dia t,
a CEEMDAN é ajustada apenas à série até esse dia (Close[:t+1]) e retém-se o último valor de
cada IMF como feature desse dia. Assim, a feature do dia t nunca depende de dados posteriores
a t — este é o mecanismo anti-leakage (mais rigoroso do que a janela por ano: zero lookahead,
nem sequer intra-ano). O número de IMFs é fixado em 10 (preenchido com zeros quando a
decomposição produz menos; truncadas as de menor energia quando produz mais), para que todos
os dias e índices tenham exatamente as mesmas colunas.

As decomposições dos vários dias são independentes entre si, pelo que são distribuídas por
vários processos (cada um com CEEMDAN em modo série para evitar multiprocessing aninhado).
Escreve data/ceemdan_imfs_{indice}.csv com checkpoints periódicos (retomável).
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

INDICES = {'sp500': 'data/SP500_Master_Dataset.csv',
           'djia': 'data/DJIA_Master_Dataset.csv',
           'nasdaq100': 'data/NASDAQ100_Master_Dataset.csv'}
K_IMF = 10                                   # nº fixo de IMFs (features) por dia
MIN_LEN = 30                                 # série mínima para decompor (dias iniciais -> zeros)
TRIALS = 100                                 # tamanho do ensemble CEEMDAN (padrão da literatura)
NW = int(os.environ.get('CEEMDAN_WORKERS', 12))

CLOSE = None
def _init(arr):
    global CLOSE
    CLOSE = arr

def _worker(i):
    if i + 1 < MIN_LEN:
        return i, np.zeros(K_IMF)
    try:
        from PyEMD import CEEMDAN
        c = CEEMDAN(trials=TRIALS, parallel=False)
        c.noise_seed(42)                     # realização de ruído fixa -> reprodutível
        imfs = c.ceemdan(CLOSE[:i + 1])      # (n_imfs, i+1)
        last = imfs[:, -1]                   # valor causal de cada IMF no dia i
        n = last.shape[0]
        if n > K_IMF:                        # truncar as de MENOR energia, manter ordem de frequência
            energia = (imfs ** 2).sum(axis=1)
            manter = np.sort(np.argsort(energia)[::-1][:K_IMF])
            last = last[manter]
            n = K_IMF
        vec = np.zeros(K_IMF)
        vec[:n] = last[:K_IMF]
        return i, vec
    except Exception:
        return i, None                       # falha -> fica NaN e é recalculada na retoma

def processar_indice(idx, path):
    d = pd.read_csv(path, index_col='Date', parse_dates=True)
    close = d['Close'].values.astype(float)
    dates = d.index
    N = len(close)
    out = f"data/ceemdan_imfs_{idx}.csv"
    cols = [f"IMF_{k + 1}" for k in range(K_IMF)]

    resultados = np.full((N, K_IMF), np.nan)
    if os.path.exists(out):                  # retoma: reaproveita linhas já calculadas
        prev = pd.read_csv(out, index_col=0, parse_dates=True)
        prev = prev.reindex(dates)
        mask = ~prev.isna().any(axis=1).values
        resultados[mask] = prev.values[mask]
    todo = [i for i in range(N) if np.isnan(resultados[i]).any()]
    todo.sort(reverse=True)                  # tarefas maiores primeiro -> melhor balanceamento
    print(f"[{idx}] {N} dias | por calcular: {len(todo)} | workers={NW}", flush=True)
    if not todo:
        return

    t0 = time.perf_counter()
    with mp.Pool(NW, initializer=_init, initargs=(close,)) as pool:
        for cnt, (i, vec) in enumerate(pool.imap_unordered(_worker, todo, chunksize=1), 1):
            if vec is not None:
                resultados[i] = vec
            if cnt % 200 == 0:
                el = time.perf_counter() - t0
                eta = (len(todo) - cnt) / (cnt / el)
                print(f"[{idx}] {cnt}/{len(todo)} | {el / 60:.1f} min | ETA {eta / 60:.1f} min", flush=True)
                pd.DataFrame(resultados, index=dates, columns=cols).to_csv(out)
    pd.DataFrame(resultados, index=dates, columns=cols).to_csv(out)
    print(f"[{idx}] CONCLUIDO em {(time.perf_counter() - t0) / 60:.1f} min -> {out}", flush=True)

if __name__ == '__main__':
    for idx, path in INDICES.items():
        processar_indice(idx, path)
    print("CEEMDAN FEATURES CONCLUIDO", flush=True)
