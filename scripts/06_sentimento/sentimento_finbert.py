"""Etapa 7 — Análise de sentimento com FinBERT (ProsusAI/finbert).

Pontua todas as manchetes (2008-2024) com um score contínuo (prob. positiva menos
prob. negativa), alinha ao calendário de mercado sem lookahead e funde a mesma série
de sentimento nos datasets dos três índices (variáveis Sentiment_Score e MA3).
"""
import os
import pandas as pd
from transformers import pipeline
import torch


LIMIT = int(os.environ.get('FINBERT_LIMIT', 0))  # >0 = so as primeiras N manchetes (smoke-test)
FICHEIRO_NOTICIAS = "data/sp500_headlines_2008_2024.csv"
INTERMEDIO = "data/headlines_sentiment_finbert.csv"   # score por manchete (parte lenta, feita 1 vez)
INDICES = {
    'SP500': 'data/SP500_Master_Dataset.csv',
    'DJIA': 'data/DJIA_Master_Dataset.csv',
    'NASDAQ100': 'data/NASDAQ100_Master_Dataset.csv',
}

# 1. Carregar TODAS as notícias (sem filtro >=2021 -> o treino passa a ter sentimento real)
df_news = pd.read_csv(FICHEIRO_NOTICIAS)
df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce')
df_news = df_news.dropna(subset=['Date', 'Title']).sort_values('Date').reset_index(drop=True)
if LIMIT > 0:
    df_news = df_news.head(LIMIT)
print(f"Manchetes: {len(df_news)}  ({df_news['Date'].min().date()} -> {df_news['Date'].max().date()})")

# 2. FinBERT por manchete: score contínuo = prob_positivo - prob_negativo  (em [-1, 1])
device = 0 if torch.cuda.is_available() else -1
print(f"\nA carregar FinBERT (ProsusAI/finbert) | device={'GPU' if device == 0 else 'CPU'}...")
clf = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device, batch_size=16)

titles = df_news['Title'].astype(str).tolist()
scores = []
CHUNK = 500
print("A pontuar manchetes...")
for i in range(0, len(titles), CHUNK):
    res = clf(titles[i:i + CHUNK], top_k=None, truncation=True, max_length=512)
    for r in res:
        if isinstance(r, dict):
            r = [r]
        pp = next((d['score'] for d in r if d['label'] == 'positive'), 0.0)
        pn = next((d['score'] for d in r if d['label'] == 'negative'), 0.0)
        scores.append(pp - pn)
    print(f"  {min(i + CHUNK, len(titles))}/{len(titles)}", flush=True)

df_news['score'] = scores
df_news[['Date', 'score']].to_csv(INTERMEDIO, index=False)
print(f"[OK] Scores por manchete guardados em {INTERMEDIO}")

# 3. Para cada índice: alinhar ao calendário de mercado (fim-de-semana -> próximo dia aberto),
#    agregar por dia (média), fundir no master, preencher dias sem notícias com 0 (neutro), MA3.
news_sorted = df_news[['Date', 'score']].sort_values('Date')
for nome, fmaster in INDICES.items():
    m = pd.read_csv(fmaster)
    m['Date'] = pd.to_datetime(m['Date'])
    cal = m[['Date']].sort_values('Date').rename(columns={'Date': 'Market_Date'})
    # merge_asof forward: cada notícia vai para o PRÓXIMO dia de mercado (sem lookahead)
    nn = pd.merge_asof(news_sorted, cal, left_on='Date', right_on='Market_Date', direction='forward')
    nn = nn.dropna(subset=['Market_Date'])
    diario = (nn.groupby('Market_Date')['score'].mean().reset_index()
                .rename(columns={'Market_Date': 'Date', 'score': 'Sentiment_Score'}))
    out = pd.merge(m, diario, on='Date', how='left')
    out['Sentiment_Score'] = out['Sentiment_Score'].fillna(0.0)
    out['Sentiment_Score_MA3'] = out['Sentiment_Score'].rolling(window=3, min_periods=1).mean()
    saida = f"data/{nome}_Multimodal_Dataset.csv"
    out.to_csv(saida, index=False)

    o = out.copy(); o['Date'] = pd.to_datetime(o['Date'])
    tr = o[o['Date'] < '2021-01-01']; te = o[o['Date'] >= '2021-01-01']
    print(f"{saida}: shape {out.shape} | sentimento!=0 -> treino {100*(tr['Sentiment_Score']!=0).mean():.1f}% | val+teste {100*(te['Sentiment_Score']!=0).mean():.1f}%")

print("\nSUCESSO! Multimodais criados/atualizados para SP500, DJIA e NASDAQ100 (sentimento partilhado).")
