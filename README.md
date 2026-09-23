# Previsão Direcional de Índices Bolsistas com Machine Learning, Deep Learning e Análise de Sentimento

Repositório de código da dissertação de mestrado sobre a previsão da **direção diária (t+1)** de três índices bolsistas norte-americanos — **S&P 500**, **DJIA** e **NASDAQ-100** — combinando modelos de machine learning clássico (Random Forest, XGBoost), redes neuronais profundas (LSTM, GRU, Transformer, Deep Transformer), *ensembles* híbridos por *soft-voting* e análise de sentimento de notícias financeiras com FinBERT.

## Pipeline experimental

1. **Aquisição de dados** — histórico OHLCV dos três índices (2005-2025) e variáveis macroeconómicas (VIX, petróleo WTI, ouro, índice do dólar, taxas de juro) via `yfinance`.
2. **Engenharia de variáveis** — indicadores técnicos (`ta`), transformações de estacionaridade (retornos, distâncias a médias móveis, *z-scores*) e alvo binário t+1; produz os *Master Datasets*.
3. **Modelos individuais** — Random Forest, XGBoost, LSTM, GRU, Transformer (1 bloco) e Deep Transformer (2 blocos), treinados de forma idêntica e independente para cada índice.
4. **Ensembles híbridos** — seis combinações por média ponderada de probabilidades (*soft-voting*): LSTM+RF, GRU+RF, LSTM+XGB, GRU+XGB, DeepTransformer+RF e DeepTransformer+XGB; o peso de fusão é ajustado na validação.
5. **Otimização de hiperparâmetros (HPO)** — Keras Tuner / Hyperband sobre as redes neuronais (lookback, unidades, dropout, learning rate, batch size), com objetivo na métrica de validação.
6. **Seleção dos campeões** — o melhor modelo de cada índice é escolhido pelo MCC no conjunto de **validação** (2021), sem qualquer observação do conjunto de teste.
7. **Análise de sentimento** — pontuação de manchetes financeiras (2008-2024) com FinBERT e ablação controlada do sentimento nos modelos campeões (comparação unimodal vs. multimodal).

### Protocolo de validação

- Divisão **estritamente cronológica**, sem embaralhamento: treino até 2020 (redes) ou 2021 com *walk-forward* interno (árvores), validação em 2021, **teste cego de 2022 a 2025**.
- Normalização (RobustScaler → MinMaxScaler) ajustada **exclusivamente no treino**.
- Seleção de variáveis por *permutation importance* fora da amostra (seletor treinado até 2019, avaliado em 2020).
- Toda a otimização (hiperparâmetros, sementes, pesos de fusão) usa apenas treino e validação; o teste é tocado uma única vez, na avaliação final.

## Extensões experimentais

Sobre a base direcional a t+1, o trabalho inclui três extensões, aplicadas aos modelos campeões:

8. **Decomposição de sinal (CEEMDAN)** — decomposição estritamente causal do preço de fecho em IMFs (janela expansível dia-a-dia, sem *lookahead*), adicionadas como variáveis aos campeões. Resultados em `results/resultados_ceemdan.csv`.
9. **Horizontes de previsão** — reavaliação dos campeões para os horizontes t+3, t+5 e t+7 (só se altera a variável-alvo). Resultados em `results/multihorizonte/`.
10. **Explicabilidade e calibração** — SHAP (TreeExplainer no ramo Random Forest, GradientExplainer nas redes) e diagramas de fiabilidade / Brier score. Resultados em `results/xai/` e figuras em `figures/xai/`.

## Estrutura do repositório

```
scripts/
  00_exploratorio/   Análise exploratória (auditoria do dataset, tabelas e correlações)
  01_dados/          Aquisição e construção dos Master Datasets + figuras dos preços
  02_modelos/        Modelos individuais (6 algoritmos × 3 índices)
  03_ensembles/      Ensembles híbridos soft-voting (6 combinações × 3 índices)
  04_hpo/            Otimização de hiperparâmetros (Hyperband) e ensembles com redes otimizadas
  05_selecao/        Seleção dos campeões por validação e benchmark de custo computacional
  06_sentimento/     Sentimento FinBERT e ablação multimodal nos campeões
  07_ceemdan/        Decomposição de sinal CEEMDAN (causal) e ablação
  08_multihorizonte/ Previsão direcional a t+3, t+5, t+7
  09_xai/            Explicabilidade (SHAP) e calibração (Brier)
data/                Datasets (brutos, mestres, multimodais, manchetes e scores FinBERT)
models/              Modelos treinados (.keras / .pkl)
results/             Resultados: tabelas HTML por modelo e consolidados CSV
figures/             Figuras da dissertação (gráficos de preços e diagramas de arquitetura)
_archive/            Material de desenvolvimento não incluído na versão final da tese
```

## Instalação

```bash
python -m venv tese_env
tese_env\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Execução

Todos os scripts se executam **a partir da raiz do repositório** (os caminhos `data/`, `models/`, `results/` e `figures/` são relativos à raiz):

```bash
# Etapas 1-2: construção dos datasets
python scripts/01_dados/download_sp500.py
python scripts/01_dados/indicadores_sp500.py
python scripts/01_dados/macro_sp500.py
python scripts/01_dados/construir_dataset_djia.py
python scripts/01_dados/construir_dataset_nasdaq100.py

# Etapa 3: um modelo individual (exemplo)
python scripts/02_modelos/lstm_sp500.py

# Etapa 4: um ensemble (exemplo)
python scripts/03_ensembles/ensemble_deep_transformer_rf_nasdaq100.py

# Etapa 5: HPO (parametrizado por variáveis de ambiente)
HPO_FAMILY=lstm HPO_INDICE=sp500 python scripts/04_hpo/hpo_redes.py

# Etapa 6: seleção dos campeões por validação
python scripts/05_selecao/selecionar_campeoes.py

# Etapa 7: sentimento e ablação multimodal
python scripts/06_sentimento/sentimento_finbert.py
MM_INDICE=nasdaq100 MM_NN=gru MM_HPO=0 python scripts/06_sentimento/ablacao_sentimento.py
```

Notas:
- As redes neuronais aceitam variáveis de ambiente para controlar o orçamento de treino (ex.: `LSTM_SEEDS`, `GRU_EPOCHS`, `HPO_MAX_EPOCHS`); os valores por omissão reproduzem a dissertação.
- Os resultados de cada execução são gravados em `results/` (uma tabela HTML por modelo e consolidados CSV: `resultados_hpo.csv`, `campeoes_validacao.csv`, `resultados_multimodal_campeoes.csv`, `resultados_tempos.csv`).

## Stack técnica

| Componente | Versão |
|---|---|
| Python | 3.12.7 |
| TensorFlow / Keras | 2.21.0 |
| Keras Tuner | 1.4.8 |
| scikit-learn | 1.8.0 |
| XGBoost | 3.2.0 |
| transformers (FinBERT) | 5.12.0 |
| PyTorch | 2.12.0 (CPU) |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| ta | 0.11.0 |
| yfinance | 1.3.0 |
| matplotlib | 3.10.9 |
| EMD-signal (CEEMDAN) | 1.10.0 |
| shap | 0.52.0 |

Todas as experiências foram executadas em **CPU** (o TensorFlow ≥ 2.11 não utiliza GPU em Windows nativo e a instalação de PyTorch é a variante CPU); os tempos reportados em `results/resultados_tempos.csv` refletem essa configuração.
