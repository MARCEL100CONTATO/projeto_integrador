# Projeto Integrador (P15) — Pipeline End-to-End de Engenharia de Dados e ML

Previsão de **churn de clientes** de um e-commerce, a partir de duas fontes
de dados reais, armazenadas em PostgreSQL com modelagem relacional,
processadas por um pipeline de ETL em Python, com engenharia de features e
um modelo de classificação treinado e avaliado.

```
Fontes de Dados → PostgreSQL → ETL Python → Features → Machine Learning → Métricas → Resultados
```

Disciplina: Banco de Dados para Engenharia de Dados — Especialização em
Projetos em Inteligência Artificial (UNIFAP). Professor: Adolfo Francesco
de Oliveira Colares. Autor: José Marcel de Oliveira Santos.

## Fontes de dados

1. **Arquivo (CSV)** — `data/clientes.csv` e `data/vendas.csv`: cadastro de
   420 clientes e histórico de vendas de um e-commerce (dados sintéticos,
   gerados por `etl/gen_fonte_csv.py`, com sujeira proposital — nulos,
   duplicatas e outliers — para o ETL ter o que limpar de verdade).
2. **API pública (IBGE)** — `servicodados.ibge.gov.br`, consumida ao vivo
   em `etl/extract_ibge.py`: lista oficial de municípios do Brasil, usada
   para enriquecer cada cliente com município, UF e região geográfica
   oficiais a partir do nome da cidade cadastrada.

## Modelagem no PostgreSQL

Esquema relacional (3FN) em `sql/schema.sql`:

- `dim_cliente` — um registro por cliente, com FK para `dim_localidade`
- `dim_produto` — catálogo de produtos por categoria
- `dim_localidade` — município/UF/região, populada pela Fonte 2 (IBGE)
- `fato_vendas` — uma linha por transação, FK para as três dimensões acima

## Pipeline de ETL (`etl/pipeline_etl.py`)

- **Extract**: lê os dois CSVs da Fonte 1 e consulta a API do IBGE (Fonte 2)
  em tempo real.
- **Transform**: remove duplicatas e linhas sem valor; remove outliers em
  `valor_total` pelo método IQR **calculado por categoria de produto**
  (um notebook de R$4.000 é normal em Eletrônicos, mas pareceria outlier
  se comparado direto com livros de R$50 — IQR global testado e descartado
  por remover compras legítimas); casa o nome da cidade de cada cliente
  com o índice de municípios do IBGE (100% de match neste dataset).
- **Load**: grava tudo no PostgreSQL respeitando o esquema acima.

Log completo de cada execução em `logs/etl.log`.

## Engenharia de features (`etl/feature_engineering.py`)

Lê os dados **já modelados no PostgreSQL** (não o CSV bruto) e aplica um
**split temporal** para evitar vazamento de dados: um corte 90 dias antes
da data mais recente separa o que vira feature (tudo até o corte) do que
vira o rótulo (se o cliente comprou **depois** do corte).

18 features: agregações (total/qtd/ticket médio de compras, categorias
distintas), temporais (dias desde última/primeira compra, dias desde
cadastro) e encoding one-hot (segmento, região do IBGE, categoria
preferida). Documentação de cada feature no notebook, seção 4.

## Modelo (`etl/treinar_modelo.py`)

Três modelos comparados com validação cruzada (5-fold, F1) e avaliados em
um conjunto de teste separado (20%, estratificado):

| Modelo | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **Random Forest** (selecionado) | 0,76 | **0,79** | 0,82 |
| Gradient Boosting | 0,72 | 0,75 | 0,84 |
| Logistic Regression | 0,66 | 0,70 | 0,66 |

`StandardScaler` ajustado **somente no treino** (sem vazamento). Modelo
final salvo em `models/modelo_churn.pkl` (dict com modelo + scaler +
colunas, via `joblib`).

## Estrutura do repositório

```
projeto_integrador_p15/
├── data/                   # CSVs de origem e features.csv gerado
├── etl/
│   ├── gen_fonte_csv.py    # gera a Fonte 1 (sintética)
│   ├── extract_ibge.py     # consome a Fonte 2 (API do IBGE)
│   ├── pipeline_etl.py     # ETL completo: extract -> transform -> load
│   ├── feature_engineering.py
│   └── treinar_modelo.py
├── sql/
│   └── schema.sql          # modelagem relacional do PostgreSQL
├── models/
│   └── modelo_churn.pkl
├── notebooks/
│   └── projeto_integrador.ipynb   # notebook final, já executado
└── logs/
    └── etl.log
```

## Como reproduzir

```bash
pip install pandas numpy sqlalchemy psycopg2-binary requests \
            scikit-learn matplotlib seaborn joblib jupyter nbformat nbclient

# 1. Banco de dados
psql -U postgres -h localhost -c "CREATE DATABASE projeto_integrador_p15;"
psql -U postgres -h localhost -d projeto_integrador_p15 -f sql/schema.sql

# 2. Gerar a Fonte 1 (ou use os CSVs já versionados em data/)
python etl/gen_fonte_csv.py

# 3. Rodar o pipeline completo (ETL -> features -> modelo)
python etl/pipeline_etl.py
python etl/feature_engineering.py
python etl/treinar_modelo.py

# 4. Ou simplesmente abrir e rodar o notebook, que já encadeia tudo isso:
jupyter nbconvert --to notebook --execute notebooks/projeto_integrador.ipynb
```

A string de conexão (`postgresql://postgres:@localhost:5432/projeto_integrador_p15`)
está hardcoded em `pipeline_etl.py` e `feature_engineering.py` — ajuste se
seu PostgreSQL usar outro usuário/senha/porta.

## Decisões técnicas e limitações conhecidas

- **IQR por categoria, não global**: ver seção ETL acima — decisão tomada
  depois de testar a versão global e constatar que descartava vendas
  legítimas de categorias mais caras.
- **Split temporal no split em vez de aleatório**: a engenharia de features
  deste projeto corrige deliberadamente um problema de vazamento de dados
  identificado na atividade anterior do módulo (P14): lá, a mesma variável
  de recência era usada como feature e como base do target, inflando as
  métricas artificialmente (F1 chegava a 1.0). Aqui, features e target
  vêm de janelas de tempo disjuntas, e as métricas resultantes (F1 ≈ 0,79)
  são mais baixas, mas também mais confiáveis.
- **Dataset sintético**: os CSVs de origem são gerados, não dados reais de
  um e-commerce de verdade — a Fonte 2 (IBGE), porém, é 100% real e
  consumida ao vivo via HTTP a cada execução do ETL.
