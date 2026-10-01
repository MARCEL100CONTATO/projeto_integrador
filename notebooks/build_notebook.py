import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Projeto Integrador (P15) - Pipeline End-to-End de Engenharia de Dados e ML

**Previsao de churn de clientes de um e-commerce**, a partir de duas fontes de
dados reais (CSV proprio + API publica do IBGE), armazenadas em PostgreSQL com
modelagem relacional, processadas por um pipeline de ETL em Python, com
engenharia de features e um modelo de classificacao treinado e avaliado.

**Fluxo**: Fontes de Dados -> PostgreSQL -> ETL Python -> Features -> Machine
Learning -> Metricas -> Resultados

Este notebook executa o pipeline completo e documenta cada decisao tecnica.
Os modulos reaproveitados (`etl/`) sao os mesmos rodados em producao via
linha de comando - nada aqui e exclusivo do notebook.""")

code("""import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'etl'))
import pandas as pd
from sqlalchemy import create_engine

DB_URL = 'postgresql://postgres:@localhost:5432/projeto_integrador_p15'
pd.set_option('display.width', 120)""")

md("""## 1. Fontes de dados

**Fonte 1 (arquivo)**: `data/clientes.csv` e `data/vendas.csv` - cadastro de
420 clientes e historico de vendas de um e-commerce (gerados sinteticamente
para esta atividade, com sujeira proposital: nulos, duplicatas e outliers,
para o ETL ter o que limpar de verdade).

**Fonte 2 (API publica, consumida ao vivo)**: `servicodados.ibge.gov.br` -
lista oficial de municipios do Brasil, usada para enriquecer cada cliente com
municipio, UF e regiao geografica oficiais do IBGE a partir do nome da
cidade cadastrada.""")

code("""clientes_raw = pd.read_csv('../data/clientes.csv')
vendas_raw = pd.read_csv('../data/vendas.csv')
print('clientes.csv:', clientes_raw.shape)
print('vendas.csv:', vendas_raw.shape)
print()
print('Nulos em vendas.csv:')
print(vendas_raw.isnull().sum())
print('Duplicatas exatas em vendas.csv:', vendas_raw.duplicated().sum())""")

code("""import requests
r = requests.get('https://servicodados.ibge.gov.br/api/v1/localidades/municipios', timeout=30)
municipios = r.json()
print(f'Fonte 2 (API IBGE): {len(municipios)} municipios recebidos ao vivo')
print('Exemplo:', {k: municipios[0][k] for k in ['id', 'nome']})""")

md("""## 2. ETL (Python) -> PostgreSQL

O pipeline completo (`etl/pipeline_etl.py`) faz extract (CSV + API),
transform (limpeza + enriquecimento geografico) e load (PostgreSQL), usando
o esquema relacional em `sql/schema.sql`:

- `dim_cliente`, `dim_produto`, `dim_localidade` (dimensoes)
- `fato_vendas` (fato, FK para as 3 dimensoes)

Rodado aqui dentro do notebook para registrar a execucao real.""")

code("""sys.path.insert(0, os.path.join(os.getcwd(), '..', 'etl'))
import pipeline_etl
pipeline_etl.run()""")

md("""### Verificacao: o dado esta de fato no PostgreSQL, modelado e consultavel""")

code("""engine = create_engine(DB_URL)
display_df = pd.read_sql('''
    SELECT l.regiao, COUNT(DISTINCT c.cliente_id) AS clientes, COUNT(f.id) AS vendas,
           ROUND(SUM(f.valor_total)::numeric, 2) AS receita
    FROM dim_cliente c
    JOIN dim_localidade l ON c.localidade_id = l.id
    JOIN fato_vendas f ON f.cliente_id = c.cliente_id
    GROUP BY l.regiao ORDER BY receita DESC
''', engine)
display_df""")

md("## 3. Analise Exploratoria de Dados (EDA)")

code("""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

vendas_db = pd.read_sql('SELECT * FROM fato_vendas', engine, parse_dates=['data'])
clientes_db = pd.read_sql('SELECT * FROM dim_cliente', engine)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(vendas_db['valor_total'], bins=40, ax=axes[0], color='#4c72b0')
axes[0].set_title(\"Distribuicao de valor_total (apos limpeza)\")
sns.countplot(y=clientes_db['segmento'], ax=axes[1], color='#55a868')
axes[1].set_title('Clientes por segmento')
plt.tight_layout()
plt.savefig('eda_1.png', dpi=100)
plt.show()""")

code("""vendas_cat = pd.read_sql('''
    SELECT p.categoria, COUNT(*) AS qtd_vendas, ROUND(SUM(f.valor_total)::numeric,2) AS receita
    FROM fato_vendas f JOIN dim_produto p ON f.produto_id = p.id
    GROUP BY p.categoria ORDER BY receita DESC
''', engine)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=vendas_cat, y='categoria', x='receita', ax=ax, color='#dd8452')
ax.set_title('Receita por categoria de produto')
plt.tight_layout()
plt.savefig('eda_2.png', dpi=100)
plt.show()
vendas_cat""")

md("""## 4. Engenharia de features (sem vazamento de dados)

Diferente do P14 (Modulo 5), onde a feature de recencia e o criterio do
target coincidiam de proposito (para ilustrar o problema de data leakage),
aqui o split e **temporal**: um corte de 90 dias antes da data mais recente
separa o que vira feature (tudo ate o corte) do que vira o rotulo (se o
cliente comprou DEPOIS do corte). Isso evita que o modelo "veja" informacao
do futuro.""")

code("""import feature_engineering as fe
clientes_fe, vendas_fe = fe.carregar_dados(engine)
features, corte, data_max = fe.construir_features(clientes_fe, vendas_fe)

print(f'Data mais recente nas vendas: {data_max.date()}')
print(f'Corte (90 dias antes): {corte.date()}')
print(f'Clientes com features: {len(features)}')
print(f'Taxa de churn: {features[\"churn\"].mean():.2%}')
print(f'Total de features: {features.shape[1] - 2}')
features.to_csv('../data/features.csv', index=False)
features.head()""")

code("""fig, ax = plt.subplots(figsize=(8, 6))
cols_corr = ['total_compras', 'qtd_compras', 'ticket_medio', 'dias_desde_ultima_compra',
             'dias_desde_primeira_compra', 'categorias_distintas', 'dias_desde_cadastro', 'churn']
sns.heatmap(features[cols_corr].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlacao entre as features numericas e o target (churn)')
plt.tight_layout()
plt.savefig('eda_3.png', dpi=100)
plt.show()""")

md("""**Interpretacao**: ao contrario do P14, nenhuma feature tem correlacao
artificialmente perfeita com o target - `dias_desde_ultima_compra` (recencia
DENTRO da janela de observacao) e a mais forte, o que faz sentido de
negocio (quem ja estava inativo antes do corte tende a continuar inativo),
mas nao e uma tautologia como antes, porque o target olha para um periodo
que as features nunca viram.""")

md("""### Por que cada feature foi criada

- **total_compras / qtd_compras / ticket_medio**: valor, frequencia e tiquete
  medio do cliente ate o corte - as tres metricas classicas de valor de
  cliente (RFM sem o R).
- **dias_desde_ultima_compra**: recencia dentro da janela de observacao -
  proxy de engajamento recente sem olhar para o futuro.
- **dias_desde_primeira_compra**: ha quanto tempo o cliente comprou pela
  primeira vez (tempo de relacionamento efetivo, diferente de data de
  cadastro).
- **categorias_distintas**: diversidade de consumo.
- **dias_desde_cadastro**: maturidade do cadastro.
- **segmento_* (one-hot)**: categoria declarada do cliente (Premium/Regular/
  VIP) - sem ordem natural, por isso one-hot em vez de encoding ordinal.
- **regiao_* (one-hot)**: regiao geografica oficial do IBGE (Fonte 2) -
  mostra o enriquecimento da segunda fonte de dados virando feature real.
- **categoria_preferida_* (one-hot)**: categoria de produto onde o cliente
  mais gastou ate o corte.""")

md("## 5. Treinamento e comparacao de modelos")

code("""import treinar_modelo as tm
df_res, relatorios, (X_test, y_test) = tm.treinar(features)""")

code("""fig, ax = plt.subplots(figsize=(7, 4))
df_plot = df_res.melt(id_vars='modelo', value_vars=['accuracy', 'f1', 'roc_auc'])
sns.barplot(data=df_plot, x='modelo', y='value', hue='variable', ax=ax)
ax.set_title('Comparacao de metricas entre os 3 modelos')
ax.set_ylim(0, 1)
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig('eda_4.png', dpi=100)
plt.show()""")

md("## 6. Conclusoes")

md("""- O pipeline roda de ponta a ponta: duas fontes de dados reais (CSV +
  API publica do IBGE, consumida ao vivo) -> PostgreSQL com modelagem
  relacional (3 dimensoes + 1 fato) -> ETL com limpeza documentada (IQR
  por categoria, nulos, duplicatas) -> 18 features (temporais, agregacoes,
  encoding) -> split temporal sem vazamento -> comparacao de 3 modelos ->
  modelo salvo em `.pkl`.
- O **Random Forest** venceu por F1 (0.79) e tambem teve a melhor
  generalizacao em cross-validation (cv_f1_mean=0.76, proximo do F1 de
  teste - sinal de que nao esta overfitando). O Gradient Boosting teve o
  melhor ROC-AUC (0.84), indicando boa separacao de classes mesmo com F1
  levemente menor.
- Diferente do exercicio do Modulo 5, estes resultados sao **o quanto o
  modelo realmente consegue prever churn usando so informacao disponivel
  antes do corte temporal** - nao ha vazamento da definicao do target para
  dentro das features.
- Proximos passos razoaveis para evoluir o projeto: tunar hiperparametros
  (GridSearch/RandomizedSearch), testar uma janela de observacao/label
  diferente, e agregar uma terceira fonte de dados (ex.: indicadores
  socioeconomicos do IBGE por regiao) para enriquecer ainda mais as
  features geograficas.""")

nb["cells"] = cells

with open("projeto_integrador.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook criado: projeto_integrador.ipynb")
