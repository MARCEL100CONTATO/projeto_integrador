"""
Engenharia de features para predicao de churn, lendo diretamente do
PostgreSQL (nao do CSV bruto - o objetivo e provar que o dado ja
armazenado/modelado no banco e reutilizavel para ML).

Split TEMPORAL, sem vazamento de dados (diferente do P13/P14 do Modulo 5,
onde a feature de recencia e o criterio do target coincidiam de proposito
para ilustrar o problema): um corte no tempo separa o que o modelo pode
"ver" (features) do que ele precisa prever (se o cliente comprou DEPOIS
do corte).

    |------- janela de observacao (features) -------|-- janela de label --|
    data_cadastro                                 CORTE                 HOJE
                                                (hoje - 90 dias)

- Features: calculadas SOMENTE com vendas ate CORTE (inclusive).
- Target: 1 (churn) se o cliente NAO comprou depois do CORTE; 0 caso
  contrario. So entram clientes que ja tinham ao menos 1 compra ate o
  corte (nao da pra prever churn de quem nunca comprou).
"""
import os

import pandas as pd
from sqlalchemy import create_engine

BASE = os.path.dirname(os.path.dirname(__file__))
DB_URL = "postgresql://postgres:@localhost:5432/projeto_integrador_p15"
JANELA_LABEL_DIAS = 90


def carregar_dados(engine):
    clientes = pd.read_sql("""
        SELECT c.cliente_id, c.segmento, c.data_cadastro, l.uf_sigla, l.regiao
        FROM dim_cliente c
        LEFT JOIN dim_localidade l ON c.localidade_id = l.id
    """, engine, parse_dates=["data_cadastro"])

    vendas = pd.read_sql("""
        SELECT f.cliente_id, f.data, f.quantidade, f.valor_total, p.categoria
        FROM fato_vendas f
        JOIN dim_produto p ON f.produto_id = p.id
    """, engine, parse_dates=["data"])

    return clientes, vendas


def construir_features(clientes: pd.DataFrame, vendas: pd.DataFrame):
    data_max = vendas["data"].max()
    corte = data_max - pd.Timedelta(days=JANELA_LABEL_DIAS)

    obs = vendas[vendas["data"] <= corte]
    label_window = vendas[vendas["data"] > corte]

    agg = obs.groupby("cliente_id").agg(
        total_compras=("valor_total", "sum"),
        qtd_compras=("valor_total", "count"),
        ticket_medio=("valor_total", "mean"),
        dias_desde_ultima_compra=("data", lambda s: (corte - s.max()).days),
        dias_desde_primeira_compra=("data", lambda s: (corte - s.min()).days),
        categorias_distintas=("categoria", "nunique"),
    ).reset_index()

    agg = agg.merge(clientes, on="cliente_id", how="left")
    agg["dias_desde_cadastro"] = (corte - agg["data_cadastro"]).dt.days

    categoria_preferida = (
        obs.groupby(["cliente_id", "categoria"])["valor_total"].sum()
        .reset_index()
        .sort_values("valor_total", ascending=False)
        .drop_duplicates("cliente_id")[["cliente_id", "categoria"]]
        .rename(columns={"categoria": "categoria_preferida"})
    )
    agg = agg.merge(categoria_preferida, on="cliente_id", how="left")

    clientes_com_compra_apos_corte = set(label_window["cliente_id"].unique())
    agg["churn"] = (~agg["cliente_id"].isin(clientes_com_compra_apos_corte)).astype(int)

    agg = pd.get_dummies(agg, columns=["segmento", "regiao", "categoria_preferida"], dummy_na=False)
    agg = agg.drop(columns=["uf_sigla", "data_cadastro"])

    return agg, corte, data_max


if __name__ == "__main__":
    engine = create_engine(DB_URL)
    clientes, vendas = carregar_dados(engine)
    features, corte, data_max = construir_features(clientes, vendas)

    print(f"Data maxima nas vendas: {data_max.date()}")
    print(f"Corte (ultimos {JANELA_LABEL_DIAS} dias viram janela de label): {corte.date()}")
    print(f"Clientes com features (ao menos 1 compra ate o corte): {len(features)}")
    print(f"Taxa de churn: {features['churn'].mean():.2%}")
    print(f"Total de colunas (features + id + target): {features.shape[1]}")
    print(f"Features: {[c for c in features.columns if c not in ('cliente_id', 'churn')]}")

    out = os.path.join(BASE, "data", "features.csv")
    features.to_csv(out, index=False)
    print(f"Salvo em: {out}")
