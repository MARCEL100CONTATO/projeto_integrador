"""
ETL completo do Projeto Integrador (P15).

EXTRACT  -> le data/clientes.csv e data/vendas.csv (Fonte 1) e consulta a
            API publica do IBGE em tempo real (Fonte 2).
TRANSFORM -> limpa vendas (duplicatas, nulos, outliers via IQR), enriquece
            clientes com municipio/UF/regiao oficiais do IBGE, casa nomes
            de produto com categoria.
LOAD     -> grava tudo em PostgreSQL (projeto_integrador_p15), respeitando
            o esquema em sql/schema.sql.

Rode a partir da raiz do projeto: python etl/pipeline_etl.py
"""
import logging
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, os.path.dirname(__file__))
from extract_ibge import extract_municipios, monta_indice_por_nome, _normaliza  # noqa: E402

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data")
LOGS = os.path.join(BASE, "logs")
os.makedirs(LOGS, exist_ok=True)

DB_URL = "postgresql://postgres:@localhost:5432/projeto_integrador_p15"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS, "etl.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("etl")


def extract():
    log.info("EXTRACT: lendo Fonte 1 (CSV) - clientes.csv e vendas.csv")
    clientes = pd.read_csv(os.path.join(DATA, "clientes.csv"), parse_dates=["data_cadastro"])
    vendas = pd.read_csv(os.path.join(DATA, "vendas.csv"), parse_dates=["data"])
    log.info(f"  clientes.csv: {len(clientes)} linhas | vendas.csv: {len(vendas)} linhas")

    log.info("EXTRACT: consultando Fonte 2 (API publica do IBGE, ao vivo)")
    municipios = extract_municipios()
    indice_ibge = monta_indice_por_nome(municipios)
    log.info(f"  {len(municipios)} municipios recebidos da API, {len(indice_ibge)} indexados por nome")

    return clientes, vendas, indice_ibge


def transform(clientes: pd.DataFrame, vendas: pd.DataFrame, indice_ibge: dict):
    log.info("TRANSFORM: limpando vendas")
    antes = len(vendas)
    vendas = vendas.drop_duplicates()
    log.info(f"  duplicatas removidas: {antes - len(vendas)}")

    antes = len(vendas)
    vendas = vendas.dropna(subset=["valor_total"])
    log.info(f"  linhas sem valor_total removidas: {antes - len(vendas)}")

    vendas["valor_total"] = pd.to_numeric(vendas["valor_total"], errors="coerce")
    vendas = vendas.dropna(subset=["valor_total"])

    # IQR calculado POR CATEGORIA, nao globalmente: um notebook de R$4.000 e
    # normal em Eletronicos mas pareceria um outlier estatistico se comparado
    # direto com livros de R$50 - aplicar o IQR no dataset inteiro misturado
    # descartava compras legitimas de categorias mais caras (testado: IQR
    # global removia 489/3641 linhas, a maioria compras validas de
    # Eletronicos/Moveis; por categoria, cai para o que de fato sao outliers
    # dentro de cada faixa de preco esperada).
    antes = len(vendas)

    q1 = vendas.groupby("categoria")["valor_total"].transform(lambda s: s.quantile(0.25))
    q3 = vendas.groupby("categoria")["valor_total"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    vendas = vendas[(vendas["valor_total"] >= q1 - 1.5 * iqr) & (vendas["valor_total"] <= q3 + 1.5 * iqr)]
    log.info(f"  outliers removidos (IQR por categoria em valor_total): {antes - len(vendas)}")
    log.info(f"  vendas limpas: {len(vendas)} linhas")

    log.info("TRANSFORM: enriquecendo clientes com dados oficiais do IBGE")
    clientes = clientes.copy()
    geo = clientes["cidade"].apply(lambda c: indice_ibge.get(_normaliza(c)))
    clientes["ibge_municipio_id"] = geo.apply(lambda g: g["ibge_municipio_id"] if g else None)
    clientes["municipio_oficial"] = geo.apply(lambda g: g["municipio"] if g else None)
    clientes["uf_sigla"] = geo.apply(lambda g: g["uf_sigla"] if g else None)
    clientes["uf_nome"] = geo.apply(lambda g: g["uf_nome"] if g else None)
    clientes["regiao"] = geo.apply(lambda g: g["regiao"] if g else None)
    casados = clientes["ibge_municipio_id"].notna().sum()
    log.info(f"  clientes casados com municipio do IBGE: {casados}/{len(clientes)}")

    produtos = vendas[["produto_nome", "categoria"]].drop_duplicates().reset_index(drop=True)
    log.info(f"  catalogo de produtos derivado: {len(produtos)} produtos distintos")

    return clientes, vendas, produtos


def load(clientes: pd.DataFrame, vendas: pd.DataFrame, produtos: pd.DataFrame):
    log.info("LOAD: gravando no PostgreSQL (projeto_integrador_p15)")
    engine = create_engine(DB_URL)

    localidades = (
        clientes[["ibge_municipio_id", "municipio_oficial", "uf_sigla", "uf_nome", "regiao"]]
        .dropna(subset=["ibge_municipio_id"])
        .drop_duplicates(subset=["ibge_municipio_id"])
        .rename(columns={"municipio_oficial": "municipio"})
    )
    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE fato_vendas, dim_cliente, dim_produto, dim_localidade RESTART IDENTITY CASCADE;")

    localidades.to_sql("dim_localidade", engine, if_exists="append", index=False)
    log.info(f"  dim_localidade: {len(localidades)} linhas")

    loc_map = pd.read_sql("SELECT id, ibge_municipio_id FROM dim_localidade", engine)
    clientes_db = clientes.merge(loc_map, on="ibge_municipio_id", how="left", suffixes=("", "_loc"))
    clientes_db = clientes_db.rename(columns={"id": "localidade_id"})[
        ["cliente_id", "nome", "email", "cidade", "segmento", "data_cadastro", "localidade_id"]
    ]
    clientes_db.to_sql("dim_cliente", engine, if_exists="append", index=False)
    log.info(f"  dim_cliente: {len(clientes_db)} linhas")

    produtos.rename(columns={"produto_nome": "produto_nome"}).to_sql("dim_produto", engine, if_exists="append", index=False)
    log.info(f"  dim_produto: {len(produtos)} linhas")

    prod_map = pd.read_sql("SELECT id, produto_nome FROM dim_produto", engine)
    vendas_db = vendas.merge(prod_map, on="produto_nome", how="left").rename(columns={"id": "produto_id"})
    vendas_db = vendas_db[["venda_id", "cliente_id", "produto_id", "data", "quantidade", "valor_unitario", "valor_total"]]
    vendas_db.to_sql("fato_vendas", engine, if_exists="append", index=False)
    log.info(f"  fato_vendas: {len(vendas_db)} linhas")

    return engine


def run():
    clientes, vendas, indice_ibge = extract()
    clientes_t, vendas_t, produtos_t = transform(clientes, vendas, indice_ibge)
    load(clientes_t, vendas_t, produtos_t)
    log.info("ETL concluido com sucesso.")


if __name__ == "__main__":
    run()
