-- Projeto Integrador P15 - modelagem relacional (3FN) para o pipeline de
-- vendas + enriquecimento geografico via IBGE.
-- dim_localidade e populada pela Fonte 2 (API do IBGE); as demais, pela
-- Fonte 1 (CSV) apos a limpeza do ETL.

DROP TABLE IF EXISTS fato_vendas CASCADE;
DROP TABLE IF EXISTS dim_cliente CASCADE;
DROP TABLE IF EXISTS dim_produto CASCADE;
DROP TABLE IF EXISTS dim_localidade CASCADE;

CREATE TABLE dim_localidade (
    id SERIAL PRIMARY KEY,
    ibge_municipio_id INTEGER UNIQUE,
    municipio VARCHAR(120) NOT NULL,
    uf_sigla CHAR(2) NOT NULL,
    uf_nome VARCHAR(60) NOT NULL,
    regiao VARCHAR(30) NOT NULL
);

CREATE TABLE dim_cliente (
    id SERIAL PRIMARY KEY,
    cliente_id INTEGER UNIQUE NOT NULL,
    nome VARCHAR(150) NOT NULL,
    email VARCHAR(150),
    cidade VARCHAR(120) NOT NULL,
    segmento VARCHAR(20) NOT NULL,
    data_cadastro DATE NOT NULL,
    localidade_id INTEGER REFERENCES dim_localidade(id)
);

CREATE TABLE dim_produto (
    id SERIAL PRIMARY KEY,
    produto_nome VARCHAR(120) UNIQUE NOT NULL,
    categoria VARCHAR(60) NOT NULL
);

CREATE TABLE fato_vendas (
    id SERIAL PRIMARY KEY,
    venda_id INTEGER UNIQUE NOT NULL,
    cliente_id INTEGER NOT NULL REFERENCES dim_cliente(cliente_id),
    produto_id INTEGER NOT NULL REFERENCES dim_produto(id),
    data DATE NOT NULL,
    quantidade INTEGER NOT NULL CHECK (quantidade > 0),
    valor_unitario NUMERIC(10, 2) NOT NULL CHECK (valor_unitario > 0),
    valor_total NUMERIC(10, 2) NOT NULL CHECK (valor_total > 0)
);

CREATE INDEX idx_fato_vendas_cliente ON fato_vendas(cliente_id);
CREATE INDEX idx_fato_vendas_data ON fato_vendas(data);
CREATE INDEX idx_dim_cliente_localidade ON dim_cliente(localidade_id);
