"""
Fonte 1 de dados: CSV sintetico de clientes e vendas de um e-commerce
(mesmo dominio usado nas atividades anteriores da disciplina: Amazonia
Digital). Gerado uma unica vez e versionado em data/ - a "coleta" real
deste pipeline e o extract_csv.py, que apenas le o que esta aqui.
"""
import random
import csv
import os
from datetime import date, timedelta

random.seed(2026)

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

NOMES = ["Ana", "Bruno", "Carla", "Diego", "Elaine", "Fabio", "Gabriela", "Hugo",
         "Isabela", "Joao", "Karina", "Lucas", "Mariana", "Nelson", "Otavia",
         "Paulo", "Queila", "Rafael", "Sonia", "Thiago", "Ursula", "Vitor",
         "Wesley", "Ximena", "Yago", "Zelia"]

CIDADES = ["Macapa", "Belem", "Manaus", "Santarem", "Boa Vista", "Sao Paulo",
           "Rio de Janeiro", "Belo Horizonte", "Salvador", "Fortaleza"]

SEGMENTOS = ["Premium", "Regular", "VIP"]

PRODUTOS = [
    ("Notebook Gamer", "Eletronicos", 4200.00),
    ("Mouse sem fio", "Eletronicos", 89.90),
    ("Teclado mecanico", "Eletronicos", 320.00),
    ("Monitor 27pol", "Eletronicos", 1350.00),
    ("Webcam HD", "Eletronicos", 180.00),
    ("Cadeira ergonomica", "Moveis", 980.00),
    ("Mesa para escritorio", "Moveis", 650.00),
    ("Estante de livros", "Moveis", 420.00),
    ("Luminaria de mesa", "Moveis", 150.00),
    ("Livro de ficcao", "Livros", 54.90),
    ("Livro tecnico", "Livros", 120.00),
    ("Livro infantil", "Livros", 39.90),
    ("Mochila executiva", "Acessorios", 210.00),
    ("Garrafa termica", "Acessorios", 75.00),
    ("Oculos de sol", "Acessorios", 160.00),
    ("Kit papelaria", "Escritorio", 65.00),
    ("Organizador de mesa", "Escritorio", 45.00),
    ("Impressora multifuncional", "Escritorio", 890.00),
]

HOJE = date(2026, 10, 1)
N_CLIENTES = 420

# ---- clientes.csv ----
clientes = []
for cid in range(1, N_CLIENTES + 1):
    nome = random.choice(NOMES)
    sobrenome_idx = cid
    cidade = random.choice(CIDADES)
    segmento = random.choices(SEGMENTOS, weights=[0.2, 0.6, 0.2])[0]
    dias_cadastro = random.randint(60, 900)
    data_cadastro = HOJE - timedelta(days=dias_cadastro)
    clientes.append({
        "cliente_id": cid,
        "nome": f"{nome} Cliente{sobrenome_idx}",
        "email": f"{nome.lower()}{sobrenome_idx}@exemplo.com",
        "cidade": cidade,
        "segmento": segmento,
        "data_cadastro": data_cadastro.isoformat(),
    })

with open(os.path.join(DATA, "clientes.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["cliente_id", "nome", "email", "cidade", "segmento", "data_cadastro"])
    w.writeheader()
    w.writerows(clientes)

# ---- vendas.csv (transacoes) ----
# Cada cliente tem uma propensao propria de continuar comprando (usada so
# para gerar dados realistas; o pipeline NAO tem acesso a essa variavel).
vendas = []
venda_id = 1
for c in clientes:
    cliente_id = c["cliente_id"]
    data_cadastro = date.fromisoformat(c["data_cadastro"])
    dias_disponiveis = (HOJE - data_cadastro).days

    propensao_ativa = random.random() < 0.6
    n_compras = random.randint(1, 16)

    for _ in range(n_compras):
        if propensao_ativa:
            # cliente tende a comprar espalhado por todo o periodo, incluindo recente
            dias_atras = random.randint(0, dias_disponiveis)
        else:
            # cliente tende a nao comprar nos ultimos ~5 meses (vai gerar churn)
            limite_min = min(150, dias_disponiveis)
            dias_atras = random.randint(limite_min, dias_disponiveis) if dias_disponiveis > limite_min else dias_disponiveis

        data_venda = HOJE - timedelta(days=dias_atras)
        if data_venda < data_cadastro:
            data_venda = data_cadastro

        produto = random.choice(PRODUTOS)
        qtd = random.randint(1, 3)
        valor_unit = round(produto[2] * random.uniform(0.92, 1.08), 2)

        vendas.append({
            "venda_id": venda_id,
            "cliente_id": cliente_id,
            "produto_nome": produto[0],
            "categoria": produto[1],
            "data": data_venda.isoformat(),
            "quantidade": qtd,
            "valor_unitario": valor_unit,
            "valor_total": round(valor_unit * qtd, 2),
        })
        venda_id += 1

# sujeira proposital (realista, para o ETL ter o que limpar)
for v in random.sample(vendas, 60):
    v["valor_total"] = ""
for v in random.sample([v for v in vendas if v["valor_total"] != ""], 25):
    v["valor_total"] = round(float(v["valor_total"]) * random.uniform(15, 30), 2)
dup = random.sample(vendas, 45)
vendas.extend([dict(v) for v in dup])
random.shuffle(vendas)

with open(os.path.join(DATA, "vendas.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["venda_id", "cliente_id", "produto_nome", "categoria",
                                       "data", "quantidade", "valor_unitario", "valor_total"])
    w.writeheader()
    w.writerows(vendas)

print(f"clientes.csv: {len(clientes)} linhas")
print(f"vendas.csv: {len(vendas)} linhas (com sujeira proposital: nulos, outliers, duplicatas)")
