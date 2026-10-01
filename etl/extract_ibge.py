"""
Fonte 2 de dados: API publica do IBGE (servicodados.ibge.gov.br), consumida
ao vivo (nao e um arquivo estatico) - enriquece cada cliente com o
municipio oficial, UF e regiao geografica do IBGE a partir do nome da
cidade cadastrada.
"""
import requests
import unicodedata


def _normaliza(txt: str) -> str:
    """Remove acentos e padroniza caixa, para casar 'Sao Paulo' com 'São Paulo'."""
    nfkd = unicodedata.normalize("NFKD", txt)
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    return sem_acento.strip().lower()


def extract_municipios() -> list[dict]:
    """Baixa a lista completa de municipios do Brasil (fonte oficial, ao vivo)."""
    url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def monta_indice_por_nome(municipios: list[dict]) -> dict:
    """Chave: nome normalizado -> dict com municipio/UF/regiao (primeira ocorrencia)."""
    indice = {}
    for m in municipios:
        chave = _normaliza(m["nome"])
        if chave in indice:
            continue
        if m.get("microrregiao"):
            uf = m["microrregiao"]["mesorregiao"]["UF"]
        else:
            uf = m["regiao-imediata"]["regiao-intermediaria"]["UF"]
        regiao = uf["regiao"]
        indice[chave] = {
            "ibge_municipio_id": m["id"],
            "municipio": m["nome"],
            "uf_sigla": uf["sigla"],
            "uf_nome": uf["nome"],
            "regiao": regiao["nome"],
        }
    return indice


if __name__ == "__main__":
    municipios = extract_municipios()
    print(f"Municipios recebidos da API do IBGE: {len(municipios)}")
    indice = monta_indice_por_nome(municipios)
    print(f"Indice por nome (normalizado) construido: {len(indice)} entradas")
    print("Exemplo:", indice.get(_normaliza("Macapa")))
