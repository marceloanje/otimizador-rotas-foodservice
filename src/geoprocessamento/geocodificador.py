"""Geocodificação de endereços via Nominatim (OpenStreetMap).

Funções principais
------------------
geocodificar_endereco(endereco) -> tuple[float, float] | None
    Geocodifica uma string de endereço. Usa cache local para evitar
    requisições repetidas ao serviço externo.

geocodificar_dataframe(df) -> pd.DataFrame
    Geocodifica todas as linhas de um DataFrame que ainda não possuem
    colunas `lat` e `lon`. Retorna o DataFrame com essas colunas adicionadas.

Cache
-----
Armazenado em `src/dados/cache_geocodificacao.json` como um dicionário
{endereço_normalizado: [lat, lon]}. Persistido após cada nova geocodificação.
"""

import json
import os
import time
import pandas as pd

# Caminho do cache relativo a este arquivo
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "dados", "cache_geocodificacao.json")
_CACHE_PATH = os.path.normpath(_CACHE_PATH)

# Intervalo mínimo entre requisições ao Nominatim (política de uso)
_DELAY_SEGUNDOS = 1.1


def _carregar_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _salvar_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def geocodificar_endereco(endereco: str) -> tuple[float, float] | None:
    """Geocodifica um endereço e retorna (lat, lon) ou None se não encontrado.

    Consulta o cache local antes de fazer requisição ao Nominatim.
    Novas geocodificações são persistidas no cache automaticamente.

    Parâmetros
    ----------
    endereco : str
        String de endereço completa, ex:
        "Avenida Afonso Pena, 2185, Centro, Belo Horizonte - MG, Brasil"
    """
    from geopy.geocoders import Nominatim

    chave = endereco.strip().upper()
    cache = _carregar_cache()

    if chave in cache:
        coords = cache[chave]
        return (coords[0], coords[1])

    geolocator = Nominatim(user_agent="otimizador-rotas-foodservice-tcc")
    time.sleep(_DELAY_SEGUNDOS)

    try:
        location = geolocator.geocode(endereco, timeout=10)
    except Exception:
        return None

    if location is None:
        return None

    resultado = (location.latitude, location.longitude)
    cache[chave] = list(resultado)
    _salvar_cache(cache)
    return resultado


def geocodificar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas `lat` e `lon` a um DataFrame de pedidos.

    Constrói a string de endereço a partir das colunas:
    `endereco_entrega`, `numero_endereco`, `bairro_entrega`, `municipio_entrega`.

    Linhas onde a geocodificação falhar recebem lat/lon = NaN.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame lido de pedidos.csv (sem colunas lat/lon).

    Retorna
    -------
    pd.DataFrame
        Mesmo DataFrame com colunas `lat` e `lon` adicionadas.
    """
    lats = []
    lons = []

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        endereco = _montar_endereco(row)
        print(f"  [{i}/{total}] Geocodificando: {endereco}")
        coords = geocodificar_endereco(endereco)
        if coords is not None:
            lats.append(coords[0])
            lons.append(coords[1])
        else:
            print(f"    AVISO: geocodificação falhou para '{endereco}'")
            lats.append(float("nan"))
            lons.append(float("nan"))

    df = df.copy()
    df["lat"] = lats
    df["lon"] = lons
    return df


def _montar_endereco(row: pd.Series) -> str:
    """Constrói string de endereço para geocodificação a partir de uma linha do DataFrame."""
    partes = []
    for col in ("endereco_entrega", "numero_endereco", "bairro_entrega", "municipio_entrega"):
        val = str(row.get(col, "")).strip()
        if val and val.lower() != "nan":
            partes.append(val)
    partes.append("Brasil")
    return ", ".join(partes)
