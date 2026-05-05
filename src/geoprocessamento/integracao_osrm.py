import math
import numpy as np
import requests


def obter_matriz_osrm(posicoes, timeout=30, max_table_size=100):
    """Calcula a matriz de distâncias reais via OSRM com suporte a batching.

    Usa a URL configurada em `config.OSRM_URL`. Para rodar localmente,
    suba o servidor OSRM via Docker apontando para os dados de MG e
    mantenha `OSRM_URL = "http://localhost:5000"` em config.py.

    Quando o número de nós excede sqrt(max_table_size), a matriz é calculada
    em blocos 2D para respeitar o limite do servidor (sources × destinations
    ≤ max_table_size). O valor padrão de max_table_size=100 corresponde ao
    padrão do OSRM (--max-table-size 100).

    Parâmetros
    ----------
    posicoes : list[tuple[float, float]]
        Lista de tuplas (lat, lon). Índice 0 é o depósito.
    timeout : int
        Timeout em segundos por requisição HTTP.
    max_table_size : int
        Limite de entradas por requisição (sources × destinations).

    Retorna
    -------
    tuple[numpy.ndarray, numpy.ndarray | None]
        ``(matriz_distancias, matriz_tempos)`` onde distâncias estão em metros
        e tempos em minutos. ``matriz_tempos`` é None se o OSRM não retornar
        durações.

    Raises
    ------
    RuntimeError
        Se o OSRM retornar código diferente de "Ok".
    requests.exceptions.RequestException
        Se uma requisição HTTP falhar.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import OSRM_URL

    n = len(posicoes)
    # OSRM exige coordenadas no formato lon,lat (não lat,lon)
    coords_str = ";".join(f"{lon},{lat}" for lat, lon in posicoes)
    base_url = f"{OSRM_URL}/table/v1/driving/{coords_str}"

    # Requisição única quando a matriz inteira cabe no limite
    if n * n <= max_table_size:
        resp = requests.get(base_url, params={"annotations": "distance,duration"}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok":
            raise RuntimeError(f"OSRM retornou código inesperado: {data.get('code')}")
        return _extrair_matrizes(data, avisar_sem_dist=True)

    # Batching 2D: batch_size² ≤ max_table_size
    batch = max(1, math.isqrt(max_table_size))
    indices_src = list(range(0, n, batch))
    indices_dst = list(range(0, n, batch))
    total_batches = len(indices_src) * len(indices_dst)
    print(f"  OSRM: {n} nós, usando {total_batches} batches (blocos {batch}×{batch})...")

    matriz_dist = np.zeros((n, n))
    matriz_tempo = np.zeros((n, n))
    tem_tempo = True
    aviso_sem_dist_dado = False

    for i_idx, i_start in enumerate(indices_src):
        src = list(range(i_start, min(i_start + batch, n)))
        for j_idx, j_start in enumerate(indices_dst):
            dst = list(range(j_start, min(j_start + batch, n)))
            atual = i_idx * len(indices_dst) + j_idx + 1
            print(f"  OSRM batch {atual}/{total_batches}...", end="\r")

            params = {
                "sources": ";".join(map(str, src)),
                "destinations": ";".join(map(str, dst)),
                "annotations": "distance,duration",
            }
            resp = requests.get(base_url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "Ok":
                raise RuntimeError(f"OSRM retornou código inesperado: {data.get('code')}")

            if "distances" in data:
                matriz_dist[np.ix_(src, dst)] = np.array(data["distances"], dtype=float)
            else:
                if not aviso_sem_dist_dado:
                    print("\nAVISO: OSRM não retornou 'distances'. Estimando a partir de 'durations' (40 km/h).")
                    aviso_sem_dist_dado = True
                durations = np.array(data["durations"], dtype=float)
                matriz_dist[np.ix_(src, dst)] = durations * 11.11

            if "durations" in data:
                matriz_tempo[np.ix_(src, dst)] = np.array(data["durations"], dtype=float) / 60.0
            else:
                tem_tempo = False

    print(f"  OSRM: {total_batches} batches concluídos.          ")

    np.fill_diagonal(matriz_dist, 0.0)
    np.fill_diagonal(matriz_tempo, 0.0)

    return matriz_dist, (matriz_tempo if tem_tempo else None)


def _extrair_matrizes(data, avisar_sem_dist=False):
    """Converte resposta OSRM (requisição única) em arrays numpy."""
    if "distances" in data:
        matriz = np.array(data["distances"], dtype=float)
        np.fill_diagonal(matriz, 0.0)
        if "durations" in data:
            matriz_tempos = np.array(data["durations"], dtype=float) / 60.0
            np.fill_diagonal(matriz_tempos, 0.0)
        else:
            matriz_tempos = None
    else:
        if avisar_sem_dist:
            print("AVISO: OSRM não retornou 'distances'. Estimando a partir de 'durations' (40 km/h).")
        durations = np.array(data["durations"], dtype=float)
        matriz = durations * 11.11
        np.fill_diagonal(matriz, 0.0)
        matriz_tempos = durations / 60.0
        np.fill_diagonal(matriz_tempos, 0.0)
    return matriz, matriz_tempos
