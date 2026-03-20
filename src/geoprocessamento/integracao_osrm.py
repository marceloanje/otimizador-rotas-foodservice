import numpy as np
import requests


def obter_matriz_osrm(posicoes, timeout=30):
    """Calcula a matriz de distâncias reais via OSRM.

    Usa a URL configurada em `config.OSRM_URL`. Para rodar localmente,
    suba o servidor OSRM via Docker apontando para os dados de MG e
    mantenha `OSRM_URL = "http://localhost:5000"` em config.py.

    Parâmetros
    ----------
    posicoes : list[tuple[float, float]]
        Lista de tuplas (lat, lon). Índice 0 é o depósito.
    timeout : int
        Timeout em segundos para a requisição HTTP.

    Retorna
    -------
    numpy.ndarray
        Matriz quadrada (n x n) com distâncias em metros.

    Raises
    ------
    RuntimeError
        Se o OSRM retornar código diferente de "Ok".
    requests.exceptions.RequestException
        Se a requisição HTTP falhar.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import OSRM_URL

    # OSRM exige coordenadas no formato lon,lat (não lat,lon)
    coords_str = ";".join(f"{lon},{lat}" for lat, lon in posicoes)
    url = f"{OSRM_URL}/table/v1/driving/{coords_str}"
    params = {"annotations": "distance,duration"}

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM retornou código inesperado: {data.get('code')}")

    if "distances" in data:
        matriz = np.array(data["distances"], dtype=float)
    else:
        # Servidor público pode não retornar distances; usar durations * velocidade média
        # 40 km/h ≈ 11.11 m/s
        print("AVISO: OSRM não retornou 'distances'. Estimando a partir de 'durations' (40 km/h).")
        matriz = np.array(data["durations"], dtype=float) * 11.11

    np.fill_diagonal(matriz, 0.0)
    return matriz
