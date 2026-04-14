"""Configuração do experimento comparativo de meta-heurísticas.

Define as instâncias, nº de runs e seed base. Editar este arquivo para
mudar o escopo do experimento sem mexer em comparador.py.
"""

import os

_DADOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")

INSTANCIAS = [
    {
        "nome": "pequena",
        "path": os.path.join(_DADOS_DIR, "pedidos_pequeno.csv"),
        "usar_exato": True,
        "tempo_limite_exato": 120,
    },
    {
        "nome": "grande",
        "path": os.path.join(_DADOS_DIR, "pedidos_grande.csv"),
        "usar_exato": False,
        "tempo_limite_exato": 0,
    },
]

N_RUNS = 30
SEED_BASE = 42
