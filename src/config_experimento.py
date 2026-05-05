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
        # 14 clientes, demanda total ≈ 12.222
        # lower bound: ceil(12222 / 5000) = 3 caminhões
        "capacidade_caminhao": 5000,
        "numero_caminhoes": 3,
    },
    {
        "nome": "grande",
        "path": os.path.join(_DADOS_DIR, "pedidos_grande.csv"),
        "usar_exato": False,
        "tempo_limite_exato": 0,
        # 143 clientes, demanda total ≈ 151.316, maior pedido individual ≈ 6.306
        # capacidade ≥ 6.306 → 7.000; lower bound: ceil(151316 / 7000) = 22; +18% folga → 26
        "capacidade_caminhao": 7000,
        "numero_caminhoes": 26,
    },
]

N_RUNS = 30
SEED_BASE = 42
