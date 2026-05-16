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
        # 143 clientes, demanda total ≈ 151.316; Q=7000 (cobre max individual 6.306)
        # lower bound: ceil(151316 / 7000) = 22; +20% folga → 27
        "capacidade_caminhao": 7000,
        "numero_caminhoes": 27,
    },
    {
        "nome": "grande_2",
        "path": os.path.join(_DADOS_DIR, "pedidos_grande_2.csv"),
        "usar_exato": False,
        "tempo_limite_exato": 0,
        # 108 clientes, demanda total ≈ 78.527; Q=7000 (cobre max individual 2.883)
        # lower bound: ceil(78527 / 7000) = 12; +20% folga → 15
        "capacidade_caminhao": 7000,
        "numero_caminhoes": 15,
    },
    {
        "nome": "grande_3",
        "path": os.path.join(_DADOS_DIR, "pedidos_grande_3.csv"),
        "usar_exato": False,
        "tempo_limite_exato": 0,
        # 106 clientes, demanda total ≈ 87.337; Q=7000 (cobre max individual 5.123)
        # lower bound: ceil(87337 / 7000) = 13; +20% folga → 16
        "capacidade_caminhao": 7000,
        "numero_caminhoes": 16,
    },
    {
        "nome": "grande_4",
        "path": os.path.join(_DADOS_DIR, "pedidos_grande_4.csv"),
        "usar_exato": False,
        "tempo_limite_exato": 0,
        # 84 clientes, demanda total ≈ 71.456; Q=7000 (cobre max individual 5.694)
        # lower bound: ceil(71456 / 7000) = 11; +20% folga → 14
        "capacidade_caminhao": 7000,
        "numero_caminhoes": 14,
    },
    {
        "nome": "grande_5",
        "path": os.path.join(_DADOS_DIR, "pedidos_grande_5.csv"),
        "usar_exato": False,
        "tempo_limite_exato": 0,
        # 58 clientes, demanda total ≈ 51.165; Q=7000 (cobre max individual 4.450)
        # lower bound: ceil(51165 / 7000) = 8; +20% folga → 10
        "capacidade_caminhao": 7000,
        "numero_caminhoes": 10,
    },
]

N_RUNS = 30
SEED_BASE = 42
