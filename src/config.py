"""Configurações de infraestrutura (OSRM e depósito).

Parâmetros de frota (capacidade e número de caminhões) são definidos por instância
em `config_experimento.py`, pois variam entre a instância pequena e a grande.
"""

# URL base do servidor OSRM para cálculo da matriz de distâncias
# Local (Docker): "http://localhost:5000"
# Servidor público (fallback): "http://router.project-osrm.org"
OSRM_URL = "http://localhost:5000"

# Depósito central (nó 0) — substituir pelo endereço real do centro de distribuição
DEPOSITO_ENDERECO = "Praça Sete de Setembro, Centro, Belo Horizonte, MG, Brasil"
DEPOSITO_LAT = -19.9191
DEPOSITO_LON = -43.9386
