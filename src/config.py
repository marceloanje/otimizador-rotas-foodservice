"""Configurações do problema CVRP (constantes da frota e depósito).

Unidades:
- `CAPACIDADE_CAMINHAO`: unidades da demanda (mesma unidade de `valor_total` no CSV).
- `NUMERO_CAMINHOES`: número de veículos disponíveis na frota.
- `CARGA_MINIMA_CAMINHAO`: carga mínima por rota (10% da capacidade); rotas abaixo são penalizadas.

As constantes ficam separadas neste arquivo para facilitar ajustes futuros.
"""

# Capacidade por caminhão (unidades)
CAPACIDADE_CAMINHAO = 5000

# Número de caminhões disponíveis na frota
NUMERO_CAMINHOES = 3

# Carga mínima por caminhão (evitar rotas quase vazias) — 10% da capacidade
CARGA_MINIMA_CAMINHAO = int(CAPACIDADE_CAMINHAO * 0.10)

# URL base do servidor OSRM para cálculo da matriz de distâncias
# Local (Docker): "http://localhost:5000"
# Servidor público (fallback): "http://router.project-osrm.org"
OSRM_URL = "http://localhost:5000"

# Depósito central (nó 0) — substituir pelo endereço real do centro de distribuição
DEPOSITO_ENDERECO = "Praça Sete de Setembro, Centro, Belo Horizonte, MG, Brasil"
DEPOSITO_LAT = -19.9191
DEPOSITO_LON = -43.9386
