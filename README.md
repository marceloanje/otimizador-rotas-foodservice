# Otimizador de Rotas para Food-Service

Projeto de TCC em Engenharia de Sistemas (UFMG)  
Autor: Marcelo Andrade de Jesus Filho
Orientadora: Gabriela Nunes Lopes

## Objetivo
Desenvolver um sistema de otimização de rotas de entrega para food-service, baseado no Problema de Roteamento de Veículos (VRP) com janelas de tempo.  
Serão comparados métodos exatos e meta-heurísticos (Colônia de Formigas, Enxame de Partículas e Busca Tabu).

## Estrutura do repositório
- `src/` → código-fonte
- `testes/` → testes unitários
- `docs/` → documentação e diagramas
- `requirements.txt` → bibliotecas necessárias

## Como rodar

### 1. Ambiente Python
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar versão inicial
python src/main.py
```

### 2. Servidor OSRM local (Docker) — matriz de distâncias reais

O projeto usa o OSRM para calcular a matriz de distâncias reais por malha viária.
Requer Docker instalado. Execute os comandos abaixo **uma única vez** para preparar os dados de Minas Gerais:

```bash
# Baixar dados OSM de Minas Gerais (~150 MB) em:
# https://download.geofabrik.de/south-america/brazil/minas-gerais.html
# Salvar o arquivo como: minas-gerais-latest.osm.pbf na raiz do projeto

# Pré-processar (executar na raiz do projeto)
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-extract -p /opt/car.lua /data/minas-gerais-latest.osm.pbf
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-partition /data/minas-gerais-latest.osrm
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-customize /data/minas-gerais-latest.osrm
```

Para subir o servidor (necessário antes de rodar o comparador):
```bash
docker run -t -p 5000:5000 -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/minas-gerais-latest.osrm
```

O servidor fica disponível em `http://localhost:5000`. A URL é configurável em `src/config.py` via `OSRM_URL`.

> **Fallback:** se o servidor OSRM estiver indisponível, o sistema usa automaticamente distância euclidiana como substituto (com aviso no console).
