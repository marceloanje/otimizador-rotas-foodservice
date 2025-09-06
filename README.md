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
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar versão inicial
python src/main.py
