"""
Testes unitários para função objetivo do CVRP.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from modelos.instancia import Instancia
from modelos.solucao import Solucao
from modelos.objetivo_config import ObjetivoConfig
from utilitarios.construtivas import nearest_neighbor_capacitado
import pandas as pd


def criar_instancia_toy():
    """
    Cria instância toy para testes: 3 clientes + 1 depósito.
    
    Estrutura:
    - Nó 0: Depósito (demanda=0)
    - Nó 1: Cliente 1 (demanda=10)
    - Nó 2: Cliente 2 (demanda=15)
    - Nó 3: Cliente 3 (demanda=20)
    - Capacidade: 30 (permite no máximo 2 clientes por rota)
    
    Matriz de distâncias:
         0   1   2   3
    0 [  0  10  20  30]
    1 [ 10   0  15  25]
    2 [ 20  15   0  10]
    3 [ 30  25  10   0]
    """
    # Criar DataFrame simulado
    df = pd.DataFrame({
        'lat': [0.0, 1.0, 2.0, 3.0],
        'lon': [0.0, 0.0, 0.0, 0.0],
        'valor_total': [0, 10, 15, 20]
    })
    
    posicoes = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    demandas = [0, 10, 15, 20]
    capacidade = 30
    
    instancia = Instancia(df=df, posicoes=posicoes, demandas=demandas, capacidade_caminhao=capacidade)
    
    # Matriz de distâncias conhecida
    instancia.matriz = np.array([
        [0,  10, 20, 30],
        [10,  0, 15, 25],
        [20, 15,  0, 10],
        [30, 25, 10,  0]
    ], dtype=float)
    
    return instancia


def teste_solucao_viavel():
    """
    Teste 1: Solução viável (dentro da capacidade) → custo_objetivo = custo_distancia
    """
    print("\n=== Teste 1: Solução Viável ===")
    
    instancia = criar_instancia_toy()
    
    # Solução viável: 2 rotas respeitando capacidade de 30
    # Rota 1: 0 -> 1 -> 2 -> 0 (demanda: 10+15=25 ≤ 30)
    # Rota 2: 0 -> 3 -> 0 (demanda: 20 ≤ 30)
    rotas = [
        [0, 1, 2, 0],  # custo: 10 + 15 + 20 = 45
        [0, 3, 0]      # custo: 30 + 30 = 60
    ]
    
    solucao = Solucao(rotas=rotas, instancia=instancia)
    
    # Avaliar
    custo_objetivo = solucao.avaliar(instancia)
    
    # Verificações
    assert solucao.eh_valida(instancia), "Solução deveria ser válida"
    assert solucao.violacoes["capacidade"] == 0, "Não deveria ter violações"
    assert solucao.custo == 105.0, f"Custo esperado: 105, obtido: {solucao.custo}"
    assert solucao.custo_objetivo == solucao.custo, f"Para solução viável, custo_objetivo deve ser igual a custo"
    
    print(f"✓ Solução viável: custo={solucao.custo}, custo_objetivo={solucao.custo_objetivo}")
    print(f"✓ Violações: {solucao.violacoes['capacidade']}")
    print("✓ Teste 1 PASSOU")
    

def teste_solucao_inviavel():
    """
    Teste 2: Solução inviável → custo_objetivo > custo_distancia
    """
    print("\n=== Teste 2: Solução Inviável ===")
    
    instancia = criar_instancia_toy()
    
    # Solução inviável: 1 rota violando capacidade
    # Rota 1: 0 -> 1 -> 2 -> 3 -> 0 (demanda: 10+15+20=45 > 30)
    rotas = [
        [0, 1, 2, 3, 0]  # custo: 10 + 15 + 10 + 30 = 65
    ]
    
    solucao = Solucao(rotas=rotas, instancia=instancia)
    
    # Avaliar
    custo_objetivo = solucao.avaliar(instancia)
    
    # Verificações
    assert not solucao.eh_valida(instancia), "Solução deveria ser inválida"
    assert solucao.violacoes["capacidade"] > 0, "Deveria ter violações"
    assert solucao.custo == 65.0, f"Custo esperado: 65, obtido: {solucao.custo}"
    assert solucao.custo_objetivo > solucao.custo, f"Para solução inviável, custo_objetivo deve ser maior que custo"
    
    penalidade = solucao.custo_objetivo - solucao.custo
    print(f"✓ Solução inviável: custo={solucao.custo}, custo_objetivo={solucao.custo_objetivo}")
    print(f"✓ Violações: {solucao.violacoes['capacidade']} | Penalidade: {penalidade:.2f}")
    print("✓ Teste 2 PASSOU")


def teste_verificacao_detalhada():
    """
    Teste 3: Verificação de violações detalhadas
    """
    print("\n=== Teste 3: Violações Detalhadas ===")
    
    instancia = criar_instancia_toy()
    
    # Solução com múltiplas rotas, algumas violando
    # Rota 0: 0 -> 1 -> 2 -> 0 (demanda: 25, OK)
    # Rota 1: 0 -> 1 -> 3 -> 0 (demanda: 30, OK no limite)
    # Rota 2: 0 -> 2 -> 3 -> 0 (demanda: 35, VIOLA por 5)
    rotas = [
        [0, 1, 2, 0],    # OK
        [0, 1, 3, 0],    # OK
        [0, 2, 3, 0]     # VIOLA
    ]
    
    solucao = Solucao(rotas=rotas, instancia=instancia)
    
    # Verificar violações detalhadas
    violacoes_detalhadas = solucao.verificar_capacidade(instancia)
    
    # Verificações
    assert 2 in violacoes_detalhadas, "Rota 2 deveria estar nas violações"
    assert violacoes_detalhadas[2] == 5.0, f"Excesso esperado: 5, obtido: {violacoes_detalhadas[2]}"
    assert len(violacoes_detalhadas) == 1, "Apenas rota 2 deveria violar"
    
    print(f"✓ Violações detalhadas: {violacoes_detalhadas}")
    print("✓ Teste 3 PASSOU")


def teste_config_penalizacao():
    """
    Teste 4: Testar diferentes configurações de penalização
    """
    print("\n=== Teste 4: Configurações de Penalização ===")
    
    instancia = criar_instancia_toy()
    
    # Solução inviável
    rotas = [[0, 1, 2, 3, 0]]  # excesso de 15 unidades
    solucao = Solucao(rotas=rotas, instancia=instancia)
    
    # Testar penalização proporcional
    config_prop = ObjetivoConfig(
        peso_capacidade=100.0,
        estrategia=ObjetivoConfig.PENALIZACAO_PROPORCIONAL
    )
    custo_obj_prop = solucao.avaliar(instancia, config_prop)
    penalidade_prop = custo_obj_prop - solucao.custo
    
    # Testar penalização fixa
    config_fixa = ObjetivoConfig(
        peso_capacidade=100.0,
        estrategia=ObjetivoConfig.PENALIZACAO_FIXA
    )
    custo_obj_fixa = solucao.avaliar(instancia, config_fixa)
    penalidade_fixa = custo_obj_fixa - solucao.custo
    
    # Verificações
    # Proporcional: 100 * 15 = 1500
    # Fixa: 100 * 1 = 100
    assert abs(penalidade_prop - 1500.0) < 0.01, f"Penalidade proporcional esperada: 1500, obtida: {penalidade_prop}"
    assert abs(penalidade_fixa - 100.0) < 0.01, f"Penalidade fixa esperada: 100, obtida: {penalidade_fixa}"
    assert penalidade_prop > penalidade_fixa, "Penalização proporcional deveria ser maior para esse caso"
    
    print(f"✓ Penalização proporcional: {penalidade_prop:.2f}")
    print(f"✓ Penalização fixa: {penalidade_fixa:.2f}")
    print("✓ Teste 4 PASSOU")


def teste_heuristica_construtiva():
    """
    Teste 5: Testar heurística construtiva (nearest neighbor)
    """
    print("\n=== Teste 5: Heurística Construtiva ===")
    
    instancia = criar_instancia_toy()
    
    # Gerar solução usando nearest neighbor
    solucao = nearest_neighbor_capacitado(instancia)
    solucao.avaliar(instancia)
    
    # Verificações
    assert len(solucao.rotas) > 0, "Deveria ter pelo menos uma rota"
    assert solucao.eh_valida(instancia), "Solução construída deveria ser viável"
    assert solucao.violacoes["capacidade"] == 0, "Não deveria ter violações"
    assert solucao.custo is not None, "Custo deveria estar calculado"
    assert solucao.custo_objetivo == solucao.custo, "Para solução viável, custos devem ser iguais"
    
    print(f"✓ Rotas geradas: {len(solucao.rotas)}")
    print(f"✓ Custo: {solucao.custo}")
    print(f"✓ Solução é viável: {solucao.eh_valida(instancia)}")
    print("✓ Teste 5 PASSOU")


def teste_basico():
    """Executa teste básico do main (compatibilidade)."""
    print("\n=== Teste Básico: Main ===")
    try:
        from main import main
        main()
        print("✓ Teste básico PASSOU")
    except Exception as e:
        print(f"✗ Teste básico FALHOU: {e}")


def executar_todos_testes():
    """Executa todos os testes."""
    print("="*60)
    print("EXECUTANDO TESTES DA FUNÇÃO OBJETIVO")
    print("="*60)
    
    try:
        teste_solucao_viavel()
        teste_solucao_inviavel()
        teste_verificacao_detalhada()
        teste_config_penalizacao()
        teste_heuristica_construtiva()
        
        print("\n" + "="*60)
        print("✓ TODOS OS TESTES PASSARAM!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ TESTE FALHOU: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERRO: {e}")
        raise


if __name__ == "__main__":
    executar_todos_testes()

