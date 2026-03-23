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
from algoritmos.busca_tabu import BuscaTabu
from algoritmos.colonia_formigas import ACO
from algoritmos.enxame_particulas import PSO
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


def teste_algoritmo_tabu():
    """Teste 6: BuscaTabu retorna solução com rotas e custo_objetivo."""
    print("\n=== Teste 6: Busca Tabu ===")
    inst = criar_instancia_toy()
    sol = BuscaTabu(inst, max_iter=10, max_no_improve=5).run()
    assert sol is not None, "BuscaTabu deve retornar uma solução"
    assert len(sol.rotas) > 0, "Solução deve ter pelo menos uma rota"
    assert sol.custo_objetivo is not None, "custo_objetivo deve estar calculado"
    print(f"✓ Rotas: {len(sol.rotas)}, custo_objetivo={sol.custo_objetivo:.2f}")
    print("✓ Teste 6 PASSOU")


def teste_algoritmo_aco():
    """Teste 7: ACO retorna solução com rotas e custo_objetivo."""
    print("\n=== Teste 7: ACO ===")
    inst = criar_instancia_toy()
    sol = ACO(inst, n_formigas=3, iter=5).run()
    assert sol is not None, "ACO deve retornar uma solução"
    assert len(sol.rotas) > 0, "Solução deve ter pelo menos uma rota"
    assert sol.custo_objetivo is not None, "custo_objetivo deve estar calculado"
    print(f"✓ Rotas: {len(sol.rotas)}, custo_objetivo={sol.custo_objetivo:.2f}")
    print("✓ Teste 7 PASSOU")


def teste_algoritmo_pso():
    """Teste 8: PSO retorna solução com rotas e custo_objetivo."""
    print("\n=== Teste 8: PSO ===")
    inst = criar_instancia_toy()
    sol = PSO(inst, n_particles=5, max_iter=5).run()
    assert sol is not None, "PSO deve retornar uma solução"
    assert len(sol.rotas) > 0, "Solução deve ter pelo menos uma rota"
    assert sol.custo_objetivo is not None, "custo_objetivo deve estar calculado"
    print(f"✓ Rotas: {len(sol.rotas)}, custo_objetivo={sol.custo_objetivo:.2f}")
    print("✓ Teste 8 PASSOU")


def teste_cobertura():
    """Teste 9: Solução com cliente faltando gera penalidade de cobertura."""
    print("\n=== Teste 9: Cobertura ===")

    instancia = criar_instancia_toy()
    instancia.numero_caminhoes = 2  # ativar verificação de frota

    # Rota que omite cliente 3
    rotas_incompletas = [[0, 1, 2, 0]]
    sol_incompleta = Solucao(rotas=rotas_incompletas, instancia=instancia)
    sol_incompleta.avaliar(instancia)

    # Solução completa para comparação
    rotas_completas = [[0, 1, 2, 0], [0, 3, 0]]
    sol_completa = Solucao(rotas=rotas_completas, instancia=instancia)
    sol_completa.avaliar(instancia)

    assert sol_incompleta.verificar_cobertura(instancia) == 1, "Deveria faltar 1 cliente"
    assert sol_incompleta.custo_objetivo > sol_completa.custo_objetivo, \
        "Solução incompleta deve ter custo_objetivo maior"
    assert not sol_incompleta.eh_valida(instancia), "Solução incompleta deve ser inválida"
    assert sol_incompleta.violacoes["cobertura"] == 1

    print(f"✓ Clientes faltando: {sol_incompleta.verificar_cobertura(instancia)}")
    print(f"✓ custo_objetivo incompleta={sol_incompleta.custo_objetivo:.2f} > completa={sol_completa.custo_objetivo:.2f}")
    print("✓ Teste 9 PASSOU")


def teste_frota():
    """Teste 10: Excesso de veículos gera penalidade e invalida solução."""
    print("\n=== Teste 10: Frota ===")

    instancia = criar_instancia_toy()
    instancia.numero_caminhoes = 2  # limite de 2 caminhões

    # 3 rotas com limite de 2 → excesso de 1
    rotas = [[0, 1, 0], [0, 2, 0], [0, 3, 0]]
    sol = Solucao(rotas=rotas, instancia=instancia)
    sol.avaliar(instancia)
    penalidade_frota = sol.violacoes.get("penalidade_veiculos", 0)
    assert penalidade_frota > 0, "Deve haver penalidade de frota em avaliar()"

    assert not sol.eh_valida(instancia), "Deve ser inválida por excesso de frota"
    assert sol.violacoes["frota_excedida"] == 1, f"Esperado excesso=1, obtido={sol.violacoes['frota_excedida']}"

    print(f"✓ Frota excedida: {sol.violacoes['frota_excedida']}")
    print(f"✓ Penalidade veículos: {penalidade_frota:.2f}")
    print("✓ Teste 10 PASSOU")


def teste_carga_minima():
    """Teste 11: Rota com carga abaixo do mínimo gera penalidade."""
    print("\n=== Teste 11: Carga Mínima ===")

    instancia = criar_instancia_toy()
    instancia.carga_minima = 20.0  # mínimo de 20 por rota

    # Rota 2: só cliente 1 com demanda 10 < 20 → déficit de 10
    rotas = [[0, 1, 2, 0], [0, 3, 0]]  # rota 0: carga=25 OK; rota 1: carga=20 == mínimo, OK
    sol_ok = Solucao(rotas=rotas, instancia=instancia)
    deficits_ok = sol_ok.verificar_carga_minima(instancia, 20.0)
    assert deficits_ok == {}, f"Não deveria ter déficit, obtido: {deficits_ok}"

    # Rota com carga 10 < 20
    rotas_deficit = [[0, 1, 2, 0], [0, 1, 0]]  # segunda rota: carga=10
    sol_deficit = Solucao(rotas=rotas_deficit, instancia=instancia)
    deficits = sol_deficit.verificar_carga_minima(instancia, 20.0)

    assert 1 in deficits, "Rota 1 deveria ter déficit"
    assert abs(deficits[1] - 10.0) < 0.01, f"Déficit esperado 10.0, obtido {deficits[1]}"

    sol_deficit.avaliar(instancia)
    assert sol_deficit.violacoes["penalidade_carga_minima"] > 0, "Deve haver penalidade de carga mínima"

    print(f"✓ Déficits: {deficits}")
    print(f"✓ Penalidade carga mínima: {sol_deficit.violacoes['penalidade_carga_minima']:.2f}")
    print("✓ Teste 11 PASSOU")


def teste_factibilidade_infactivel():
    """Teste 12: Instância infactível levanta ValueError."""
    print("\n=== Teste 12: Factibilidade Infactível ===")

    df = pd.DataFrame({'lat': [0.0, 1.0], 'lon': [0.0, 0.0], 'valor_total': [0, 9999]})
    posicoes = [(0.0, 0.0), (1.0, 0.0)]
    demandas = [0.0, 9999.0]

    # 1 caminhão com capacidade 100 → 9999 > 100 → infactível
    instancia = Instancia(df=df, posicoes=posicoes, demandas=demandas,
                          capacidade_caminhao=100, numero_caminhoes=1)

    try:
        instancia.verificar_factibilidade()
        assert False, "Deveria ter levantado ValueError"
    except ValueError as e:
        assert "infactível" in str(e).lower() or "Infact" in str(e), f"Mensagem inesperada: {e}"
        print(f"✓ ValueError levantado corretamente: {e}")

    print("✓ Teste 12 PASSOU")


def teste_factibilidade_factivel():
    """Teste 13: Instância factível não levanta exceção."""
    print("\n=== Teste 13: Factibilidade Factível ===")

    instancia = criar_instancia_toy()
    instancia.numero_caminhoes = 2
    # demanda total = 10+15+20 = 45, capacidade total = 30*2 = 60 >= 45

    try:
        instancia.verificar_factibilidade()
        print("✓ Nenhuma exceção levantada")
    except ValueError as e:
        assert False, f"Não deveria ter levantado exceção: {e}"

    print("✓ Teste 13 PASSOU")


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
        teste_algoritmo_tabu()
        teste_algoritmo_aco()
        teste_algoritmo_pso()
        teste_cobertura()
        teste_frota()
        teste_carga_minima()
        teste_factibilidade_infactivel()
        teste_factibilidade_factivel()

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

