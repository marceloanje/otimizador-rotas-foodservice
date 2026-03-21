"""
Heurísticas construtivas para geração de soluções iniciais do CVRP.

Implementa algoritmos gulosos para construir soluções viáveis que respeitam
restrições de capacidade dos veículos.
"""

import numpy as np
from modelos.solucao import Solucao


def nearest_neighbor_capacitado(instancia, inicio_deposito=True):
    """
    Constrói solução CVRP usando nearest neighbor com restrição de capacidade.

    Algoritmo guloso que constrói rotas sequencialmente:
    1. Começa uma nova rota no depósito
    2. Adiciona o cliente mais próximo que cabe na capacidade restante
    3. Quando nenhum cliente mais cabe, fecha a rota e começa outra
    4. Repete até visitar todos os clientes

    Garante factibilidade de capacidade, mas não garante solução ótima.
    """
    matriz = instancia.matriz
    demandas = instancia.demandas
    capacidade = instancia.capacidade_caminhao
    n = len(matriz)

    deposito = 0
    nao_visitados = set(range(1, n))
    rotas = []

    while nao_visitados:
        rota = [deposito]
        carga_atual = 0.0
        posicao_atual = deposito

        while True:
            melhor_cliente = None
            melhor_distancia = float('inf')

            for cliente in nao_visitados:
                demanda_cliente = demandas[cliente]
                if carga_atual + demanda_cliente <= capacidade:
                    distancia = matriz[posicao_atual][cliente]
                    if distancia < melhor_distancia:
                        melhor_distancia = distancia
                        melhor_cliente = cliente

            if melhor_cliente is None:
                break

            rota.append(melhor_cliente)
            carga_atual += demandas[melhor_cliente]
            posicao_atual = melhor_cliente
            nao_visitados.remove(melhor_cliente)

        rota.append(deposito)
        rotas.append(rota)

    solucao = Solucao(rotas=rotas, instancia=instancia)
    solucao.calcular_custo(instancia)
    return solucao


def savings_parallel(instancia):
    """
    Constrói solução CVRP usando o algoritmo de Clarke-Wright (Savings).

    Começa com rotas individuais (depósito→cliente→depósito) e vai
    combinando pares em ordem decrescente de savings até não ser mais
    possível sem violar capacidade.

    Savings(i,j) = dist(0,i) + dist(0,j) - dist(i,j)
    """
    matriz = instancia.matriz
    demandas = instancia.demandas
    capacidade = instancia.capacidade_caminhao
    n = len(matriz)

    deposito = 0

    # 1. Rota individual para cada cliente
    rotas = []
    cargas = []
    for cliente in range(1, n):
        rotas.append([deposito, cliente, deposito])
        cargas.append(demandas[cliente])

    # 2. Calcular e ordenar savings
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            saving = matriz[deposito][i] + matriz[deposito][j] - matriz[i][j]
            savings.append((saving, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    # 3. Combinar rotas baseado em savings
    for saving_value, i, j in savings:
        rota_i = None
        rota_j = None
        idx_i = None
        idx_j = None

        for idx, rota in enumerate(rotas):
            if i in rota:
                rota_i = rota
                idx_i = idx
            if j in rota:
                rota_j = rota
                idx_j = idx

        if idx_i == idx_j:
            continue

        # i e j precisam ser extremos (não internos) das suas rotas
        if not (rota_i[1] == i or rota_i[-2] == i):
            continue
        if not (rota_j[1] == j or rota_j[-2] == j):
            continue

        carga_combinada = cargas[idx_i] + cargas[idx_j]
        if carga_combinada > capacidade:
            continue

        # Orientar rotas para que a junção fique contínua
        if rota_i[-2] == i and rota_j[1] == j:
            nova_rota = rota_i[:-1] + rota_j[1:]
        elif rota_i[1] == i and rota_j[-2] == j:
            nova_rota = rota_j[:-1] + rota_i[1:]
        elif rota_i[1] == i and rota_j[1] == j:
            nova_rota = [deposito] + rota_i[1:-1][::-1] + rota_j[1:]
        elif rota_i[-2] == i and rota_j[-2] == j:
            nova_rota = rota_i[:-1] + rota_j[1:-1][::-1] + [deposito]
        else:
            continue

        # Remover índice maior primeiro para não deslocar o menor
        if idx_i > idx_j:
            del rotas[idx_i]
            del cargas[idx_i]
            del rotas[idx_j]
            del cargas[idx_j]
        else:
            del rotas[idx_j]
            del cargas[idx_j]
            del rotas[idx_i]
            del cargas[idx_i]

        rotas.append(nova_rota)
        cargas.append(carga_combinada)

    solucao = Solucao(rotas=rotas, instancia=instancia)
    solucao.calcular_custo(instancia)
    return solucao


def solucao_inicial_aleatoria(instancia, seed=None):
    """
    Cria solução inicial aleatória respeitando capacidade.

    Útil para testes e para adicionar diversidade em populações.
    """
    if seed is not None:
        np.random.seed(seed)

    matriz = instancia.matriz
    demandas = instancia.demandas
    capacidade = instancia.capacidade_caminhao
    n = len(matriz)

    deposito = 0
    clientes = list(range(1, n))
    np.random.shuffle(clientes)

    rotas = []
    rota_atual = [deposito]
    carga_atual = 0.0

    for cliente in clientes:
        demanda_cliente = demandas[cliente]
        if carga_atual + demanda_cliente > capacidade:
            rota_atual.append(deposito)
            rotas.append(rota_atual)
            rota_atual = [deposito]
            carga_atual = 0.0

        rota_atual.append(cliente)
        carga_atual += demanda_cliente

    if len(rota_atual) > 1:
        rota_atual.append(deposito)
        rotas.append(rota_atual)

    solucao = Solucao(rotas=rotas, instancia=instancia)
    solucao.calcular_custo(instancia)
    return solucao
