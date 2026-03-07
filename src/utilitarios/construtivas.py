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
    
    Parameters
    ----------
    instancia : Instancia
        Instância do problema com matriz, demandas e capacidade.
    inicio_deposito : bool
        Se True, sempre inicia rotas no depósito (nó 0). Padrão: True.
        
    Returns
    -------
    Solucao
        Solução com múltiplas rotas respeitando capacidade.
        
    Notes
    -----
    - Cada rota começa e termina no depósito (nó 0)
    - Garante factibilidade de capacidade
    - Não garante solução ótima (é heurística gulosa)
    """
    matriz = instancia.matriz
    demandas = instancia.demandas
    capacidade = instancia.capacidade_caminhao
    n = len(matriz)
    
    # Depósito é o nó 0
    deposito = 0
    
    # Clientes a visitar (todos exceto depósito)
    nao_visitados = set(range(1, n))
    
    rotas = []
    
    while nao_visitados:
        # Começar nova rota
        rota = [deposito]
        carga_atual = 0.0
        posicao_atual = deposito
        
        while True:
            # Encontrar cliente mais próximo que cabe na capacidade
            melhor_cliente = None
            melhor_distancia = float('inf')
            
            for cliente in nao_visitados:
                demanda_cliente = demandas[cliente]
                
                # Verificar se cabe na capacidade
                if carga_atual + demanda_cliente <= capacidade:
                    distancia = matriz[posicao_atual][cliente]
                    
                    if distancia < melhor_distancia:
                        melhor_distancia = distancia
                        melhor_cliente = cliente
            
            # Se nenhum cliente cabe, fechar a rota
            if melhor_cliente is None:
                break
            
            # Adicionar cliente à rota
            rota.append(melhor_cliente)
            carga_atual += demandas[melhor_cliente]
            posicao_atual = melhor_cliente
            nao_visitados.remove(melhor_cliente)
        
        # Retornar ao depósito
        rota.append(deposito)
        rotas.append(rota)
    
    # Criar solução
    solucao = Solucao(rotas=rotas, instancia=instancia)
    solucao.calcular_custo(instancia)
    
    return solucao


def savings_parallel(instancia):
    """
    Constrói solução CVRP usando o algoritmo de Clarke-Wright (Savings).
    
    Algoritmo clássico que:
    1. Começa com rotas individuais para cada cliente (depósito-cliente-depósito)
    2. Calcula savings para combinar pares de rotas
    3. Combina rotas em ordem decrescente de savings, respeitando capacidade
    
    Parameters
    ----------
    instancia : Instancia
        Instância do problema com matriz, demandas e capacidade.
        
    Returns
    -------
    Solucao
        Solução com múltiplas rotas respeitando capacidade.
        
    Notes
    -----
    - Frequentemente produz soluções melhores que nearest neighbor
    - Savings(i,j) = dist(0,i) + dist(0,j) - dist(i,j)
    """
    matriz = instancia.matriz
    demandas = instancia.demandas
    capacidade = instancia.capacidade_caminhao
    n = len(matriz)
    
    deposito = 0
    
    # 1. Criar rotas individuais para cada cliente
    rotas = []
    cargas = []
    
    for cliente in range(1, n):
        rotas.append([deposito, cliente, deposito])
        cargas.append(demandas[cliente])
    
    # 2. Calcular savings para todos os pares de clientes
    savings = []
    
    for i in range(1, n):
        for j in range(i + 1, n):
            saving = matriz[deposito][i] + matriz[deposito][j] - matriz[i][j]
            savings.append((saving, i, j))
    
    # Ordenar savings em ordem decrescente
    savings.sort(reverse=True, key=lambda x: x[0])
    
    # 3. Combinar rotas baseado em savings
    for saving_value, i, j in savings:
        # Encontrar rotas que contêm i e j
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
        
        # Se estão na mesma rota, não combinar
        if idx_i == idx_j:
            continue
        
        # Verificar se i é extremo da rota_i e j é extremo da rota_j
        # (extremos = primeiro ou último cliente, não o depósito)
        if not (rota_i[1] == i or rota_i[-2] == i):
            continue
        if not (rota_j[1] == j or rota_j[-2] == j):
            continue
        
        # Verificar capacidade
        carga_combinada = cargas[idx_i] + cargas[idx_j]
        if carga_combinada > capacidade:
            continue
        
        # Combinar rotas
        # Orientar rotas corretamente
        if rota_i[-2] == i and rota_j[1] == j:
            # i é final de rota_i, j é início de rota_j
            nova_rota = rota_i[:-1] + rota_j[1:]
        elif rota_i[1] == i and rota_j[-2] == j:
            # i é início de rota_i, j é final de rota_j
            nova_rota = rota_j[:-1] + rota_i[1:]
        elif rota_i[1] == i and rota_j[1] == j:
            # ambos são início
            nova_rota = [deposito] + rota_i[1:-1][::-1] + rota_j[1:]
        elif rota_i[-2] == i and rota_j[-2] == j:
            # ambos são final
            nova_rota = rota_i[:-1] + rota_j[1:-1][::-1] + [deposito]
        else:
            continue
        
        # Remover rotas antigas e adicionar nova
        # Remover índice maior primeiro para não afetar o menor
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
    
    # Criar solução
    solucao = Solucao(rotas=rotas, instancia=instancia)
    solucao.calcular_custo(instancia)
    
    return solucao


def solucao_inicial_aleatoria(instancia, seed=None):
    """
    Cria solução inicial aleatória respeitando capacidade.
    
    Útil para testes e para adicionar diversidade em populações.
    
    Parameters
    ----------
    instancia : Instancia
        Instância do problema.
    seed : int | None
        Semente para geração aleatória. Se None, usa aleatório.
        
    Returns
    -------
    Solucao
        Solução aleatória viável.
    """
    if seed is not None:
        np.random.seed(seed)
    
    matriz = instancia.matriz
    demandas = instancia.demandas
    capacidade = instancia.capacidade_caminhao
    n = len(matriz)
    
    deposito = 0
    
    # Embaralhar ordem de visita aos clientes
    clientes = list(range(1, n))
    np.random.shuffle(clientes)
    
    # Construir rotas respeitando capacidade
    rotas = []
    rota_atual = [deposito]
    carga_atual = 0.0
    
    for cliente in clientes:
        demanda_cliente = demandas[cliente]
        
        # Se não cabe, fechar rota e começar nova
        if carga_atual + demanda_cliente > capacidade:
            rota_atual.append(deposito)
            rotas.append(rota_atual)
            rota_atual = [deposito]
            carga_atual = 0.0
        
        # Adicionar cliente à rota
        rota_atual.append(cliente)
        carga_atual += demanda_cliente
    
    # Fechar última rota
    if len(rota_atual) > 1:
        rota_atual.append(deposito)
        rotas.append(rota_atual)
    
    # Criar solução
    solucao = Solucao(rotas=rotas, instancia=instancia)
    solucao.calcular_custo(instancia)
    
    return solucao
