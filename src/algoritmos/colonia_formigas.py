import numpy as np
from modelos.representacao import Representacao
from modelos.solucao import Solucao
from modelos.objetivo_config import ObjetivoConfig
from utilitarios.construtivas import nearest_neighbor_capacitado
import random

class ACO:
    def __init__(self, instancia, n_formigas=10, iter=20, alpha=1.0, beta=2.0, evaporacao=0.1, config=None):
        self.inst = instancia
        self.matriz = instancia.matriz
        self.n = len(self.matriz)
        self.demandas = instancia.demandas
        self.capacidade = instancia.capacidade_caminhao

        self.n_formigas = n_formigas
        self.iter = iter
        self.alpha = alpha
        self.beta = beta
        self.evaporacao = evaporacao

        # Configuração de objetivo
        self.config = config if config is not None else ObjetivoConfig(matriz=self.matriz)

        # Feromônio inicial
        self.feromonio = np.ones((self.n, self.n))

    def construir_solucao(self):
        """
        Constrói uma solução CVRP completa (múltiplas rotas) respeitando capacidade.
        Usa feromônios para guiar a construção probabilística.
        """
        deposito = 0
        nao_visitados = set(range(1, self.n))
        rotas = []
        
        while nao_visitados:
            # Construir uma rota respeitando capacidade
            rota = [deposito]
            carga_atual = 0.0
            posicao_atual = deposito
            
            while nao_visitados:
                # Candidatos viáveis (cabem na capacidade)
                candidatos = []
                for cliente in nao_visitados:
                    if carga_atual + self.demandas[cliente] <= self.capacidade:
                        candidatos.append(cliente)
                
                if not candidatos:
                    # Nenhum cliente cabe, fechar rota
                    break
                
                # Calcular probabilidades usando feromônio e heurística
                probabilidades = []
                for j in candidatos:
                    distancia = self.matriz[posicao_atual][j]
                    if distancia == 0:
                        distancia = 0.0001
                    
                    tau = self.feromonio[posicao_atual][j] ** self.alpha
                    eta = (1.0 / distancia) ** self.beta
                    
                    probabilidades.append(tau * eta)
                
                # Normalizar probabilidades
                probabilidades = np.array(probabilidades)
                soma = probabilidades.sum()
                if soma > 0:
                    probabilidades = probabilidades / soma
                else:
                    # Uniforme se todas as probabilidades são zero
                    probabilidades = np.ones(len(candidatos)) / len(candidatos)
                
                # Selecionar próximo cliente
                proximo = random.choices(candidatos, weights=probabilidades)[0]
                
                # Adicionar à rota
                rota.append(proximo)
                carga_atual += self.demandas[proximo]
                posicao_atual = proximo
                nao_visitados.remove(proximo)
            
            # Fechar rota retornando ao depósito
            rota.append(deposito)
            rotas.append(rota)
        
        # Criar solução
        solucao = Solucao(rotas=rotas, instancia=self.inst)
        solucao.avaliar(self.inst, self.config)
        
        return solucao

    def atualizar_feromonio(self, solucoes):
        """
        Atualiza matriz de feromônios baseado nas soluções.
        
        Parameters
        ----------
        solucoes : list[Solucao]
            Lista de soluções para atualizar feromônio.
        """
        # Evaporação
        self.feromonio *= (1 - self.evaporacao)
        
        # Depositar feromônio de cada solução
        for solucao in solucoes:
            # Usar custo objetivo para determinar quantidade de feromônio
            delta = 1.0 / (solucao.custo_objetivo if solucao.custo_objetivo else solucao.custo)
            
            # Depositar em todas as arestas usadas
            for rota in solucao.rotas:
                for i in range(len(rota) - 1):
                    a = rota[i]
                    b = rota[i + 1]
                    self.feromonio[a][b] += delta
                    self.feromonio[b][a] += delta

    def run(self):
        """Executa o algoritmo ACO para CVRP."""
        melhor_solucao = None
        melhor_custo_objetivo = float("inf")

        for _ in range(self.iter):
            # Construir soluções para todas as formigas
            solucoes = []
            
            for f in range(self.n_formigas):
                solucao = self.construir_solucao()
                solucoes.append(solucao)
                
                # Atualizar melhor solução
                if solucao.custo_objetivo < melhor_custo_objetivo:
                    melhor_solucao = solucao
                    melhor_custo_objetivo = solucao.custo_objetivo
            
            # Atualizar feromônio com as soluções desta iteração
            self.atualizar_feromonio(solucoes)

        return melhor_solucao

