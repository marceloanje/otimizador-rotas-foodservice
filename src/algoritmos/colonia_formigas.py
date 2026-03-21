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

        self.config = config if config is not None else ObjetivoConfig(matriz=self.matriz)
        self.feromonio = np.ones((self.n, self.n))

    def construir_solucao(self):
        """Constrói uma solução CVRP completa usando feromônios para guiar a construção."""
        deposito = 0
        nao_visitados = set(range(1, self.n))
        rotas = []

        while nao_visitados:
            rota = [deposito]
            carga_atual = 0.0
            posicao_atual = deposito

            while nao_visitados:
                candidatos = [c for c in nao_visitados
                              if carga_atual + self.demandas[c] <= self.capacidade]

                if not candidatos:
                    break

                probabilidades = []
                for j in candidatos:
                    distancia = self.matriz[posicao_atual][j] or 0.0001
                    tau = self.feromonio[posicao_atual][j] ** self.alpha
                    eta = (1.0 / distancia) ** self.beta
                    probabilidades.append(tau * eta)

                probabilidades = np.array(probabilidades)
                soma = probabilidades.sum()
                if soma > 0:
                    probabilidades = probabilidades / soma
                else:
                    # todas zero, distribuição uniforme
                    probabilidades = np.ones(len(candidatos)) / len(candidatos)

                proximo = random.choices(candidatos, weights=probabilidades)[0]

                rota.append(proximo)
                carga_atual += self.demandas[proximo]
                posicao_atual = proximo
                nao_visitados.remove(proximo)

            rota.append(deposito)
            rotas.append(rota)

        solucao = Solucao(rotas=rotas, instancia=self.inst)
        solucao.avaliar(self.inst, self.config)
        return solucao

    def atualizar_feromonio(self, solucoes):
        """Evaporação + depósito de feromônio proporcional à qualidade de cada solução."""
        self.feromonio *= (1 - self.evaporacao)

        for solucao in solucoes:
            # custo_objetivo penaliza soluções inviáveis, então usamos ele como base
            delta = 1.0 / (solucao.custo_objetivo if solucao.custo_objetivo else solucao.custo)

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
            solucoes = []

            for f in range(self.n_formigas):
                solucao = self.construir_solucao()
                solucoes.append(solucao)

                if solucao.custo_objetivo < melhor_custo_objetivo:
                    melhor_solucao = solucao
                    melhor_custo_objetivo = solucao.custo_objetivo

            self.atualizar_feromonio(solucoes)

        return melhor_solucao
