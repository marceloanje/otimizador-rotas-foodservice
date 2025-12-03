import numpy as np
from modelos.representacao import Representacao
from modelos.solucao import Solucao
import random

class ACO:
    def __init__(self, instancia, n_formigas=10, iter=20, alpha=1.0, beta=2.0, evaporacao=0.1):
        self.inst = instancia
        self.matriz = instancia.matriz
        self.n = len(self.matriz)

        self.n_formigas = n_formigas
        self.iter = iter
        self.alpha = alpha
        self.beta = beta
        self.evaporacao = evaporacao

        # Feromônio inicial
        self.feromonio = np.ones((self.n, self.n))

    def construir_rota(self):
        n = self.n
        nao_visitados = list(range(n))
        atual = random.choice(nao_visitados)
        rota = [atual]
        nao_visitados.remove(atual)

        while nao_visitados:
            probabilidades = []

            for j in nao_visitados:
                distancia = self.matriz[atual][j]
                if distancia == 0:
                    distancia = 0.0001

                tau = self.feromonio[atual][j] ** self.alpha
                eta = (1.0 / distancia) ** self.beta

                probabilidades.append(tau + eta)

            probabilidades = np.array(probabilidades)
            probabilidades = probabilidades / probabilidades.sum()

            proximo = random.choices(nao_visitados, weights=probabilidades)[0]

            rota.append(proximo)
            nao_visitados.remove(proximo)
            atual = proximo

        return rota

    # Atualização do feromônio
    def atualizar_feromonio(self, rota, custo):
        self.feromonio *= (1 - self.evaporacao)

        for i in range(len(rota) - 1):
            a = rota[i]
            b = rota[i + 1]
            self.feromonio[a][b] += 1.0 / custo
            self.feromonio[b][a] += 1.0 / custo

    # Loop principal do ACO
    def run(self):
        melhor_rota = None
        melhor_custo = float("inf")

        for _ in range(self.iter):
            for f in range(self.n_formigas):
                rota = self.construir_rota()

                custo = Representacao(rota).custo(self.matriz)

                if custo < melhor_custo:
                    melhor_rota = rota
                    melhor_custo = custo

            self.atualizar_feromonio(melhor_rota, melhor_custo)

        return Solucao(melhor_rota, melhor_custo)
