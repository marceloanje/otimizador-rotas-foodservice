import random
import time
import numpy as np
from modelos.representacao import Representacao
from modelos.solucao import Solucao

class BuscaTabu:
    def __init__(self, instancia, max_iter=500, tabu_tenure=15, max_no_improve=100):
        self.inst = instancia
        self.matriz = instancia.matriz
        self.n = len(self.matriz)

        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        self.max_no_improve = max_no_improve

    def _rota_inicial(self):
        rota = list(range(self.n))
        random.shuffle(rota)
        return rota

    def _custo_rota(self, rota):
        return Representacao(rota).custo(self.matriz)

    def _two_opt_neighbors(self, rota):
        # Gera vizinhança 2-opt (todos os pares i<j)
        n = len(rota)
        for i in range(0, n-1):
            for j in range(i+1, n):
                new = rota[:i] + rota[i:j+1][::-1] + rota[j+1:]
                yield (i, j, new)

    def run(self):
        start = time.time()

        current = self._rota_inicial()
        best = list(current)
        best_cost = self._custo_rota(best)

        # Babu list: dicionário de (i,j) -> tenure
        tabu = dict()
        iter_no_improve = 0

        for it in range(self.max_iter):
            neighborhood_best = None
            neighborhood_best_cost = float("inf")
            neighborhood_best_move = None

            # Explorar vizinhança (2-opt)
            for i, j, candidate in self._two_opt_neighbors(current):
                move = (i, j)
                cand_cost = self._custo_rota(candidate)

                is_tabu = move in tabu and tabu[move] > 0

                if is_tabu and cand_cost >= best_cost:
                    continue

                if cand_cost < neighborhood_best_cost:
                    neighborhood_best = candidate
                    neighborhood_best_cost = cand_cost
                    neighborhood_best_move = move

            if neighborhood_best is None:
                break

            # Aplicar melhor vizinho
            current = neighborhood_best

            # Atualizar tabu: decrementar e adicionar movimento atual
            keys = list(tabu.keys())
            for k in keys:
                tabu[k] -= 1
                if tabu[k] <= 0:
                    del tabu[k]

            tabu[neighborhood_best_move] = self.tabu_tenure

            # Atualizar melhor global
            if neighborhood_best_cost < best_cost:
                best = list(neighborhood_best)
                best_cost = neighborhood_best_cost
                iter_no_improve = 0
            else:
                iter_no_improve += 1

            if iter_no_improve >= self.max_no_improve:
                break

        elapsed = time.time() - start
        sol = Solucao(best, best_cost)
        sol.meta = {"tempo": elapsed, "iter": it+1}
        return sol
