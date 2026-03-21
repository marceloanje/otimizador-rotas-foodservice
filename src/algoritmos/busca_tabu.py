import random
import time
import numpy as np
from modelos.representacao import Representacao
from modelos.solucao import Solucao
from modelos.objetivo_config import ObjetivoConfig
from utilitarios.construtivas import nearest_neighbor_capacitado
import copy

class BuscaTabu:
    def __init__(self, instancia, max_iter=500, tabu_tenure=15, max_no_improve=100, config=None):
        self.inst = instancia
        self.matriz = instancia.matriz
        self.n = len(self.matriz)
        self.demandas = instancia.demandas
        self.capacidade = instancia.capacidade_caminhao

        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        self.max_no_improve = max_no_improve

        self.config = config if config is not None else ObjetivoConfig(matriz=self.matriz)

    def _solucao_inicial(self):
        """Gera solução inicial usando heurística construtiva."""
        return nearest_neighbor_capacitado(self.inst)

    def _avaliar_solucao(self, solucao):
        """Avalia solução usando função objetivo com penalidades."""
        return solucao.avaliar(self.inst, self.config)

    def _two_opt_intra_rota(self, solucao):
        """Gera vizinhos aplicando 2-opt dentro de cada rota."""
        for idx_rota, rota in enumerate(solucao.rotas):
            # 2-opt só faz sentido para rotas com pelo menos 4 nós (dep, c1, c2, dep)
            if len(rota) < 4:
                continue

            # não mexer nos depósitos (primeiro e último)
            for i in range(1, len(rota) - 2):
                for j in range(i + 1, len(rota) - 1):
                    nova_rota = rota[:i] + rota[i:j+1][::-1] + rota[j+1:]

                    novas_rotas = [r[:] for r in solucao.rotas]
                    novas_rotas[idx_rota] = nova_rota

                    nova_solucao = Solucao(rotas=novas_rotas, instancia=self.inst)

                    move_id = ('2opt_intra', idx_rota, i, j)
                    yield (move_id, nova_solucao)

    def _relocate_cliente(self, solucao):
        """Gera vizinhos movendo um cliente de uma rota para outra."""
        n_rotas = len(solucao.rotas)

        for idx_origem in range(n_rotas):
            rota_origem = solucao.rotas[idx_origem]

            for pos_cliente in range(1, len(rota_origem) - 1):
                cliente = rota_origem[pos_cliente]

                for idx_destino in range(n_rotas):
                    if idx_origem == idx_destino:
                        continue

                    rota_destino = solucao.rotas[idx_destino]

                    for pos_insercao in range(1, len(rota_destino)):
                        nova_rota_origem = rota_origem[:pos_cliente] + rota_origem[pos_cliente+1:]
                        nova_rota_destino = rota_destino[:pos_insercao] + [cliente] + rota_destino[pos_insercao:]

                        novas_rotas = [r[:] for r in solucao.rotas]
                        novas_rotas[idx_origem] = nova_rota_origem
                        novas_rotas[idx_destino] = nova_rota_destino

                        nova_solucao = Solucao(rotas=novas_rotas, instancia=self.inst)

                        move_id = ('relocate', idx_origem, pos_cliente, idx_destino, pos_insercao)
                        yield (move_id, nova_solucao)

    def _swap_clientes(self, solucao):
        """Gera vizinhos trocando clientes entre duas rotas."""
        n_rotas = len(solucao.rotas)

        for idx_rota1 in range(n_rotas):
            for idx_rota2 in range(idx_rota1 + 1, n_rotas):
                rota1 = solucao.rotas[idx_rota1]
                rota2 = solucao.rotas[idx_rota2]

                for pos1 in range(1, len(rota1) - 1):
                    for pos2 in range(1, len(rota2) - 1):
                        nova_rota1 = rota1[:]
                        nova_rota2 = rota2[:]
                        nova_rota1[pos1], nova_rota2[pos2] = nova_rota2[pos2], nova_rota1[pos1]

                        novas_rotas = [r[:] for r in solucao.rotas]
                        novas_rotas[idx_rota1] = nova_rota1
                        novas_rotas[idx_rota2] = nova_rota2

                        nova_solucao = Solucao(rotas=novas_rotas, instancia=self.inst)

                        move_id = ('swap', idx_rota1, pos1, idx_rota2, pos2)
                        yield (move_id, nova_solucao)

    def _gerar_vizinhanca(self, solucao):
        """Combina 2-opt intra-rota, relocate e swap entre rotas."""
        yield from self._two_opt_intra_rota(solucao)
        yield from self._relocate_cliente(solucao)
        yield from self._swap_clientes(solucao)

    def run(self):
        start = time.time()

        current = self._solucao_inicial()
        self._avaliar_solucao(current)

        best = copy.deepcopy(current)
        best_cost = best.custo_objetivo

        # lista tabu: move_id → tenure restante
        tabu = dict()
        iter_no_improve = 0

        for it in range(self.max_iter):
            neighborhood_best = None
            neighborhood_best_cost = float("inf")
            neighborhood_best_move = None

            for move_id, candidate in self._gerar_vizinhanca(current):
                cand_cost = self._avaliar_solucao(candidate)

                is_tabu = move_id in tabu and tabu[move_id] > 0

                # critério de aspiração: aceita movimento tabu se bate o melhor global
                if is_tabu and cand_cost >= best_cost:
                    continue

                if cand_cost < neighborhood_best_cost:
                    neighborhood_best = candidate
                    neighborhood_best_cost = cand_cost
                    neighborhood_best_move = move_id

            if neighborhood_best is None:
                break

            current = neighborhood_best

            # decrementar tenure e adicionar o movimento escolhido
            keys = list(tabu.keys())
            for k in keys:
                tabu[k] -= 1
                if tabu[k] <= 0:
                    del tabu[k]

            tabu[neighborhood_best_move] = self.tabu_tenure

            if neighborhood_best_cost < best_cost:
                best = copy.deepcopy(neighborhood_best)
                best_cost = neighborhood_best_cost
                iter_no_improve = 0
            else:
                iter_no_improve += 1

            if iter_no_improve >= self.max_no_improve:
                break

        elapsed = time.time() - start
        best.tempo_computacional = elapsed
        best.meta = {"tempo": elapsed, "iter": it+1}
        return best
