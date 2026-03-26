import random
import time
import numpy as np
from modelos.representacao import Representacao
from modelos.solucao import Solucao
from modelos.objetivo_config import ObjetivoConfig
from utilitarios.construtivas import nearest_neighbor_capacitado
import copy

class BuscaTabu:
    def __init__(self, instancia, max_iter=500, tabu_tenure=15, max_no_improve=100,
                 config=None, estrategia='sample', max_vizinhos=200):
        self.inst = instancia
        self.matriz = instancia.matriz
        self.n = len(self.matriz)
        self.demandas = instancia.demandas
        self.capacidade = instancia.capacidade_caminhao

        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        self.max_no_improve = max_no_improve
        self.estrategia = estrategia
        self.max_vizinhos = max_vizinhos

        self.config = config if config is not None else ObjetivoConfig(matriz=self.matriz)

    def _solucao_inicial(self):
        """Gera solução inicial usando heurística construtiva."""
        return nearest_neighbor_capacitado(self.inst)

    def _avaliar_solucao(self, solucao):
        """Avalia solução usando função objetivo com penalidades."""
        return solucao.avaliar(self.inst, self.config)

    # --- Geradores de move_ids (leves, sem criação de Solucao) ---

    def _gen_moves_2opt(self, solucao):
        """Gera move_ids 2-opt intra-rota."""
        for idx_rota, rota in enumerate(solucao.rotas):
            if len(rota) < 4:
                continue
            for i in range(1, len(rota) - 2):
                for j in range(i + 1, len(rota) - 1):
                    yield ('2opt_intra', idx_rota, i, j)

    def _gen_moves_relocate(self, solucao):
        """Gera move_ids de relocate de cliente entre rotas."""
        n_rotas = len(solucao.rotas)
        for idx_origem in range(n_rotas):
            for pos_cliente in range(1, len(solucao.rotas[idx_origem]) - 1):
                for idx_destino in range(n_rotas):
                    if idx_origem == idx_destino:
                        continue
                    for pos_insercao in range(1, len(solucao.rotas[idx_destino])):
                        yield ('relocate', idx_origem, pos_cliente, idx_destino, pos_insercao)

    def _gen_moves_swap(self, solucao):
        """Gera move_ids de swap de clientes entre duas rotas."""
        n_rotas = len(solucao.rotas)
        for i in range(n_rotas):
            for j in range(i + 1, n_rotas):
                for p1 in range(1, len(solucao.rotas[i]) - 1):
                    for p2 in range(1, len(solucao.rotas[j]) - 1):
                        yield ('swap', i, p1, j, p2)

    def _gerar_todos_moves(self, solucao):
        """Combina todos os geradores de move_ids."""
        yield from self._gen_moves_2opt(solucao)
        yield from self._gen_moves_relocate(solucao)
        yield from self._gen_moves_swap(solucao)

    def _aplicar_move(self, solucao, move_id):
        """Aplica um movimento e retorna nova Solucao."""
        rotas = [r[:] for r in solucao.rotas]
        tipo = move_id[0]

        if tipo == '2opt_intra':
            _, idx, i, j = move_id
            rotas[idx] = rotas[idx][:i] + rotas[idx][i:j+1][::-1] + rotas[idx][j+1:]

        elif tipo == 'relocate':
            _, io, pc, id_, pi = move_id
            cliente = rotas[io][pc]
            rotas[io] = rotas[io][:pc] + rotas[io][pc+1:]
            rotas[id_] = rotas[id_][:pi] + [cliente] + rotas[id_][pi:]

        elif tipo == 'swap':
            _, i, p1, j, p2 = move_id
            rotas[i][p1], rotas[j][p2] = rotas[j][p2], rotas[i][p1]

        return Solucao(rotas=rotas, instancia=self.inst)

    def _melhor_candidato(self, solucao, tabu, best_cost):
        """Seleciona o melhor candidato da vizinhança conforme a estratégia."""
        if self.estrategia == 'sample':
            moves = list(self._gerar_todos_moves(solucao))
            if len(moves) > self.max_vizinhos:
                moves = random.sample(moves, self.max_vizinhos)
        else:
            moves = self._gerar_todos_moves(solucao)

        best_cand, best_cand_cost, best_move = None, float('inf'), None

        for move_id in moves:
            is_tabu = tabu.get(move_id, 0) > 0
            candidate = self._aplicar_move(solucao, move_id)
            cand_cost = self._avaliar_solucao(candidate)

            if is_tabu and cand_cost >= best_cost:
                continue

            if cand_cost < best_cand_cost:
                best_cand = candidate
                best_cand_cost = cand_cost
                best_move = move_id
                if self.estrategia == 'first' and cand_cost < best_cost:
                    break

        return best_cand, best_cand_cost, best_move

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
            neighborhood_best, neighborhood_best_cost, neighborhood_best_move = \
                self._melhor_candidato(current, tabu, best_cost)

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
