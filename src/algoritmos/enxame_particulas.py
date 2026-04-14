import random
import time
import numpy as np
from modelos.representacao import Representacao
from modelos.solucao import Solucao
from modelos.objetivo_config import ObjetivoConfig
from utilitarios.construtivas import nearest_neighbor_capacitado
from utilitarios.local_search import two_opt_intra

def split_into_routes(permutation, instancia):
    """
    Divide um giant tour (permutação de clientes) em rotas viáveis respeitando capacidade.

    É a função de decodificação usada pelo PSO: a posição de cada partícula
    é uma permutação, e esse split transforma em rotas concretas.
    """
    deposito = 0
    demandas = instancia.demandas
    capacidade = instancia.capacidade_caminhao

    rotas = []
    rota_atual = [deposito]
    carga_atual = 0.0

    for cliente in permutation:
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
    return solucao


def apply_swaps(permutation, swaps):
    """Aplica uma sequência de trocas a uma permutação."""
    p = permutation[:]
    for (i, j) in swaps:
        if 0 <= i < len(p) and 0 <= j < len(p):
            p[i], p[j] = p[j], p[i]
    return p


def generate_swaps_to_move(a, b):
    """
    Gera sequência de trocas para transformar a permutação a em b.

    Usado para calcular a "velocidade" no PSO: a diferença entre
    posição atual e pbest/gbest é expressa como lista de swaps.
    """
    swaps = []
    a = a[:]
    pos = {val: idx for idx, val in enumerate(a)}
    for i in range(len(a)):
        if a[i] != b[i]:
            j = pos[b[i]]
            swaps.append((i, j))
            a[i], a[j] = a[j], a[i]
            pos[a[j]] = j
            pos[a[i]] = i
    return swaps


class PSO:
    def __init__(self, instancia, n_particles=20, max_iter=100, c1=0.6, c2=0.6, inertia=0.9, config=None):
        self.inst = instancia
        self.matriz = instancia.matriz
        self.n = len(self.matriz)
        self.n_clientes = self.n - 1

        self.n_particles = n_particles
        self.max_iter = max_iter
        self.c1 = c1        # peso cognitivo (atração ao pbest)
        self.c2 = c2        # peso social (atração ao gbest)
        self.inertia = inertia

        self.config = config if config is not None else ObjetivoConfig(matriz=self.matriz)

    def _random_perm(self):
        """Permutação aleatória dos clientes (sem depósito)."""
        p = list(range(1, self.n))
        random.shuffle(p)
        return p

    def _evaluate_permutation(self, perm):
        """Decodifica a permutação em rotas e avalia a função objetivo."""
        solucao = split_into_routes(perm, self.inst)
        solucao = two_opt_intra(solucao, self.inst, self.config)
        return solucao.custo_objetivo, solucao

    def run(self):
        start = time.time()

        particles = [self._random_perm() for _ in range(self.n_particles)]

        pbest_costs = []
        pbests = []
        solucoes = []

        for p in particles:
            cost, sol = self._evaluate_permutation(p)
            pbests.append(list(p))
            pbest_costs.append(cost)
            solucoes.append(sol)

        gbest_idx = int(np.argmin(pbest_costs))
        gbest = list(pbests[gbest_idx])
        gbest_cost = pbest_costs[gbest_idx]
        gbest_solucao = solucoes[gbest_idx]

        historico = []

        for it in range(self.max_iter):
            for i in range(self.n_particles):
                current = particles[i]

                swaps_to_pbest = generate_swaps_to_move(current, pbests[i])
                swaps_to_gbest = generate_swaps_to_move(current, gbest)

                chosen_swaps = []

                # inércia: swap aleatório para manter diversidade
                if random.random() < self.inertia and len(current) > 1:
                    a = random.randrange(len(current))
                    b = random.randrange(len(current))
                    chosen_swaps.append((a, b))

                for sw in swaps_to_pbest:
                    if random.random() < self.c1:
                        chosen_swaps.append(sw)

                for sw in swaps_to_gbest:
                    if random.random() < self.c2:
                        chosen_swaps.append(sw)

                new = apply_swaps(current, chosen_swaps) if chosen_swaps else current[:]
                particles[i] = new

                cost, sol = self._evaluate_permutation(new)

                if cost < pbest_costs[i]:
                    pbests[i] = list(new)
                    pbest_costs[i] = cost
                    solucoes[i] = sol

                    if cost < gbest_cost:
                        gbest = list(new)
                        gbest_cost = cost
                        gbest_solucao = sol

            historico.append(gbest_cost)

        elapsed = time.time() - start
        gbest_solucao.tempo_computacional = elapsed
        gbest_solucao.meta = {
            "tempo": elapsed,
            "iter": self.max_iter,
            "historico_convergencia": historico,
            "max_iter": self.max_iter,
        }

        return gbest_solucao
