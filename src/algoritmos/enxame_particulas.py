import random
import time
import numpy as np
from modelos.representacao import Representacao
from modelos.solucao import Solucao

def custo_rota(matriz, rota):
    return Representacao(rota).custo(matriz)

def apply_swaps(rota, swaps):
    r = rota[:]
    for (i, j) in swaps:
        r[i], r[j] = r[j], r[i]
    return r

def generate_swaps_to_move(a, b):
    # Gera sequência de swaps para transformar a em b (simples)
    swaps = []
    a = a[:]
    pos = {val: idx for idx, val in enumerate(a)}
    for i in range(len(a)):
        if a[i] != b[i]:
            j = pos[b[i]]
            swaps.append((i, j))
            # Aplicar swap em 'a' e atualizar posições
            a[i], a[j] = a[j], a[i]
            pos[a[j]] = j
            pos[a[i]] = i
    return swaps

class PSO:
    def __init__(self, instancia, n_particles=20, max_iter=100, c1=0.6, c2=0.6, inertia=0.9):
        self.inst = instancia
        self.matriz = instancia.matriz
        self.n = len(self.matriz)

        self.n_particles = n_particles
        self.max_iter = max_iter
        self.c1 = c1  # Peso cognitivo
        self.c2 = c2  # Peso social
        self.inertia = inertia

    def _random_perm(self):
        p = list(range(self.n))
        random.shuffle(p)
        return p

    def run(self):
        start = time.time()

        particles = [self._random_perm() for _ in range(self.n_particles)]
        pbests = [list(p) for p in particles]
        pbest_costs = [custo_rota(self.matriz, p) for p in particles]

        gbest_idx = int(np.argmin(pbest_costs))
        gbest = list(pbests[gbest_idx])
        gbest_cost = pbest_costs[gbest_idx]

        for it in range(self.max_iter):
            for i in range(self.n_particles):
                current = particles[i]

                # Construir movimentos rumo ao pbest e gbest
                swaps_to_pbest = generate_swaps_to_move(current, pbests[i])
                swaps_to_gbest = generate_swaps_to_move(current, gbest)

                chosen_swaps = []
                if random.random() < self.inertia and len(current) > 1:
                    a = random.randrange(self.n)
                    b = random.randrange(self.n)
                    chosen_swaps.append((a,b))

                # Aplicar alguns swaps direcionados ao pbest
                for sw in swaps_to_pbest:
                    if random.random() < self.c1:
                        chosen_swaps.append(sw)

                # Aplicar alguns swaps direcionados ao gbest
                for sw in swaps_to_gbest:
                    if random.random() < self.c2:
                        chosen_swaps.append(sw)

                # Aplicar swaps (atenção a duplicidades)
                if chosen_swaps:
                    new = apply_swaps(current, chosen_swaps)
                else:
                    new = current[:]

                particles[i] = new

                cost = custo_rota(self.matriz, new)

                if cost < pbest_costs[i]:
                    pbests[i] = list(new)
                    pbest_costs[i] = cost

                    if cost < gbest_cost:
                        gbest = list(new)
                        gbest_cost = cost

        elapsed = time.time() - start
        sol = Solucao(gbest, gbest_cost)
        sol.meta = {"tempo": elapsed, "iter": self.max_iter}
        return sol
