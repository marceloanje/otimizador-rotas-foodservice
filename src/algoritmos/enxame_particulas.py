import random
import time
import numpy as np
from modelos.representacao import Representacao
from modelos.solucao import Solucao
from modelos.objetivo_config import ObjetivoConfig
from utilitarios.construtivas import nearest_neighbor_capacitado

def split_into_routes(permutation, instancia):
    """
    Split a giant tour (permutation of customers) into feasible routes.
    
    Parameters
    ----------
    permutation : list
        Permutation of customers (excluding depot).
    instancia : Instancia
        Problem instance with demands and capacity.
        
    Returns
    -------
    Solucao
        Solution with multiple routes respecting capacity.
    """
    deposito = 0
    demandas = instancia.demandas
    capacidade = instancia.capacidade_caminhao
    
    rotas = []
    rota_atual = [deposito]
    carga_atual = 0.0
    
    for cliente in permutation:
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
    
    solucao = Solucao(rotas=rotas, instancia=instancia)
    return solucao


def apply_swaps(permutation, swaps):
    """Apply a sequence of swaps to a permutation."""
    p = permutation[:]
    for (i, j) in swaps:
        if 0 <= i < len(p) and 0 <= j < len(p):
            p[i], p[j] = p[j], p[i]
    return p


def generate_swaps_to_move(a, b):
    """
    Generate sequence of swaps to transform permutation a into b.
    
    Parameters
    ----------
    a : list
        Source permutation.
    b : list
        Target permutation.
        
    Returns
    -------
    list
        List of swap tuples (i, j).
    """
    swaps = []
    a = a[:]
    pos = {val: idx for idx, val in enumerate(a)}
    for i in range(len(a)):
        if a[i] != b[i]:
            j = pos[b[i]]
            swaps.append((i, j))
            # Apply swap to 'a' and update positions
            a[i], a[j] = a[j], a[i]
            pos[a[j]] = j
            pos[a[i]] = i
    return swaps


class PSO:
    def __init__(self, instancia, n_particles=20, max_iter=100, c1=0.6, c2=0.6, inertia=0.9, config=None):
        self.inst = instancia
        self.matriz = instancia.matriz
        self.n = len(self.matriz)
        
        # Number of customers (excluding depot)
        self.n_clientes = self.n - 1

        self.n_particles = n_particles
        self.max_iter = max_iter
        self.c1 = c1  # Cognitive weight
        self.c2 = c2  # Social weight
        self.inertia = inertia
        
        # Configuração de objetivo
        self.config = config if config is not None else ObjetivoConfig(matriz=self.matriz)

    def _random_perm(self):
        """Generate random permutation of customers (excluding depot)."""
        p = list(range(1, self.n))
        random.shuffle(p)
        return p
    
    def _evaluate_permutation(self, perm):
        """Evaluate a permutation by splitting into routes and computing objective."""
        solucao = split_into_routes(perm, self.inst)
        return solucao.avaliar(self.inst, self.config), solucao

    def run(self):
        start = time.time()

        # Initialize particles as permutations of customers
        particles = [self._random_perm() for _ in range(self.n_particles)]
        
        # Evaluate initial particles
        pbest_costs = []
        pbests = []
        solucoes = []
        
        for p in particles:
            cost, sol = self._evaluate_permutation(p)
            pbests.append(list(p))
            pbest_costs.append(cost)
            solucoes.append(sol)

        # Global best
        gbest_idx = int(np.argmin(pbest_costs))
        gbest = list(pbests[gbest_idx])
        gbest_cost = pbest_costs[gbest_idx]
        gbest_solucao = solucoes[gbest_idx]

        for it in range(self.max_iter):
            for i in range(self.n_particles):
                current = particles[i]

                # Construct movements towards pbest and gbest
                swaps_to_pbest = generate_swaps_to_move(current, pbests[i])
                swaps_to_gbest = generate_swaps_to_move(current, gbest)

                chosen_swaps = []
                
                # Inertia: random swap
                if random.random() < self.inertia and len(current) > 1:
                    a = random.randrange(len(current))
                    b = random.randrange(len(current))
                    chosen_swaps.append((a, b))

                # Apply some swaps towards pbest
                for sw in swaps_to_pbest:
                    if random.random() < self.c1:
                        chosen_swaps.append(sw)

                # Apply some swaps towards gbest
                for sw in swaps_to_gbest:
                    if random.random() < self.c2:
                        chosen_swaps.append(sw)

                # Apply swaps
                if chosen_swaps:
                    new = apply_swaps(current, chosen_swaps)
                else:
                    new = current[:]

                particles[i] = new

                # Evaluate new permutation
                cost, sol = self._evaluate_permutation(new)

                # Update personal best
                if cost < pbest_costs[i]:
                    pbests[i] = list(new)
                    pbest_costs[i] = cost
                    solucoes[i] = sol

                    # Update global best
                    if cost < gbest_cost:
                        gbest = list(new)
                        gbest_cost = cost
                        gbest_solucao = sol

        elapsed = time.time() - start
        gbest_solucao.tempo_computacional = elapsed
        gbest_solucao.meta = {"tempo": elapsed, "iter": self.max_iter}
        
        return gbest_solucao

