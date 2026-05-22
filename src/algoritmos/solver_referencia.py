import time

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from modelos.solucao import Solucao


class SolverExato:
    """Solver exato para CVRP/VRPTW usando OR-Tools como baseline de qualidade.

    Expõe a mesma interface dos meta-heurísticos:
        solver = SolverExato(instancia)
        solucao = solver.run()

    Detecta automaticamente se a instância tem janelas de tempo (VRPTW)
    ou é CVRP puro. Utilizar com n_clientes <= 20.

    Parâmetros
    ----------
    tempo_limite : int
        Limite de tempo em segundos para o solver (default: 60).
    config : ObjetivoConfig | None
        Configuração de penalidades para avaliar a solução retornada.
        Se None, usa os pesos padrão derivados da matriz.
    """

    def __init__(self, instancia, tempo_limite=60, config=None):
        self.instancia = instancia
        self.tempo_limite = tempo_limite
        self.config = config

    def run(self) -> Solucao:
        inst = self.instancia
        n_nodes = inst.n_clientes + 1  # inclui depósito (nó 0)

        manager = pywrapcp.RoutingIndexManager(n_nodes, inst.numero_caminhoes, 0)
        routing = pywrapcp.RoutingModel(manager)

        # --- Callback de distância -----------------------------------------
        def distance_callback(from_idx, to_idx):
            i = manager.IndexToNode(from_idx)
            j = manager.IndexToNode(to_idx)
            return int(inst.matriz[i][j])

        transit_cb = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

        # --- Dimensão de capacidade ----------------------------------------
        def demand_callback(from_idx):
            node = manager.IndexToNode(from_idx)
            return int(inst.demandas[node])

        demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb,
            0,  # sem slack de capacidade
            [int(inst.capacidade_caminhao)] * inst.numero_caminhoes,
            True,
            "Capacity",
        )

        # --- Dimensão de tempo (VRPTW) — ativa apenas se disponível --------
        matriz_tempos = getattr(inst, "matriz_tempos", None)
        janelas_tempo = getattr(inst, "janelas_tempo", None)
        tempos_servico = getattr(inst, "tempos_servico", None)
        vrptw = matriz_tempos is not None and janelas_tempo is not None

        if vrptw:
            def time_callback(from_idx, to_idx):
                i = manager.IndexToNode(from_idx)
                j = manager.IndexToNode(to_idx)
                travel = int(matriz_tempos[i][j])
                service = int(tempos_servico[i]) if tempos_servico else 0
                return travel + service

            time_cb = routing.RegisterTransitCallback(time_callback)
            routing.AddDimension(
                time_cb,
                60,    # espera máxima num nó (minutos)
                1440,  # horizonte do dia (minutos)
                False,
                "Time",
            )
            time_dim = routing.GetDimensionOrDie("Time")
            for node in range(1, n_nodes):
                idx = manager.NodeToIndex(node)
                ini, fim = janelas_tempo[node]
                time_dim.CumulVar(idx).SetRange(int(ini), int(fim))

        # --- Parâmetros de busca -------------------------------------------
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = self.tempo_limite

        # --- Resolver -------------------------------------------------------
        t0 = time.time()
        solution = routing.SolveWithParameters(search_params)
        t1 = time.time()

        status = routing.status()

        # --- Extrair rotas --------------------------------------------------
        rotas = []
        if solution:
            for v in range(inst.numero_caminhoes):
                rota = []
                index = routing.Start(v)
                while not routing.IsEnd(index):
                    rota.append(manager.IndexToNode(index))
                    index = solution.Value(routing.NextVar(index))
                rota.append(0)  # depósito final
                if len(rota) > 2:  # ignora rotas vazias [0, 0]
                    rotas.append(rota)

        sol = Solucao(rotas=rotas, instancia=inst)
        sol.tempo_computacional = t1 - t0
        sol.meta = {
            "status_ortools": status,
            "vrptw": vrptw,
            "tempo_limite": self.tempo_limite,
        }
        sol.avaliar(inst, self.config)
        return sol
