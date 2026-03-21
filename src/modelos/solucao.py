from modelos.representacao import Representacao
from modelos.objetivo_config import ObjetivoConfig


class Solucao:
    def __init__(self, rotas=None, custo=None, instancia=None):
        # rotas: lista de rotas, cada rota é uma lista de nós (ex: [0,1,2,0])
        self.rotas = rotas or []
        self.custo = custo
        self.n_veiculos = len(self.rotas)
        self.tempo_computacional = None
        self.violacoes = {}
        self.meta = {}
        self.instancia = instancia
        self.custo_objetivo = None
        self.objetivo_config = None

    def adicionar_rota(self, rota):
        self.rotas.append(rota)
        self.n_veiculos = len(self.rotas)

    def calcular_custo(self, instancia=None):
        inst = instancia or self.instancia
        if inst is None:
            raise ValueError("Instancia (com matriz) necessária para calcular custo")
        matriz = getattr(inst, "matriz", None)
        if matriz is None:
            raise ValueError("Instancia não possui matriz de distâncias")

        total = 0.0
        for rota in self.rotas:
            total += Representacao(rota).custo(matriz)

        self.custo = total
        return total

    def verificar_capacidade(self, instancia=None):
        """Retorna {rota_idx: excesso} para cada rota que viola capacidade."""
        inst = instancia or self.instancia
        if inst is None:
            raise ValueError("Instancia necessária para checar capacidade")

        capacidade = getattr(inst, "capacidade_caminhao", None)
        demandas = getattr(inst, "demandas", getattr(inst, "demands", None))

        if capacidade is None or demandas is None:
            return {}

        violacoes_detalhadas = {}

        for idx, rota in enumerate(self.rotas):
            carga = 0.0
            for node in rota:
                if node == 0:  # depósito
                    continue
                carga += float(demandas[node])

            excesso = carga - capacidade
            if excesso > 0:
                violacoes_detalhadas[idx] = excesso

        return violacoes_detalhadas

    def eh_valida(self, instancia=None):
        """True se nenhuma rota viola a capacidade."""
        violacoes_capacidade = self.verificar_capacidade(instancia)
        n_violacoes = len(violacoes_capacidade)
        self.violacoes = {"capacidade": n_violacoes}
        return n_violacoes == 0

    def avaliar(self, instancia=None, config=None):
        """Calcula a função objetivo: distância total + penalidades por violações.

        É a métrica unificada usada por todos os algoritmos — nunca use
        calcular_custo() diretamente para comparar soluções.
        """
        inst = instancia or self.instancia
        if inst is None:
            raise ValueError("Instancia necessária para avaliar")

        custo_distancia = self.calcular_custo(inst)

        if config is None:
            matriz = getattr(inst, "matriz", None)
            config = ObjetivoConfig(matriz=matriz)
        self.objetivo_config = config

        violacoes_capacidade = self.verificar_capacidade(inst)
        penalidade_capacidade = config.calcular_penalidade_capacidade(violacoes_capacidade)
        penalidade_veiculos = config.calcular_penalidade_veiculos(self.n_veiculos)

        custo_objetivo = custo_distancia + penalidade_capacidade + penalidade_veiculos

        self.custo_objetivo = custo_objetivo
        self.violacoes = {
            "capacidade": len(violacoes_capacidade),
            "capacidade_detalhes": violacoes_capacidade,
            "penalidade_capacidade": penalidade_capacidade,
            "penalidade_veiculos": penalidade_veiculos
        }

        return custo_objetivo

    def __repr__(self):
        repr_str = f"Solucao(n_rotas={self.n_veiculos}, custo={self.custo}"
        if self.custo_objetivo is not None:
            repr_str += f", custo_objetivo={self.custo_objetivo:.2f}"
        repr_str += f", violacoes={self.violacoes})"
        return repr_str
