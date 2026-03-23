import statistics

from modelos.representacao import Representacao
from modelos.objetivo_config import ObjetivoConfig


class Solucao:
    """Representação de uma solução CVRP como lista de rotas.

    Invariante: n_veiculos == len(rotas) — cada rota corresponde
    exatamente a um caminhão; não há multi-trip por veículo.
    """

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

    def verificar_cobertura(self, instancia=None):
        """Retorna o número de clientes não visitados na solução."""
        inst = instancia or self.instancia
        if inst is None:
            return 0
        n_clientes = getattr(inst, "n_clientes", 0)
        if n_clientes == 0:
            return 0
        todos_clientes = set(range(1, n_clientes + 1))
        visitados = {node for rota in self.rotas for node in rota if node != 0}
        return len(todos_clientes - visitados)

    def verificar_carga_minima(self, instancia=None, carga_minima=0.0):
        """Retorna {rota_idx: deficit} para rotas com carga abaixo do mínimo."""
        inst = instancia or self.instancia
        if inst is None or carga_minima <= 0:
            return {}
        demandas = getattr(inst, "demandas", None)
        if demandas is None:
            return {}
        deficits = {}
        for idx, rota in enumerate(self.rotas):
            carga = sum(float(demandas[node]) for node in rota if node != 0)
            deficit = carga_minima - carga
            if deficit > 0:
                deficits[idx] = deficit
        return deficits

    def verificar_desequilibrio_carga(self, instancia=None, limiar=0.3):
        """Retorna o excesso de coeficiente de variação da carga entre rotas.

        O coeficiente de variação (CV = desvio_padrão / média) mede o desequilíbrio
        relativo de carga entre caminhões. Penaliza apenas quando CV > limiar.
        Retorna 0.0 se houver menos de 2 rotas ou média zero.
        """
        inst = instancia or self.instancia
        demandas = getattr(inst, "demandas", None) if inst else None
        if demandas is None or len(self.rotas) < 2:
            return 0.0
        cargas = [
            sum(float(demandas[node]) for node in rota if node != 0)
            for rota in self.rotas
        ]
        media = statistics.mean(cargas)
        if media == 0:
            return 0.0
        cv = statistics.pstdev(cargas) / media
        return max(0.0, cv - limiar)

    def eh_valida(self, instancia=None):
        """True se a solução satisfaz todas as restrições hard: capacidade, cobertura e frota."""
        inst = instancia or self.instancia

        violacoes_capacidade = self.verificar_capacidade(inst)
        clientes_faltando = self.verificar_cobertura(inst)

        numero_caminhoes = getattr(inst, "numero_caminhoes", None)
        frota_excedida = (
            max(0, self.n_veiculos - numero_caminhoes)
            if numero_caminhoes is not None
            else 0
        )

        n_cap = len(violacoes_capacidade)
        self.violacoes = {
            "capacidade": n_cap,
            "cobertura": clientes_faltando,
            "frota_excedida": frota_excedida,
        }

        return n_cap == 0 and clientes_faltando == 0 and frota_excedida == 0

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
            config = ObjetivoConfig(
                matriz=matriz,
                numero_caminhoes=getattr(inst, "numero_caminhoes", None),
                n_clientes=getattr(inst, "n_clientes", None),
            )
        self.objetivo_config = config

        # Capacidade
        violacoes_capacidade = self.verificar_capacidade(inst)
        penalidade_capacidade = config.calcular_penalidade_capacidade(violacoes_capacidade)

        # Frota (penaliza apenas excesso acima do limite)
        penalidade_veiculos = config.calcular_penalidade_veiculos(self.n_veiculos)

        # Cobertura (clientes não visitados — quase hard constraint)
        clientes_faltando = self.verificar_cobertura(inst)
        penalidade_cobertura = config.calcular_penalidade_cobertura(clientes_faltando)

        # Carga mínima por rota
        carga_minima = getattr(inst, "carga_minima", 0.0)
        deficits_carga = self.verificar_carga_minima(inst, carga_minima)
        penalidade_carga_minima = config.calcular_penalidade_carga_minima(deficits_carga)

        # Desequilíbrio de carga entre rotas
        excesso_cv = self.verificar_desequilibrio_carga(inst, getattr(config, "limiar_desequilibrio", 0.3))
        penalidade_desequilibrio = config.calcular_penalidade_desequilibrio(excesso_cv)

        custo_objetivo = (
            custo_distancia
            + penalidade_capacidade
            + penalidade_veiculos
            + penalidade_cobertura
            + penalidade_carga_minima
            + penalidade_desequilibrio
        )

        self.custo_objetivo = custo_objetivo
        self.violacoes = {
            "capacidade": len(violacoes_capacidade),
            "capacidade_detalhes": violacoes_capacidade,
            "penalidade_capacidade": penalidade_capacidade,
            "penalidade_veiculos": penalidade_veiculos,
            "cobertura_clientes_faltando": clientes_faltando,
            "penalidade_cobertura": penalidade_cobertura,
            "carga_minima_deficits": deficits_carga,
            "penalidade_carga_minima": penalidade_carga_minima,
            "desequilibrio_cv_excesso": excesso_cv,
            "penalidade_desequilibrio": penalidade_desequilibrio,
        }

        return custo_objetivo

    def __repr__(self):
        repr_str = f"Solucao(n_rotas={self.n_veiculos}, custo={self.custo}"
        if self.custo_objetivo is not None:
            repr_str += f", custo_objetivo={self.custo_objetivo:.2f}"
        repr_str += f", violacoes={self.violacoes})"
        return repr_str
