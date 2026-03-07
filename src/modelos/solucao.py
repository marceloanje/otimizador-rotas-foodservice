from modelos.representacao import Representacao


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

    def eh_valida(self, instancia=None):
        inst = instancia or self.instancia
        if inst is None:
            raise ValueError("Instancia necessária para checar validade")
        capacidade = getattr(inst, "capacidade_caminhao", None)
        matr = getattr(inst, "matriz", None)
        demandas = getattr(inst, "demandas", getattr(inst, "demands", None))

        viol_cap = 0

        for rota in self.rotas:
            # Checagem de capacidade da rota
            if capacidade is not None and demandas is not None:
                carga = 0.0
                for node in rota:
                    # assumir nó 0 como depósito
                    if node == 0:
                        continue
                    carga += float(demandas[node])
                if carga > capacidade:
                    viol_cap += 1

        self.violacoes = {"capacidade": viol_cap}
        return viol_cap == 0

    def __repr__(self):
        return f"Solucao(n_rotas={self.n_veiculos}, custo={self.custo}, violacoes={self.violacoes})"
