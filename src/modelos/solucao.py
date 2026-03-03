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

        capacidade = getattr(inst, "capacidade_veiculo", None)
        matr = getattr(inst, "matriz", None)
        demandas = getattr(inst, "demandas", getattr(inst, "demands", None))
        janelas = getattr(inst, "janelas_tempo", None)
        tempos_serv = getattr(inst, "tempos_servico", None)
        depot = getattr(inst, "depot_index", 0)

        viol_cap = 0
        viol_tw = 0

        for rota in self.rotas:
            # Capacidade
            if capacidade is not None and demandas is not None:
                carga = 0.0
                for node in rota:
                    if node == depot:
                        continue
                    carga += float(demandas[node])
                if carga > capacidade:
                    viol_cap += 1

            # Janelas de tempo (checagem aproximada usando matriz de distâncias)
            if matr is not None and janelas is not None and tempos_serv is not None:
                tempo = 0.0
                ok = True
                for i in range(len(rota) - 1):
                    a = rota[i]
                    b = rota[i + 1]
                    # tempo de viagem
                    tempo += float(matr[a][b])
                    # chegada em b
                    tw_start, tw_end = janelas[b]
                    if tempo < tw_start:
                        tempo = tw_start
                    if tempo > tw_end:
                        ok = False
                        break
                    # adicionar tempo de serviço em b
                    tempo += float(tempos_serv[b])
                if not ok:
                    viol_tw += 1

        self.violacoes = {"capacidade": viol_cap, "janela_tempo": viol_tw}
        return (viol_cap + viol_tw) == 0

    def __repr__(self):
        return f"Solucao(n_rotas={self.n_veiculos}, custo={self.custo}, violacoes={self.violacoes})"
