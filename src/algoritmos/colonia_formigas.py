class ACO:
    def __init__(self, instancia, n_formigas=10, iter=20):
        self.inst = instancia
        self.n_formigas = n_formigas
        self.iter = iter
        self.feromonio = np.ones(instancia.matriz.shape)

    def run(self):
        melhor_rota = None
        melhor_custo = float("inf")

        for _ in range(self.iter):
            for f in range(self.n_formigas):
                rota = self.construir_rota()
                custo = Representacao(rota).custo(self.inst.matriz)

                if custo < melhor_custo:
                    melhor_custo = custo
                    melhor_rota = rota
            
            self.atualizar_feromonio(melhor_rota, melhor_custo)

        return Solucao(melhor_rota, melhor_custo)
