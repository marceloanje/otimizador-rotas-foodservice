class Representacao:
    def __init__(self, rota):
        self.rota = rota

    def custo(self, matriz):
        total = 0
        for i in range(len(self.rota)-1):
            total += matriz[self.rota[i]][self.rota[i+1]]
        return total
