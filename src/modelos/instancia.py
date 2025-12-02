import pandas as pd
import numpy as np

class Instancia:
    def __init__(self, clientes, posicoes):
        self.clientes = clientes
        self.pos = posicoes
        self.matriz = None

    @classmethod
    def from_csv(cls, path):
        df = pd.read_csv(path)
        posicoes = list(zip(df["lat"], df["lon"]))
        return cls(df, posicoes)

    def gerar_matriz_distancias_ficticia(self):
        n = len(self.pos)
        self.matriz = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    self.matriz[i][j] = 0
                else:
                    # Distância euclidiana fictícia
                    self.matriz[i][j] = np.linalg.norm(
                        np.array(self.pos[i]) - np.array(self.pos[j])
                    )
