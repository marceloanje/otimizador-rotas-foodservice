import pandas as pd
import numpy as np
import ast

class Instancia:
    def __init__(self, df, posicoes):
        self.df = df
        self.posicoes = posicoes
        self.matriz = None

    @classmethod
    def from_csv(cls, path):
        df = pd.read_csv(path)

        if "posicao" in df.columns:
            lats = []
            lons = []
            for p in df["posicao"]:
                # Converte a string "(a, b)" para tupla numérica
                lat, lon = ast.literal_eval(p)
                lats.append(lat)
                lons.append(lon)

            df["lat"] = lats
            df["lon"] = lons

        posicoes = list(zip(df["lat"], df["lon"]))

        return cls(df, posicoes)

    def gerar_matriz_distancias_ficticia(self):
        n = len(self.posicoes)
        self.matriz = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    self.matriz[i][j] = 0
                else:
                    self.matriz[i][j] = np.linalg.norm(
                        np.array(self.posicoes[i]) - np.array(self.posicoes[j])
                    )
