import pandas as pd
import numpy as np
import ast

class Instancia:
    """Representação de uma instância do problema VRPTW. Teste

    Attributes
    ----------
    df : pandas.DataFrame
        DataFrame cru lido a partir do CSV de pedidos (inclui depósito na linha 0).
    posicoes : list[tuple]
        Lista de tuplas (lat, lon) para cada nó, índice 0 é o depósito.
    matriz : numpy.ndarray | None
        Matriz de distâncias (n x n). Pode ser gerada ficticiamente ou carregada.
    demandas : list[float]
        Demanda de cada nó (unidades genéricas). Índice 0 corresponde ao depósito
        e normalmente tem demanda 0.
    janelas_tempo : list[tuple]
        Lista de tuplas (inicio, fim) em minutos desde o início do dia para cada nó.
    tempos_servico : list[float]
        Tempo de atendimento (em minutos) em cada nó.
    capacidade_veiculo : int | None
        Capacidade (unidades) dos veículos (assume frota homogênea quando fornecido).
    n_clientes : int
        Número de clientes (excluindo depósito).
    depot_index : int
        Índice do depósito (sempre 0 no formato adotado).
    """

    def __init__(self, df, posicoes,
                 demandas=None,
                 janelas_tempo=None,
                 tempos_servico=None,
                 capacidade_veiculo=None,
                 n_vehicles=1,
                 depot_index=0):
        self.df = df
        self.posicoes = posicoes
        self.matriz = None

        # VRPTW-specific attributes
        n = len(posicoes)
        self.demands = (
            list(demandas) if demandas is not None else [0.0] * n
        )
        # Alias em português para compatibilidade com a especificação
        self.demandas = self.demands
        self.janelas_tempo = (
            list(janelas_tempo) if janelas_tempo is not None else [(0, 1440)] * n
        )
        self.tempos_servico = (
            list(tempos_servico) if tempos_servico is not None else [0.0] * n
        )
        self.capacidade_veiculo = capacidade_veiculo
        self.n_vehicles = n_vehicles
        self.n_clientes = max(0, n - 1)
        self.depot_index = depot_index

    @classmethod
    def from_csv(cls, path, vehicle_capacity=None, n_vehicles=1, tw_window_width=None):
        """Cria uma `Instancia` a partir de um CSV de pedidos.

        Parameters
        ----------
        path : str
            Caminho para o arquivo CSV contendo pedidos. Deve conter colunas
            `posicao` (ex: "(lat, lon)") ou `lat`/`lon` separadas. A primeira
            linha é considerada o depósito (nó 0).
        vehicle_capacity : int, optional
            Capacidade dos veículos (se conhecida). Caso contrário fica `None`.
        n_vehicles : int
            Número de veículos disponíveis (padrão 1).
        tw_window_width : int | None
            Se fornecido e a coluna `hora_de_entrega` for a única disponível,
            cria janelas simétricas de largura `tw_window_width` (em minutos)
            centradas em `hora_de_entrega`.

        Notes
        -----
        Colunas reconhecidas (opcionais): `quantidade_produto` (demand),
        `hora_de_entrega` (horário desejado, pode ser convertido em janela),
        `tw_start`, `tw_end` (janelas explícitas em minutos), `service_time`.
        """

        df = pd.read_csv(path)

        # Normaliza `hora_de_entrega`: aceita strings "HH:MM" ou números, converte em minutos
        if "hora_de_entrega" in df.columns:
            def _hora_para_minutos(h):
                if pd.isna(h) or h == "":
                    return 0.0
                # strings no formato HH:MM
                if isinstance(h, str):
                    s = h.strip()
                    if ":" in s:
                        parts = s.split(":")
                        try:
                            hh = int(parts[0])
                            mm = int(parts[1])
                            return float(hh * 60 + mm)
                        except Exception:
                            try:
                                return float(s)
                            except Exception:
                                return 0.0
                    else:
                        try:
                            return float(s)
                        except Exception:
                            return 0.0
                try:
                    return float(h)
                except Exception:
                    return 0.0

            df["hora_de_entrega"] = df["hora_de_entrega"].apply(_hora_para_minutos)

        # Parse positions
        if "posicao" in df.columns:
            lats = []
            lons = []
            for p in df["posicao"]:
                lat, lon = ast.literal_eval(p)
                lats.append(lat)
                lons.append(lon)

            df["lat"] = lats
            df["lon"] = lons

        posicoes = list(zip(df["lat"], df["lon"]))

        # Demands
        if "quantidade_produto" in df.columns:
            demandas = df["quantidade_produto"].fillna(0).astype(float).tolist()
        else:
            demandas = [0.0] * len(df)

        # Service times
        if "service_time" in df.columns:
            tempos_servico = df["service_time"].fillna(0).astype(float).tolist()
        elif "tempo_servico" in df.columns:
            tempos_servico = df["tempo_servico"].fillna(0).astype(float).tolist()
        else:
            tempos_servico = [0.0] * len(df)

        # Time windows: prefer explicit `tw_start`/`tw_end`, else try `hora_de_entrega`
        janelas = []
        if "tw_start" in df.columns and "tw_end" in df.columns:
            for s, e in zip(df["tw_start"].fillna(0), df["tw_end"].fillna(1440)):
                janelas.append((float(s), float(e)))
        elif "hora_de_entrega" in df.columns and tw_window_width is not None:
            half = tw_window_width / 2.0
            for h in df["hora_de_entrega"].fillna(0):
                cen = float(h)
                janelas.append((max(0.0, cen - half), cen + half))
        else:
            # Defaults: depot has full day window, others 0-1440
            for i in range(len(df)):
                if i == 0:
                    janelas.append((0.0, 1440.0))
                else:
                    janelas.append((0.0, 1440.0))

        instancia = cls(
            df=df,
            posicoes=posicoes,
            demandas=demandas,
            janelas_tempo=janelas,
            tempos_servico=tempos_servico,
            capacidade_veiculo=vehicle_capacity,
            n_vehicles=n_vehicles,
            depot_index=0,
        )

        instancia.validate()
        return instancia

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

    def validate(self):
        """Valida consistência básica da instância.

        Levanta `ValueError` se tamanhos divergirem ou se o depósito não for o nó 0.
        """
        n = len(self.posicoes)

        if len(self.demandas) != n:
            raise ValueError("Comprimento de `demandas` incompatível com `posicoes`")
        if len(self.janelas_tempo) != n:
            raise ValueError("Comprimento de `janelas_tempo` incompatível com `posicoes`")
        if len(self.tempos_servico) != n:
            raise ValueError("Comprimento de `tempos_servico` incompatível com `posicoes`")
        if self.depot_index != 0:
            raise ValueError("Depósito deve ser o nó 0 (depot_index=0)")

