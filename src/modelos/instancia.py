import pandas as pd
import numpy as np
import ast

class Instancia:
    """Representação de uma instância do problema VRPTW.

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
        self._validar(requer_matriz=False)

    def _validar(self, requer_matriz: bool = False, epsilon: float = 1e-8):
        """Implementação interna de validação.

        Parâmetros
        ----------
        requer_matriz : bool
            Se True, lança erro quando `self.matriz` for None. Padrão False
            para compatibilidade; chamadores (algoritmos) podem exigir a
            matriz antes de executar.
        epsilon : float
            Tolerância usada para verificar se elementos da diagonal estão
            próximos de zero.
        """

        import numbers as numeros

        n = len(self.posicoes)

        # Consistência de tamanhos
        if not hasattr(self.posicoes, "__len__"):
            raise TypeError(f"Inválido 'posicoes': esperada uma sequência de posições, obteve {type(self.posicoes)}")

        if len(self.demandas) != n:
            raise ValueError(f"Comprimento inválido em 'demandas': esperado {n} (igual a 'posicoes'), obtido {len(self.demandas)}")
        if len(self.janelas_tempo) != n:
            raise ValueError(f"Comprimento inválido em 'janelas_tempo': esperado {n} (igual a 'posicoes'), obtido {len(self.janelas_tempo)}")
        if len(self.tempos_servico) != n:
            raise ValueError(f"Comprimento inválido em 'tempos_servico': esperado {n} (igual a 'posicoes'), obtido {len(self.tempos_servico)}")

        # Verifica depot_index
        if not isinstance(self.depot_index, int):
            raise TypeError(f"Índice de depósito inválido: esperado int, obteve {type(self.depot_index)}")
        if not (0 <= self.depot_index < n):
            raise ValueError(f"Índice de depósito inválido: esperado valor em [0, {n-1}], obteve {self.depot_index}")

        # Demandas: numéricas e não-negativas
        for idx, d in enumerate(self.demandas):
            if not isinstance(d, numeros.Number):
                raise TypeError(f"'demandas' inválida no índice {idx}: esperado numérico, obteve {type(d)}")
            if d < 0:
                raise ValueError(f"'demandas' inválida no índice {idx}: demanda deve ser >= 0, obteve {d}")

        # Tempos de serviço: numéricos e não-negativos
        for idx, s in enumerate(self.tempos_servico):
            if not isinstance(s, numeros.Number):
                raise TypeError(f"'tempos_servico' inválido no índice {idx}: esperado numérico, obteve {type(s)}")
            if s < 0:
                raise ValueError(f"'tempos_servico' inválido no índice {idx}: deve ser >= 0, obteve {s}")

        # Janelas de tempo: pares (start, end) com start <= end
        for idx, tw in enumerate(self.janelas_tempo):
            if not (isinstance(tw, (list, tuple)) and len(tw) == 2):
                raise TypeError(f"'janelas_tempo' inválida no índice {idx}: esperado (inicio, fim), obteve {tw}")
            inicio, fim = tw
            if not (isinstance(inicio, numeros.Number) and isinstance(fim, numeros.Number)):
                raise TypeError(f"'janelas_tempo' inválida no índice {idx}: inicio/fim devem ser numéricos, obteve {type(inicio)}/{type(fim)}")
            if inicio < 0:
                raise ValueError(f"'janelas_tempo' inválida no índice {idx}: inicio < 0 ({inicio})")
            if inicio > fim:
                raise ValueError(f"'janelas_tempo' inválida no índice {idx}: inicio {inicio} > fim {fim}")

        # Capacidade do veículo: se fornecida, deve ser positiva e >= maior demanda
        if self.capacidade_veiculo is not None:
            if not isinstance(self.capacidade_veiculo, numeros.Number):
                raise TypeError(f"'capacidade_veiculo' inválida: esperado numérico, obteve {type(self.capacidade_veiculo)}")
            if self.capacidade_veiculo <= 0:
                raise ValueError(f"'capacidade_veiculo' inválida: deve ser > 0, obteve {self.capacidade_veiculo}")
            max_d = max(self.demandas) if self.demandas else 0
            if self.capacidade_veiculo < max_d:
                raise ValueError(f"'capacidade_veiculo' inválida: {self.capacidade_veiculo} < maior demanda {max_d}")

        # Validações da matriz de distâncias
        if self.matriz is None:
            if requer_matriz:
                raise ValueError("'matriz' é None: chame gerar_matriz_distancias_ficticia() ou carregue uma matriz antes de executar os algoritmos")
        else:
            if not hasattr(self.matriz, "ndim"):
                raise TypeError(f"Tipo inválido para 'matriz': esperado numpy.ndarray, obteve {type(self.matriz)}")
            try:
                import numpy as _np
            except Exception:
                _np = None

            if _np is None or not isinstance(self.matriz, _np.ndarray):
                raise TypeError(f"Tipo inválido para 'matriz': esperado numpy.ndarray, obteve {type(self.matriz)}")

            if self.matriz.ndim != 2:
                raise ValueError(f"Matriz inválida: esperado array 2D, obteve ndim={self.matriz.ndim}")
            linhas, colunas = self.matriz.shape
            if linhas != colunas:
                raise ValueError(f"Matriz inválida: esperado matriz quadrada, obteve shape ({linhas},{colunas})")
            if linhas != n:
                raise ValueError(f"Matriz inválida: esperado shape ({n},{n}) correspondente a 'posicoes', obteve ({linhas},{colunas})")

            # Checagens numéricas na matriz
            if not _np.isfinite(self.matriz).all():
                mask = ~_np.isfinite(self.matriz)
                idxs = list(zip(*_np.where(mask)))
                i0, j0 = idxs[0]
                raise ValueError(f"Matriz inválida: valor não finito em [{i0},{j0}] = {self.matriz[i0,j0]}")
            neg_mask = (self.matriz < 0)
            if neg_mask.any():
                idxs = list(zip(*_np.where(neg_mask)))
                i0, j0 = idxs[0]
                raise ValueError(f"Matriz inválida: distância negativa em [{i0},{j0}] = {self.matriz[i0,j0]}")

            # Diagonal próxima de zero
            diag_abs = _np.abs(_np.diag(self.matriz))
            diag_ruim = _np.where(diag_abs > epsilon)[0]
            if diag_ruim.size > 0:
                i0 = int(diag_ruim[0])
                raise ValueError(f"Matriz inválida: elemento diagonal [{i0},{i0}] = {self.matriz[i0,i0]} fora da tolerância {epsilon} em relação a 0")

        # Todas as validações passaram
        return True

