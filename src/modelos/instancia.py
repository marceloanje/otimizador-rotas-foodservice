import pandas as pd
import numpy as np

class Instancia:
    """
    Attributes
    ----------
    df : pandas.DataFrame
        DataFrame cru lido a partir do CSV de pedidos (inclui depósito na linha 0).
    posicoes : list[tuple]
        Lista de tuplas (lat, lon) para cada nó, índice 0 é o depósito.
    matriz : numpy.ndarray | None
        Matriz de distâncias (n x n). Pode ser gerada ficticiamente ou carregada.
    demandas : list[float]
        Demanda de cada nó (unidades). Índice 0 corresponde ao depósito e normalmente tem demanda 0.
    capacidade_caminhao : int | None
        Capacidade (unidades) dos caminhões (fro­ta homogênea quando fornecido).
    n_clientes : int
        Número de clientes (excluindo depósito).
    """

    def __init__(self, df, posicoes, demandas=None, capacidade_caminhao=None):
        self.df = df
        self.posicoes = posicoes
        self.matriz = None

        n = len(posicoes)
        self.demandas = list(demandas) if demandas is not None else [0.0] * n
        self.capacidade_caminhao = capacidade_caminhao
        self.n_clientes = max(0, n - 1)

    @classmethod
    def do_csv(cls, path, capacidade_caminhao=None):
        """Cria uma `Instancia` a partir do CSV de pedidos (CVRP).

        O depósito (nó 0) é inserido automaticamente a partir das constantes
        `DEPOSITO_LAT` e `DEPOSITO_LON` definidas em `config.py` — ele não deve
        constar no CSV.

        Se o CSV não possuir colunas `lat` e `lon`, o geocodificador é acionado
        automaticamente para cada endereço. Resultados são armazenados em cache
        local (`src/dados/cache_geocodificacao.json`) para evitar requisições
        repetidas.

        Parâmetros
        ----------
        path : str
            Caminho para o CSV de pedidos.
        capacidade_caminhao : int | None
            Capacidade por caminhão; se None usa o valor padrão de `config.py`.
        """
        from config import CAPACIDADE_CAMINHAO, DEPOSITO_LAT, DEPOSITO_LON

        from geoprocessamento.preprocessamento import limpar_pedidos
        df = limpar_pedidos(
            path,
            path_saida="src/dados/dados_processados/pedidos_limpos.csv",
        )

        # Geocodificar se lat/lon ainda não estiverem presentes
        if "lat" not in df.columns or "lon" not in df.columns:
            from geoprocessamento.geocodificador import geocodificar_dataframe
            print("Coordenadas não encontradas no CSV. Iniciando geocodificação...")
            df = geocodificar_dataframe(df)
            # Persistir lat/lon no CSV para evitar regeocificação nas próximas execuções
            df.to_csv(path, index=False)
            print(f"Coordenadas salvas em {path}.")

        # Descartar linhas onde a geocodificação falhou
        n_antes = len(df)
        df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        n_descartados = n_antes - len(df)
        if n_descartados > 0:
            print(f"AVISO: {n_descartados} linha(s) descartada(s) por falha na geocodificação.")

        # Prepender depósito como nó 0
        deposito_row = {col: None for col in df.columns}
        deposito_row["lat"] = DEPOSITO_LAT
        deposito_row["lon"] = DEPOSITO_LON
        if "valor_total" in deposito_row:
            deposito_row["valor_total"] = 0.0
        df = pd.concat([pd.DataFrame([deposito_row]), df], ignore_index=True)

        posicoes = list(zip(df["lat"].astype(float), df["lon"].astype(float)))

        # Tentar obter matriz real via OSRM
        try:
            from geoprocessamento.integracao_osrm import obter_matriz_osrm
            print("Calculando matriz de distâncias via OSRM...")
            matriz_real = obter_matriz_osrm(posicoes)
            print(f"Matriz OSRM obtida: shape {matriz_real.shape}, unidade: metros.")
        except Exception as e:
            print(f"AVISO: OSRM indisponível ({e}). Usando matriz euclidiana como fallback.")
            matriz_real = None

        # Demandas: coluna valor_total; depósito (índice 0) recebe 0.0
        if "valor_total" in df.columns:
            demandas = df["valor_total"].fillna(0).astype(float).tolist()
        else:
            demandas = [0.0] * len(df)
        demandas[0] = 0.0  # garante que o depósito tem demanda zero

        capacidade = capacidade_caminhao if capacidade_caminhao is not None else CAPACIDADE_CAMINHAO

        instancia = cls(
            df=df,
            posicoes=posicoes,
            demandas=demandas,
            capacidade_caminhao=capacidade,
        )

        if matriz_real is not None:
            instancia.matriz = matriz_real
        else:
            instancia.gerar_matriz_distancias_ficticia()

        instancia.validar()
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

    def validar(self):
        """Valida consistência básica da instância (CVRP).

        Levanta exceções quando tamanhos divergirem, demandas forem inválidas
        ou quando a matriz (se requerida) não estiver presente.
        """
        self._validar(requer_matriz=False)

    def _validar(self, requer_matriz: bool = False, epsilon: float = 1e-8):

        import numbers as numeros

        n = len(self.posicoes)

        # Consistência de tamanhos
        if not hasattr(self.posicoes, "__len__"):
            raise TypeError(f"Inválido 'posicoes': esperada uma sequência de posições, obteve {type(self.posicoes)}")

        if len(self.demandas) != n:
            raise ValueError(f"Comprimento inválido em 'demandas': esperado {n} (igual a 'posicoes'), obtido {len(self.demandas)}")

        # Demandas: numéricas e não-negativas
        for idx, d in enumerate(self.demandas):
            if not isinstance(d, numeros.Number):
                raise TypeError(f"'demandas' inválida no índice {idx}: esperado numérico, obteve {type(d)}")
            if d < 0:
                raise ValueError(f"'demandas' inválida no índice {idx}: demanda deve ser >= 0, obteve {d}")

        # Capacidade do caminhão: se fornecida, deve ser positiva e >= maior demanda
        if self.capacidade_caminhao is not None:
            if not isinstance(self.capacidade_caminhao, numeros.Number):
                raise TypeError(f"'capacidade_caminhao' inválida: esperado numérico, obteve {type(self.capacidade_caminhao)}")
            if self.capacidade_caminhao <= 0:
                raise ValueError(f"'capacidade_caminhao' inválida: deve ser > 0, obteve {self.capacidade_caminhao}")
            max_d = max(self.demandas) if self.demandas else 0
            if self.capacidade_caminhao < max_d:
                raise ValueError(f"'capacidade_caminhao' inválida: {self.capacidade_caminhao} < maior demanda {max_d}")

        # Validações da matriz de distâncias
        if self.matriz is None:
            if requer_matriz:
                raise ValueError("'matriz' é None: gere ou carregue uma matriz antes de executar os algoritmos")
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

        return True

