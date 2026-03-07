import pandas as pd
import numpy as np
import ast

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

        Lê as colunas `posicao` (ou `lat`/`lon`) e `valor_total` do CSV e usa
        `valor_total` como demanda de cada nó. A primeira linha do CSV é
        considerada o depósito (nó 0).

        Parâmetros
        ---------
        path : str
            Caminho para o CSV de pedidos.
        capacidade_caminhao : int | None
            Capacidade por caminhão; se None usa o valor padrão de `src/config.py`.
        """

        from config import CAPACIDADE_CAMINHAO

        # Ler CSV - pode ter problemas com colunas devido a vírgulas extras
        df = pd.read_csv(path)
        
        # Se a última coluna estiver vazia (NaN), remover
        if df.columns[-1] == '' or (isinstance(df.columns[-1], str) and df.columns[-1].strip() == ''):
            df = df.iloc[:, :-1]
        elif df[df.columns[-1]].isna().all():
            df = df.drop(columns=[df.columns[-1]])

        # Parse posições
        # Tentar encontrar a coluna com coordenadas
        posicao_col = None
        if "posicao" in df.columns:
            posicao_col = "posicao"
        elif "valor_total" in df.columns:
            # Verificar se valor_total tem tuplas de coordenadas (string com parênteses)
            sample = str(df["valor_total"].iloc[0])
            if sample.startswith('(') and ',' in sample:
                posicao_col = "valor_total"
        
        if posicao_col:
            lats = []
            lons = []
            for p in df[posicao_col]:
                # Skip NaN or empty values
                if pd.isna(p) or p == "" or str(p).strip() == "":
                    # Use dummy coordinates if missing
                    lats.append(0.0)
                    lons.append(0.0)
                else:
                    try:
                        lat, lon = ast.literal_eval(str(p))
                        lats.append(lat)
                        lons.append(lon)
                    except (ValueError, SyntaxError):
                        # If parsing fails, use dummy coordinates
                        lats.append(0.0)
                        lons.append(0.0)

            df["lat"] = lats
            df["lon"] = lons
        elif "lat" in df.columns and "lon" in df.columns:
            # Já tem lat/lon nas colunas
            pass
        else:
            raise ValueError("CSV deve ter coluna 'posicao' com coordenadas ou colunas 'lat' e 'lon'")


        posicoes = list(zip(df["lat"], df["lon"]))

        # Demandas: tentar diferentes colunas
        demandas = None
        if "valor_total" in df.columns:
            # Verificar se valor_total é numérico
            try:
                demandas = df["valor_total"].fillna(0).astype(float).tolist()
            except (ValueError, TypeError):
                # Se não for numérico, tentar codigo_cobranca
                pass
        
        if demandas is None and "codigo_cobranca" in df.columns:
            try:
                demandas = df["codigo_cobranca"].fillna(0).astype(float).tolist()
            except (ValueError, TypeError):
                pass
        
        if demandas is None:
            # Fallback: usar zeros
            demandas = [0.0] * len(df)

        capacidade = capacidade_caminhao if capacidade_caminhao is not None else CAPACIDADE_CAMINHAO

        instancia = cls(
            df=df,
            posicoes=posicoes,
            demandas=demandas,
            capacidade_caminhao=capacidade,
        )

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
        """Validação interna simplificada para CVRP.

        Remove checagens relacionadas a janelas de tempo e tempos de serviço.
        """

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

        # Todas as validações passaram
        return True

