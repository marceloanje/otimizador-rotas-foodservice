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

    def __init__(self, df, posicoes, demandas=None, capacidade_caminhao=None,
                 numero_caminhoes=None, carga_minima=0):
        self.df = df
        self.posicoes = posicoes
        self.matriz = None
        self.janelas_tempo  = None  # list[tuple[int,int]] — (inicio, fim) por nó, em minutos
        self.tempos_servico = None  # list[int] — tempo de atendimento por nó, em minutos
        self.matriz_tempos  = None  # np.ndarray n×n — tempos de viagem em minutos

        n = len(posicoes)
        self.demandas = list(demandas) if demandas is not None else [0.0] * n
        self.capacidade_caminhao = capacidade_caminhao
        self.n_clientes = max(0, n - 1)
        self.numero_caminhoes = numero_caminhoes
        self.carga_minima = float(carga_minima)

    @classmethod
    def do_csv(cls, path, capacidade_caminhao=None, numero_caminhoes=None):
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
        from config import DEPOSITO_LAT, DEPOSITO_LON

        from geoprocessamento.preprocessamento import limpar_pedidos
        df = limpar_pedidos(
            path,
            path_saida="src/dados/dados_processados/pedidos_limpos.csv",
        )

        # Verificar antecipadamente se a capacidade comporta a maior demanda individual
        if capacidade_caminhao is None:
            raise ValueError("capacidade_caminhao é obrigatório. Defina-o em config_experimento.py.")
        capacidade = capacidade_caminhao
        if "valor_total" in df.columns:
            max_demanda = df["valor_total"].fillna(0).astype(float).max()
            if max_demanda > capacidade:
                raise ValueError(
                    f"Instância infactível: maior demanda individual ({max_demanda:.2f}) "
                    f"excede capacidade do caminhão ({capacidade}). "
                    f"Ajuste 'capacidade_caminhao' em config_experimento.py."
                )

        # Geocodificar se lat/lon ainda não estiverem presentes
        if "lat" not in df.columns or "lon" not in df.columns:
            from geoprocessamento.geocodificador import geocodificar_dataframe
            print("Coordenadas não encontradas no CSV. Iniciando geocodificação...")
            df = geocodificar_dataframe(df)

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

        # Gerar janelas de tempo sintéticas (idempotente se colunas já existirem)
        from geoprocessamento.preprocessamento import gerar_janelas_tempo
        df = gerar_janelas_tempo(df)

        posicoes = list(zip(df["lat"].astype(float), df["lon"].astype(float)))

        # Tentar obter matriz real via OSRM
        matriz_tempos_osrm = None
        try:
            from geoprocessamento.integracao_osrm import obter_matriz_osrm
            print("Calculando matrizes de distâncias e tempos via OSRM...")
            matriz_real, matriz_tempos_osrm = obter_matriz_osrm(posicoes)
            print(f"Matrizes OSRM obtidas: shape {matriz_real.shape}, distâncias em metros, tempos em minutos.")
        except Exception as e:
            print(f"AVISO: OSRM indisponível ({e}). Usando matriz euclidiana como fallback.")
            matriz_real = None

        # Demandas: coluna valor_total; depósito (índice 0) recebe 0.0
        if "valor_total" in df.columns:
            demandas = df["valor_total"].fillna(0).astype(float).tolist()
        else:
            demandas = [0.0] * len(df)
        demandas[0] = 0.0  # garante que o depósito tem demanda zero

        if numero_caminhoes is None:
            raise ValueError("numero_caminhoes é obrigatório. Defina-o em config_experimento.py.")
        carga_minima = int(capacidade * 0.10)

        instancia = cls(
            df=df,
            posicoes=posicoes,
            demandas=demandas,
            capacidade_caminhao=capacidade,
            numero_caminhoes=numero_caminhoes,
            carga_minima=carga_minima,
        )

        if matriz_real is not None:
            instancia.matriz = matriz_real
            if matriz_tempos_osrm is not None:
                instancia.matriz_tempos = matriz_tempos_osrm
            else:
                # OSRM retornou apenas distâncias — estimar tempos a 40 km/h
                instancia.matriz_tempos = matriz_real / (40_000 / 60)  # m → min
        else:
            instancia.gerar_matriz_distancias_ficticia()
            instancia.matriz_tempos = instancia.gerar_matriz_tempos_ficticia()

        instancia.janelas_tempo  = list(zip(
            df["janela_inicio"].astype(int), df["janela_fim"].astype(int)
        ))
        instancia.tempos_servico = df["tempo_servico"].astype(int).tolist()

        instancia.validar()
        return instancia

    def gerar_matriz_tempos_ficticia(self):
        """Proxy de tempos a partir da matriz euclidiana (graus → minutos a 40 km/h em BH).

        1 grau ≈ 111 km; 111 km / 40 km/h × 60 min/h = 166,5 min/grau.
        Deve ser chamado após gerar_matriz_distancias_ficticia().
        """
        return self.matriz * 166.5

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

    def verificar_factibilidade(self):
        """Verifica se o problema é matematicamente factível com a frota disponível.

        Levanta ValueError se a demanda total exceder a capacidade total da frota.
        """
        if self.capacidade_caminhao is None or self.numero_caminhoes is None:
            return
        demanda_total = sum(self.demandas[1:])  # exclui depósito (índice 0)
        capacidade_total = self.capacidade_caminhao * self.numero_caminhoes
        if demanda_total > capacidade_total:
            raise ValueError(
                f"Instância infactível: demanda total ({demanda_total:.1f}) "
                f"excede capacidade total da frota "
                f"({self.numero_caminhoes} caminhões × {self.capacidade_caminhao} = {capacidade_total:.1f}). "
                f"Ajuste 'numero_caminhoes' ou 'capacidade_caminhao' em config_experimento.py."
            )

    def validar(self):
        """Valida consistência básica da instância (CVRP).

        Levanta exceções quando tamanhos divergirem, demandas forem inválidas
        ou quando a matriz (se requerida) não estiver presente.
        Também verifica factibilidade da frota.
        """
        self._validar(requer_matriz=False)
        self.verificar_factibilidade()

    def _validar(self, requer_matriz: bool = False, epsilon: float = 1e-8):

        import numbers as numeros

        n = len(self.posicoes)

        # Pelo menos um cliente além do depósito
        if self.n_clientes == 0:
            raise ValueError(
                "Instância inválida: nenhum cliente encontrado após o processamento do CSV. "
                "Verifique o arquivo de dados (separador, colunas obrigatórias e valores de demanda)."
            )

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

