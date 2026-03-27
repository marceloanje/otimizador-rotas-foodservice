"""
Configuração para função objetivo do CVRP.

Define pesos de penalização e estratégias para avaliar soluções
viáveis e inviáveis durante a busca metaheurística.

Penalidades implementadas
--------------------------
- Capacidade       : proporcional ao excesso de carga por rota
- Frota            : proporcional ao número de veículos ACIMA do limite
- Cobertura        : por cliente não visitado (calibrada para dominar todas as outras)
- Carga mínima     : proporcional ao déficit de carga em rotas quase vazias
- Desequilíbrio    : baseado no coeficiente de variação da carga entre rotas
"""


class ObjetivoConfig:
    """Configuração de função objetivo com penalizações."""

    PENALIZACAO_FIXA = "fixa"
    PENALIZACAO_PROPORCIONAL = "proporcional"

    def __init__(
        self,
        peso_capacidade=None,
        peso_veiculos=None,
        peso_cobertura=None,
        peso_carga_minima=None,
        peso_desequilibrio=None,
        peso_janela_tempo=None,
        limiar_desequilibrio=0.3,
        estrategia=PENALIZACAO_PROPORCIONAL,
        matriz=None,
        numero_caminhoes=None,
        n_clientes=None,
    ):
        """
        Parâmetros
        ----------
        peso_capacidade : float | None
            Peso da penalização por violação de capacidade.
            Se None, calcula automaticamente como max(matriz) * 10.
        peso_veiculos : float | None
            Peso por veículo EXCEDENTE acima do limite da frota.
            Se None, calcula automaticamente como max(matriz) * 5.
        peso_cobertura : float | None
            Peso por cliente não visitado.
            Se None, calcula como max(matriz) * 100 * n_clientes (domina todas as outras).
        peso_carga_minima : float | None
            Peso por unidade de déficit de carga mínima por rota.
            Se None, calcula como max(matriz) * 10.
        peso_desequilibrio : float | None
            Peso para penalizar o coeficiente de variação da carga entre rotas.
            Se None, calcula como max(matriz) * n_clientes.
        peso_janela_tempo : float | None
            Peso por minuto de atraso acumulado nas janelas de tempo (VRPTW).
            Se None, calcula como max(matriz) * 2.0.
        limiar_desequilibrio : float
            Coeficiente de variação mínimo para acionar penalidade (padrão: 0.3 = 30%).
        estrategia : str
            "fixa" penaliza por rota violada; "proporcional" penaliza pelo excesso acumulado.
        matriz : numpy.ndarray | None
            Matriz de distâncias para calcular pesos automáticos.
        numero_caminhoes : int | None
            Limite máximo de veículos; se None, não penaliza excesso de frota.
        n_clientes : int | None
            Número de clientes; usado para calibrar peso_cobertura e peso_desequilibrio.
        """
        self.estrategia = estrategia
        self.numero_caminhoes = numero_caminhoes
        self.limiar_desequilibrio = limiar_desequilibrio

        max_dist = float(matriz.max()) if matriz is not None else 1000.0
        nc = n_clientes if n_clientes is not None and n_clientes > 0 else 1

        self.peso_capacidade = (
            peso_capacidade if peso_capacidade is not None else max_dist * 10.0
        )
        self.peso_veiculos = (
            float(peso_veiculos) if peso_veiculos is not None else max_dist * 5.0
        )
        self.peso_cobertura = (
            float(peso_cobertura) if peso_cobertura is not None else max_dist * 100.0 * nc
        )
        self.peso_carga_minima = (
            float(peso_carga_minima) if peso_carga_minima is not None else max_dist * 10.0
        )
        self.peso_desequilibrio = (
            float(peso_desequilibrio) if peso_desequilibrio is not None else max_dist * float(nc)
        )
        self.peso_janela_tempo = (
            float(peso_janela_tempo) if peso_janela_tempo is not None else max_dist * 2.0
        )

    def calcular_penalidade_capacidade(self, violacoes_capacidade):
        """Penalidade total dado o dicionário {rota_idx: excesso}."""
        if not violacoes_capacidade:
            return 0.0

        if self.estrategia == self.PENALIZACAO_FIXA:
            return self.peso_capacidade * len(violacoes_capacidade)

        elif self.estrategia == self.PENALIZACAO_PROPORCIONAL:
            total_excesso = sum(violacoes_capacidade.values())
            return self.peso_capacidade * total_excesso

        else:
            raise ValueError(f"Estratégia desconhecida: {self.estrategia}")

    def calcular_penalidade_veiculos(self, n_veiculos):
        """Penalidade pelo excesso de veículos acima do limite da frota."""
        if self.numero_caminhoes is None:
            return 0.0
        excesso = max(0, n_veiculos - self.numero_caminhoes)
        return self.peso_veiculos * excesso

    def calcular_penalidade_cobertura(self, clientes_faltando):
        """Penalidade proporcional ao número de clientes não visitados."""
        if clientes_faltando <= 0:
            return 0.0
        return self.peso_cobertura * clientes_faltando

    def calcular_penalidade_carga_minima(self, deficits):
        """Penalidade para rotas com carga abaixo do mínimo.

        deficits : dict {rota_idx: deficit}
            Onde deficit = carga_minima - carga_rota > 0.
        """
        if not deficits:
            return 0.0
        if self.estrategia == self.PENALIZACAO_PROPORCIONAL:
            return self.peso_carga_minima * sum(deficits.values())
        else:  # fixa
            return self.peso_carga_minima * len(deficits)

    def calcular_penalidade_janela_tempo(self, violacoes_jt):
        """Penalidade proporcional ao atraso total acumulado (em minutos).

        violacoes_jt : dict {rota_idx: [(cliente, atraso_min), ...]}
        """
        if not violacoes_jt:
            return 0.0
        total_atraso = sum(
            atraso for lista in violacoes_jt.values() for (_, atraso) in lista
        )
        return self.peso_janela_tempo * total_atraso

    def calcular_penalidade_desequilibrio(self, excesso_cv):
        """Penalidade pelo coeficiente de variação da carga acima do limiar.

        excesso_cv : float
            max(0, coef_variacao - limiar_desequilibrio)
        """
        if excesso_cv <= 0.0:
            return 0.0
        return self.peso_desequilibrio * excesso_cv

    def __repr__(self):
        return (
            f"ObjetivoConfig("
            f"peso_capacidade={self.peso_capacidade:.2f}, "
            f"peso_veiculos={self.peso_veiculos:.2f}, "
            f"peso_cobertura={self.peso_cobertura:.2f}, "
            f"peso_carga_minima={self.peso_carga_minima:.2f}, "
            f"peso_desequilibrio={self.peso_desequilibrio:.2f}, "
            f"peso_janela_tempo={self.peso_janela_tempo:.2f}, "
            f"estrategia='{self.estrategia}')"
        )
