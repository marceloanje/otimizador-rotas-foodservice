"""
Configuração para função objetivo do CVRP.

Define pesos de penalização e estratégias para avaliar soluções
viáveis e inviáveis durante a busca metaheurística.
"""


class ObjetivoConfig:
    """Configuração de função objetivo com penalizações."""

    PENALIZACAO_FIXA = "fixa"
    PENALIZACAO_PROPORCIONAL = "proporcional"

    def __init__(
        self,
        peso_capacidade=None,
        peso_veiculos=0.0,
        estrategia=PENALIZACAO_PROPORCIONAL,
        matriz=None
    ):
        """
        Parâmetros
        ----------
        peso_capacidade : float | None
            Peso da penalização por violação de capacidade.
            Se None, calcula automaticamente como max(matriz) * 10.
        peso_veiculos : float
            Peso para penalizar número de veículos (padrão: 0.0 = não penaliza).
        estrategia : str
            "fixa" penaliza por rota violada; "proporcional" penaliza pelo excesso acumulado.
        matriz : numpy.ndarray | None
            Matriz de distâncias para calcular peso automático.
        """
        self.estrategia = estrategia
        self.peso_veiculos = peso_veiculos

        if peso_capacidade is None:
            if matriz is not None:
                self.peso_capacidade = matriz.max() * 10.0
            else:
                self.peso_capacidade = 1000.0
        else:
            self.peso_capacidade = peso_capacidade

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
        """Penalidade linear pelo número de veículos usados."""
        return self.peso_veiculos * n_veiculos

    def __repr__(self):
        return (
            f"ObjetivoConfig("
            f"peso_capacidade={self.peso_capacidade:.2f}, "
            f"peso_veiculos={self.peso_veiculos:.2f}, "
            f"estrategia='{self.estrategia}')"
        )
