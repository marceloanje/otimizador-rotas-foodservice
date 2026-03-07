"""
Configuração para função objetivo do CVRP.

Define pesos de penalização e estratégias para avaliar soluções
viáveis e inviáveis durante a busca metaheurística.
"""


class ObjetivoConfig:
    """Configuração de função objetivo com penalizações."""
    
    # Estratégias de penalização disponíveis
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
        Inicializa configuração de função objetivo.
        
        Parameters
        ----------
        peso_capacidade : float | None
            Peso da penalização por violação de capacidade.
            Se None, calcula automaticamente como max(matriz) * 10.
        peso_veiculos : float
            Peso para penalizar número de veículos (padrão: 0.0 = não penaliza).
        estrategia : str
            Tipo de penalização: "fixa" ou "proporcional" (padrão).
            - fixa: penaliza peso_capacidade por rota violada
            - proporcional: penaliza peso_capacidade * excesso_carga
        matriz : numpy.ndarray | None
            Matriz de distâncias para calcular peso automático.
        """
        self.estrategia = estrategia
        self.peso_veiculos = peso_veiculos
        
        # Calcular peso automático se não fornecido
        if peso_capacidade is None:
            if matriz is not None:
                max_dist = matriz.max()
                self.peso_capacidade = max_dist * 10.0
            else:
                # Valor padrão conservador
                self.peso_capacidade = 1000.0
        else:
            self.peso_capacidade = peso_capacidade
    
    def calcular_penalidade_capacidade(self, violacoes_capacidade):
        """
        Calcula penalidade total por violações de capacidade.
        
        Parameters
        ----------
        violacoes_capacidade : dict
            Dicionário {rota_idx: excesso_de_carga}.
            
        Returns
        -------
        float
            Valor total da penalidade.
        """
        if not violacoes_capacidade:
            return 0.0
        
        if self.estrategia == self.PENALIZACAO_FIXA:
            # Penalidade fixa por rota violada
            return self.peso_capacidade * len(violacoes_capacidade)
        
        elif self.estrategia == self.PENALIZACAO_PROPORCIONAL:
            # Penalidade proporcional ao excesso de carga
            total_excesso = sum(violacoes_capacidade.values())
            return self.peso_capacidade * total_excesso
        
        else:
            raise ValueError(f"Estratégia desconhecida: {self.estrategia}")
    
    def calcular_penalidade_veiculos(self, n_veiculos):
        """
        Calcula penalidade por número de veículos.
        
        Parameters
        ----------
        n_veiculos : int
            Número de veículos utilizados na solução.
            
        Returns
        -------
        float
            Valor da penalidade.
        """
        return self.peso_veiculos * n_veiculos
    
    def __repr__(self):
        return (
            f"ObjetivoConfig("
            f"peso_capacidade={self.peso_capacidade:.2f}, "
            f"peso_veiculos={self.peso_veiculos:.2f}, "
            f"estrategia='{self.estrategia}')"
        )
