from modelos.representacao import Representacao
from modelos.objetivo_config import ObjetivoConfig


class Solucao:
    def __init__(self, rotas=None, custo=None, instancia=None):
        # rotas: lista de rotas, cada rota é uma lista de nós (ex: [0,1,2,0])
        self.rotas = rotas or []
        self.custo = custo
        self.n_veiculos = len(self.rotas)
        self.tempo_computacional = None
        self.violacoes = {}
        self.meta = {}
        self.instancia = instancia
        # Atributos para função objetivo
        self.custo_objetivo = None
        self.objetivo_config = None

    def adicionar_rota(self, rota):
        self.rotas.append(rota)
        self.n_veiculos = len(self.rotas)

    def calcular_custo(self, instancia=None):
        inst = instancia or self.instancia
        if inst is None:
            raise ValueError("Instancia (com matriz) necessária para calcular custo")
        matriz = getattr(inst, "matriz", None)
        if matriz is None:
            raise ValueError("Instancia não possui matriz de distâncias")

        total = 0.0
        for rota in self.rotas:
            total += Representacao(rota).custo(matriz)

        self.custo = total
        return total

    def verificar_capacidade(self, instancia=None):
        """
        Verifica violações de capacidade em cada rota.
        
        Parameters
        ----------
        instancia : Instancia | None
            Instância com demandas e capacidade. Se None, usa self.instancia.
            
        Returns
        -------
        dict
            Dicionário {rota_idx: excesso_de_carga} com violações detalhadas.
            Retorna dicionário vazio se todas as rotas são válidas.
        """
        inst = instancia or self.instancia
        if inst is None:
            raise ValueError("Instancia necessária para checar capacidade")
        
        capacidade = getattr(inst, "capacidade_caminhao", None)
        demandas = getattr(inst, "demandas", getattr(inst, "demands", None))
        
        if capacidade is None or demandas is None:
            return {}
        
        violacoes_detalhadas = {}
        
        for idx, rota in enumerate(self.rotas):
            carga = 0.0
            for node in rota:
                # Assumir nó 0 como depósito
                if node == 0:
                    continue
                carga += float(demandas[node])
            
            excesso = carga - capacidade
            if excesso > 0:
                violacoes_detalhadas[idx] = excesso
        
        return violacoes_detalhadas

    def eh_valida(self, instancia=None):
        """
        Verifica se a solução é válida (sem violações de capacidade).
        
        Parameters
        ----------
        instancia : Instancia | None
            Instância para verificação. Se None, usa self.instancia.
            
        Returns
        -------
        bool
            True se a solução não viola restrições, False caso contrário.
        """
        violacoes_capacidade = self.verificar_capacidade(instancia)
        n_violacoes = len(violacoes_capacidade)
        
        self.violacoes = {"capacidade": n_violacoes}
        return n_violacoes == 0
    
    def avaliar(self, instancia=None, config=None):
        """
        Calcula função objetivo completa: custo de distância + penalizações.
        
        Esta é a função objetivo unificada que deve ser usada por todos os
        algoritmos de otimização. Combina:
        - Distância total percorrida (soma de todas as rotas)
        - Penalizações por violações de capacidade
        - Penalizações por número de veículos (se configurado)
        
        Parameters
        ----------
        instancia : Instancia | None
            Instância com matriz, demandas e capacidade. Se None, usa self.instancia.
        config : ObjetivoConfig | None
            Configuração de penalizações. Se None, cria config padrão.
            
        Returns
        -------
        float
            Valor da função objetivo (custo_distancia + penalizações).
            
        Notes
        -----
        Após execução, atualiza:
        - self.custo: distância total (sem penalizações)
        - self.custo_objetivo: valor completo com penalizações
        - self.violacoes: dicionário detalhado de violações
        """
        inst = instancia or self.instancia
        if inst is None:
            raise ValueError("Instancia necessária para avaliar")
        
        # Calcular custo de distância
        custo_distancia = self.calcular_custo(inst)
        
        # Configuração de penalizações
        if config is None:
            matriz = getattr(inst, "matriz", None)
            config = ObjetivoConfig(matriz=matriz)
        self.objetivo_config = config
        
        # Verificar violações de capacidade
        violacoes_capacidade = self.verificar_capacidade(inst)
        
        # Calcular penalizações
        penalidade_capacidade = config.calcular_penalidade_capacidade(violacoes_capacidade)
        penalidade_veiculos = config.calcular_penalidade_veiculos(self.n_veiculos)
        
        # Função objetivo completa
        custo_objetivo = custo_distancia + penalidade_capacidade + penalidade_veiculos
        
        # Armazenar resultados
        self.custo_objetivo = custo_objetivo
        self.violacoes = {
            "capacidade": len(violacoes_capacidade),
            "capacidade_detalhes": violacoes_capacidade,
            "penalidade_capacidade": penalidade_capacidade,
            "penalidade_veiculos": penalidade_veiculos
        }
        
        return custo_objetivo

    def __repr__(self):
        repr_str = f"Solucao(n_rotas={self.n_veiculos}, custo={self.custo}"
        if self.custo_objetivo is not None:
            repr_str += f", custo_objetivo={self.custo_objetivo:.2f}"
        repr_str += f", violacoes={self.violacoes})"
        return repr_str
