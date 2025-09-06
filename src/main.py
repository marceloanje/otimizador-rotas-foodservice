import pandas as pd
from algoritmos import colonia_formigas

def main():
    print("Otimizador de Rotas - Protótipo Inicial")

    # Carregar dados de exemplo
    dados = pd.read_csv("src/dados/exemplo.csv")
    print("Dados de pedidos carregados:")
    print(dados.head())

    # Rodar versão inicial da colônia de formigas
    colonia_formigas.resolver(dados)

if __name__ == "__main__":
    main()
