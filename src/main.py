from algoritmos.colonia_formigas import ACO
from modelos.instancia import Instancia

def main():
    print("Otimizador de Rotas - Protótipo Inicial")

    # Carregar dados ficticios
    instancia = Instancia.from_csv("src/dados/pedidos.csv")
    instancia.gerar_matriz_distancias_ficticia()

    # Rodar versão inicial da colônia de formigas
    solver = ACO(instancia)
    solucao = solver.run()

    print("Rota:", solucao.rota)
    print("Custo:", solucao.custo)

if __name__ == "__main__":
    main()
