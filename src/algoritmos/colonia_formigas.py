import random

def resolver(dados):
    print("\n Iniciando Colônia de Formigas...")
    pedidos = list(dados["id_pedido"])
    rota = random.sample(pedidos, len(pedidos))  # embaralha pedidos
    print(f"Rota sugerida (aleatória): {rota}")

    return rota
