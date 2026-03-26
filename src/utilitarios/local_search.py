from modelos.solucao import Solucao


def two_opt_intra(solucao, instancia, config, max_passes=1):
    """
    Aplica 2-opt intra-rota como pós-processamento leve.

    Para cada rota, testa reversões de segmento e aceita as que reduzem
    a distância total. Preserva viabilidade de capacidade pois não move
    clientes entre rotas.
    """
    rotas = [r[:] for r in solucao.rotas]
    matriz = instancia.matriz
    melhorou = True
    passes = 0

    while melhorou and passes < max_passes:
        melhorou = False
        passes += 1
        for idx, rota in enumerate(rotas):
            if len(rota) < 4:
                continue
            for i in range(1, len(rota) - 2):
                for j in range(i + 1, len(rota) - 1):
                    a, b = rota[i - 1], rota[i]
                    c, d = rota[j], rota[j + 1]
                    antes = matriz[a][b] + matriz[c][d]
                    depois = matriz[a][c] + matriz[b][d]
                    if depois < antes - 1e-9:
                        rotas[idx] = rota[:i] + rota[i:j + 1][::-1] + rota[j + 1:]
                        rota = rotas[idx]
                        melhorou = True

    nova = Solucao(rotas=rotas, instancia=instancia)
    nova.avaliar(instancia, config)
    return nova
