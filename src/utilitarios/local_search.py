from modelos.solucao import Solucao


def _rota_tw_ok(rota, instancia):
    """True se a rota respeita todas as janelas de tempo (ou se VRPTW inativo)."""
    matriz_tempos = getattr(instancia, "matriz_tempos", None)
    janelas = getattr(instancia, "janelas_tempo", None)
    tempos_servico = getattr(instancia, "tempos_servico", None)
    if any(x is None for x in [matriz_tempos, janelas, tempos_servico]):
        return True

    tempo_atual = 0.0
    pos_ant = 0
    for node in rota:
        if node == 0:
            continue
        t_chegada = tempo_atual + float(matriz_tempos[pos_ant][node])
        ini, fim = janelas[node]
        if t_chegada > fim:
            return False
        tempo_atual = max(t_chegada, ini) + float(tempos_servico[node])
        pos_ant = node
    return True


def _carga_rota(rota, demandas):
    return sum(float(demandas[n]) for n in rota if n != 0)


def two_opt_intra(solucao, instancia, config, max_passes=1):
    """
    Aplica 2-opt intra-rota como pós-processamento.

    Testa reversões de segmento e aceita as que reduzem distância. Se
    houver janelas de tempo (VRPTW), rejeita reversões que introduzem
    violações que não existiam na rota original.
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
            tw_ok_antes = _rota_tw_ok(rota, instancia)
            for i in range(1, len(rota) - 2):
                for j in range(i + 1, len(rota) - 1):
                    a, b = rota[i - 1], rota[i]
                    c, d = rota[j], rota[j + 1]
                    antes = matriz[a][b] + matriz[c][d]
                    depois = matriz[a][c] + matriz[b][d]
                    if depois < antes - 1e-9:
                        nova = rota[:i] + rota[i:j + 1][::-1] + rota[j + 1:]
                        # Em VRPTW, só aceita se não piora a viabilidade temporal.
                        if tw_ok_antes and not _rota_tw_ok(nova, instancia):
                            continue
                        rotas[idx] = nova
                        rota = nova
                        tw_ok_antes = _rota_tw_ok(rota, instancia)
                        melhorou = True

    nova_sol = Solucao(rotas=rotas, instancia=instancia)
    nova_sol.avaliar(instancia, config)
    return nova_sol


def relocate_inter(solucao, instancia, config, max_passes=1):
    """Relocate inter-rota: move um cliente da rota A para a rota B se reduzir o
    objetivo e respeitar capacidade e janelas de tempo.
    """
    rotas = [r[:] for r in solucao.rotas]
    demandas = instancia.demandas
    capacidade = instancia.capacidade_caminhao

    def avaliar(rotas_temp):
        s = Solucao(rotas=rotas_temp, instancia=instancia)
        s.avaliar(instancia, config)
        return s.custo_objetivo

    custo_atual = avaliar(rotas)
    melhorou = True
    passes = 0

    while melhorou and passes < max_passes:
        melhorou = False
        passes += 1
        n_rotas = len(rotas)
        for io in range(n_rotas):
            for pc in range(1, len(rotas[io]) - 1):
                cliente = rotas[io][pc]
                demanda_c = float(demandas[cliente])
                for id_ in range(n_rotas):
                    if id_ == io:
                        continue
                    carga_destino = _carga_rota(rotas[id_], demandas)
                    if carga_destino + demanda_c > capacidade:
                        continue
                    for pi in range(1, len(rotas[id_])):
                        novas = [r[:] for r in rotas]
                        novas[io] = novas[io][:pc] + novas[io][pc + 1:]
                        if len(novas[io]) <= 2:  # rota ficaria vazia
                            continue
                        novas[id_] = novas[id_][:pi] + [cliente] + novas[id_][pi:]
                        if not (_rota_tw_ok(novas[io], instancia) and _rota_tw_ok(novas[id_], instancia)):
                            continue
                        novo_custo = avaliar(novas)
                        if novo_custo < custo_atual - 1e-9:
                            rotas = novas
                            custo_atual = novo_custo
                            melhorou = True
                            break
                    if melhorou:
                        break
                if melhorou:
                    break
            if melhorou:
                break

    nova_sol = Solucao(rotas=rotas, instancia=instancia)
    nova_sol.avaliar(instancia, config)
    return nova_sol


def busca_local(solucao, instancia, config, passes=2):
    """Pipeline de busca local: 2-opt intra + relocate inter, alternados."""
    sol = solucao
    for _ in range(passes):
        sol = two_opt_intra(sol, instancia, config, max_passes=2)
        sol = relocate_inter(sol, instancia, config, max_passes=1)
    return sol
