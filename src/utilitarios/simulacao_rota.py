"""Simulação da cronologia de uma rota VRPTW.

Emite a timeline completa por nó: chegada, início de serviço, saída, atraso e espera.
Espelha a lógica de Solucao.verificar_janelas_tempo mas com saída estruturada,
usada pelo diagrama de Gantt e análises de janela de tempo.
"""


def simular_rota(rota, instancia):
    """Reproduz a cronologia de uma rota dado instancia com dados VRPTW.

    Parameters
    ----------
    rota : list[int]
        Lista de nós incluindo depósito no início e fim (ex: [0, 3, 1, 5, 0]).
    instancia : Instancia
        Deve conter matriz_tempos, janelas_tempo e tempos_servico.

    Returns
    -------
    list[dict]
        Lista de dicts por nó (excluindo depósito), com as chaves:
        no, t_chegada, t_inicio_servico, t_saida, atraso, espera,
        t_servico, carga_acumulada, janela_ini, janela_fim.
        Retorna [] se os dados VRPTW não estiverem disponíveis.
    """
    matriz_tempos = getattr(instancia, "matriz_tempos", None)
    janelas_tempo = getattr(instancia, "janelas_tempo", None)
    tempos_servico = getattr(instancia, "tempos_servico", None)
    demandas = getattr(instancia, "demandas", None)

    if any(x is None for x in [matriz_tempos, janelas_tempo, tempos_servico]):
        return []

    timeline = []
    tempo_atual = 0.0
    pos_ant = 0
    carga_acumulada = 0.0

    for node in rota:
        if node == 0:
            continue

        t_chegada = tempo_atual + float(matriz_tempos[pos_ant][node])
        ini, fim = janelas_tempo[node]

        if t_chegada < ini:
            t_inicio_servico = ini
            espera = ini - t_chegada
            atraso = 0.0
        elif t_chegada <= fim:
            t_inicio_servico = t_chegada
            espera = 0.0
            atraso = 0.0
        else:
            t_inicio_servico = t_chegada
            espera = 0.0
            atraso = t_chegada - fim

        t_servico_dur = float(tempos_servico[node])
        t_saida = t_inicio_servico + t_servico_dur

        if demandas is not None:
            carga_acumulada += float(demandas[node])

        timeline.append({
            "no": node,
            "t_chegada": t_chegada,
            "t_inicio_servico": t_inicio_servico,
            "t_saida": t_saida,
            "atraso": atraso,
            "espera": espera,
            "t_servico": t_servico_dur,
            "carga_acumulada": carga_acumulada,
            "janela_ini": ini,
            "janela_fim": fim,
        })

        tempo_atual = t_saida
        pos_ant = node

    return timeline
