"""Geração de gráficos para o experimento comparativo.

Funções de convergência (por iteração e por tempo), distribuição de custos,
análise estatística, estrutura de rotas (Gantt, capacidade) e comparações
agregadas (gap ao ótimo, número de veículos).
"""

import os
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _extrair_custos(historico):
    """Extrai array de custos de historico (float puro ou tuple (iter, custo, tempo))."""
    if not historico:
        return np.array([])
    if isinstance(historico[0], (tuple, list)):
        return np.array([item[1] for item in historico], dtype=float)
    return np.asarray(historico, dtype=float)


def _extrair_tempos(historico):
    """Extrai array de tempos acumulados de historico. None se formato antigo (float puro)."""
    if not historico:
        return None
    if isinstance(historico[0], (tuple, list)):
        return np.array([item[2] for item in historico], dtype=float)
    return None


def _interp_historico(historico, n_pontos=100):
    """Reamostra uma curva de convergência para n_pontos no eixo [0, 1]."""
    if not historico:
        return np.full(n_pontos, np.nan)
    hist = _extrair_custos(historico)
    if len(hist) == 1:
        return np.full(n_pontos, hist[0])
    x_original = np.linspace(0.0, 1.0, len(hist))
    x_target = np.linspace(0.0, 1.0, n_pontos)
    return np.interp(x_target, x_original, hist)


def _interp_historico_por_tempo(historico, t_grid):
    """Interpola custo(t) numa grade de tempo comum.

    Usa o primeiro custo para t < t_min e o último custo para t > t_max.
    """
    tempos = _extrair_tempos(historico)
    custos = _extrair_custos(historico)
    if tempos is None or len(tempos) == 0:
        return np.full(len(t_grid), np.nan)
    return np.interp(t_grid, tempos, custos, left=custos[0], right=custos[-1])


# ---------------------------------------------------------------------------
# A1 — Convergência por iteração (mantida, retrocompatível com formato antigo)
# ---------------------------------------------------------------------------

def plotar_convergencia(convergencia_por_alg, instancia_nome, output_path, n_pontos=100):
    """Mediana + banda IQR por algoritmo; eixo X = fração de iterações.

    Parameters
    ----------
    convergencia_por_alg : dict[str, list[list]]
        algoritmo -> lista de históricos por run (float ou tuplas).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(0.0, 1.0, n_pontos)

    for alg, historicos in convergencia_por_alg.items():
        if not historicos:
            continue
        curvas = np.vstack([_interp_historico(h, n_pontos) for h in historicos])
        mediana = np.nanmedian(curvas, axis=0)
        q1 = np.nanpercentile(curvas, 25, axis=0)
        q3 = np.nanpercentile(curvas, 75, axis=0)
        linha, = ax.plot(x, mediana, label=alg, linewidth=2)
        ax.fill_between(x, q1, q3, alpha=0.2, color=linha.get_color())

    ax.set_yscale("log")
    ax.set_xlabel("Fração de iterações")
    ax.set_ylabel("Melhor custo objetivo (escala log)")
    ax.set_title(f"Convergência — instância {instancia_nome}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Boxplot e tempo (originais, mantidos)
# ---------------------------------------------------------------------------

def plotar_boxplot_custos(runs_por_alg, instancia_nome, output_path):
    """Boxplot do custo_objetivo final por algoritmo.

    Parameters
    ----------
    runs_por_alg : dict[str, list[float]]
        algoritmo -> lista de custo_objetivo finais.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    nomes = list(runs_por_alg.keys())
    dados = [runs_por_alg[n] for n in nomes]
    ax.boxplot(dados, labels=nomes, showmeans=True)
    ax.set_ylabel("Custo objetivo")
    ax.set_title(f"Distribuição do custo final — instância {instancia_nome}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plotar_tempo_execucao(tempos_por_inst_alg, output_path):
    """Barras de tempo médio por algoritmo x instância.

    Para n_instancias >= 5 usa barras horizontais (legível com muitos nomes).

    Parameters
    ----------
    tempos_por_inst_alg : dict[str, dict[str, float]]
        {instancia: {algoritmo: tempo_medio}}.
    """
    instancias = list(tempos_por_inst_alg.keys())
    if not instancias:
        return
    algoritmos = sorted({a for d in tempos_por_inst_alg.values() for a in d})

    horizontal = len(instancias) >= 5
    pos = np.arange(len(instancias))
    largura = 0.8 / max(len(algoritmos), 1)

    if horizontal:
        fig, ax = plt.subplots(figsize=(9, max(5, len(instancias) * 0.7)))
        for i, alg in enumerate(algoritmos):
            valores = [tempos_por_inst_alg[inst].get(alg, 0.0) for inst in instancias]
            ax.barh(pos + i * largura - 0.4 + largura / 2, valores, largura, label=alg)
        ax.set_yticks(pos)
        ax.set_yticklabels(instancias)
        ax.invert_yaxis()
        ax.set_xlabel("Tempo médio de execução (s)")
        ax.grid(True, axis="x", alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(max(9, len(instancias) * 1.5), 5))
        for i, alg in enumerate(algoritmos):
            valores = [tempos_por_inst_alg[inst].get(alg, 0.0) for inst in instancias]
            ax.bar(pos + i * largura - 0.4 + largura / 2, valores, largura, label=alg)
        ax.set_xticks(pos)
        ax.set_xticklabels(instancias, rotation=20, ha="right")
        ax.set_ylabel("Tempo médio de execução (s)")
        ax.grid(True, axis="y", alpha=0.3)

    ax.set_title("Tempo médio por algoritmo e instância")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# A2 — Convergência por tempo real
# ---------------------------------------------------------------------------

def plotar_convergencia_tempo(convergencia_por_alg, instancia_nome, output_path, n_pontos=100):
    """Mediana + banda IQR por algoritmo; eixo X = tempo acumulado (s).

    Requer históricos no formato tupla (iter, custo, tempo_acum).
    Algoritmos sem timestamp são silenciosamente omitidos.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    t_max = 0.0
    for historicos in convergencia_por_alg.values():
        for h in historicos:
            t = _extrair_tempos(h)
            if t is not None and len(t) > 0:
                t_max = max(t_max, float(t[-1]))

    if t_max <= 0:
        plt.close(fig)
        return

    t_grid = np.linspace(0.0, t_max, n_pontos)

    for alg, historicos in convergencia_por_alg.items():
        curvas = []
        for h in historicos:
            if h and _extrair_tempos(h) is not None:
                c = _interp_historico_por_tempo(h, t_grid)
                if not np.all(np.isnan(c)):
                    curvas.append(c)
        if not curvas:
            continue
        curvas = np.vstack(curvas)
        mediana = np.nanmedian(curvas, axis=0)
        q1 = np.nanpercentile(curvas, 25, axis=0)
        q3 = np.nanpercentile(curvas, 75, axis=0)
        linha, = ax.plot(t_grid, mediana, label=alg, linewidth=2)
        ax.fill_between(t_grid, q1, q3, alpha=0.2, color=linha.get_color())

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Melhor custo objetivo")
    ax.set_title(f"Convergência por tempo — instância {instancia_nome}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# A3 — Time-to-target (ECDF)
# ---------------------------------------------------------------------------

def plotar_time_to_target(convergencia_por_alg, alvo_custo, instancia_nome, output_path):
    """ECDF de fração de runs que atingiu custo ≤ alvo_custo ao longo do tempo.

    Parameters
    ----------
    alvo_custo : float
        Custo-alvo (ex: 1.05 * melhor custo global).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    tem_dados = False

    for alg, historicos in convergencia_por_alg.items():
        n_total = len(historicos)
        times_to_target = []

        for h in historicos:
            tempos = _extrair_tempos(h)
            custos = _extrair_custos(h)
            if tempos is None or len(custos) == 0:
                continue
            reached = np.where(custos <= alvo_custo)[0]
            if len(reached) > 0:
                times_to_target.append(float(tempos[reached[0]]))

        if not times_to_target:
            continue

        times_sorted = np.sort(times_to_target)
        frac = np.arange(1, len(times_sorted) + 1) / n_total
        ax.step(times_sorted, frac, where="post", label=alg, linewidth=2)
        tem_dados = True

    if not tem_dados:
        plt.close(fig)
        return

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Fração de runs que atingiram o alvo")
    ax.set_title(f"Time-to-target (alvo ≤ {alvo_custo:.0f}) — instância {instancia_nome}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B1 — Violino + box
# ---------------------------------------------------------------------------

def plotar_violino_custos(runs_por_alg, instancia_nome, output_path):
    """Violino com box interior mostrando distribuição completa do custo_objetivo.

    Parameters
    ----------
    runs_por_alg : dict[str, list[float]]
        algoritmo -> lista de custo_objetivo finais.
    """
    registros = [
        {"algoritmo": alg, "custo": v}
        for alg, vals in runs_por_alg.items()
        for v in vals
    ]
    if not registros:
        return

    df = pd.DataFrame(registros)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x="algoritmo", y="custo", inner="box", ax=ax)
    ax.set_xlabel("Algoritmo")
    ax.set_ylabel("Custo objetivo")
    ax.set_title(f"Distribuição do custo final (violino) — instância {instancia_nome}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B2 — Scatter custo × tempo
# ---------------------------------------------------------------------------

def plotar_scatter_custo_tempo(tempos_por_alg, custos_por_alg, instancia_nome, output_path):
    """Scatter de (tempo_run, custo_objetivo) por algoritmo — Pareto visual.

    Parameters
    ----------
    tempos_por_alg : dict[str, list[float]]
        algoritmo -> lista de tempos de execução por run.
    custos_por_alg : dict[str, list[float]]
        algoritmo -> lista de custo_objetivo por run.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in tempos_por_alg:
        tempos = tempos_por_alg[alg]
        custos = custos_por_alg.get(alg, [])
        if tempos and custos:
            ax.scatter(tempos, custos, label=alg, alpha=0.7, s=40)

    ax.set_xlabel("Tempo de execução (s)")
    ax.set_ylabel("Custo objetivo")
    ax.set_title(f"Custo vs Tempo — instância {instancia_nome}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B3 — Heatmap de gap ao melhor conhecido
# ---------------------------------------------------------------------------

def plotar_heatmap_gap(resumo_rows, output_path):
    """Heatmap instância × algoritmo colorido pelo gap percentual ao exato.

    Apenas instâncias/algoritmos com gap_exato preenchido são exibidos.
    """
    linhas = [r for r in resumo_rows if r.get("gap_exato") not in ("", None)]
    if not linhas:
        return

    df = pd.DataFrame(linhas)[["instancia", "algoritmo", "gap_exato"]]
    df["gap_exato"] = pd.to_numeric(df["gap_exato"], errors="coerce")
    df = df.dropna(subset=["gap_exato"])
    if df.empty:
        return

    pivot = df.pivot(index="instancia", columns="algoritmo", values="gap_exato")
    n_cols = len(pivot.columns)
    n_rows = len(pivot)

    fig, ax = plt.subplots(figsize=(max(5, n_cols * 2), max(3, n_rows * 1.5)))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax,
        linewidths=0.5, cbar_kws={"label": "Gap (%)"},
    )
    ax.set_title("Gap percentual ao melhor conhecido (%)")
    ax.set_xlabel("Algoritmo")
    ax.set_ylabel("Instância")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# C1 — Diagrama de Gantt por veículo
# ---------------------------------------------------------------------------

def plotar_gantt_rota(solucao, instancia, output_path):
    """Diagrama de Gantt VRPTW: uma linha por veículo, blocos = atendimentos.

    Cores: verde = no prazo, laranja = aguardou abertura, vermelho = atraso.
    Marcadores verticais indicam as janelas de tempo de cada cliente.
    Ignorado silenciosamente se instancia não tiver dados VRPTW.
    """
    from utilitarios.simulacao_rota import simular_rota

    if getattr(instancia, "matriz_tempos", None) is None:
        return

    n_rotas = len(solucao.rotas)
    if n_rotas == 0:
        return

    cores = {"ok": "#2ecc71", "espera": "#f39c12", "atraso": "#e74c3c"}

    fig, ax = plt.subplots(figsize=(12, max(3, n_rotas * 0.9)))

    for idx, rota in enumerate(solucao.rotas):
        timeline = simular_rota(rota, instancia)
        y = n_rotas - 1 - idx  # veículo 1 no topo

        for ev in timeline:
            cor = (
                cores["atraso"] if ev["atraso"] > 0
                else cores["espera"] if ev["espera"] > 0
                else cores["ok"]
            )
            ax.barh(y, ev["t_servico"], left=ev["t_inicio_servico"], height=0.6,
                    color=cor, edgecolor="white", linewidth=0.5)
            ax.text(
                ev["t_inicio_servico"] + ev["t_servico"] / 2, y,
                str(ev["no"]), ha="center", va="center", fontsize=7, color="white",
            )
            # Marcadores de janela de tempo
            for t_jw in [ev["janela_ini"], ev["janela_fim"]]:
                ax.plot([t_jw, t_jw], [y - 0.35, y + 0.35],
                        color="navy", linewidth=1.0, alpha=0.5)

    ax.set_yticks(range(n_rotas))
    ax.set_yticklabels([f"Veículo {n_rotas - i}" for i in range(n_rotas)])
    ax.set_xlabel("Tempo (min)")
    nome_fig = os.path.splitext(os.path.basename(output_path))[0]
    ax.set_title(f"Diagrama de Gantt — {nome_fig}")
    ax.legend(handles=[
        Patch(facecolor=cores["ok"], label="No prazo"),
        Patch(facecolor=cores["espera"], label="Aguardou abertura"),
        Patch(facecolor=cores["atraso"], label="Atraso"),
    ], loc="upper right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# C2 — Histograma de utilização de capacidade
# ---------------------------------------------------------------------------

def plotar_utilizacao_capacidade(solucao, instancia, output_path):
    """Barras de % de capacidade usada por rota.

    Verde ≤ 80 %, laranja 80-100 %, vermelho > 100 % (violação).
    """
    capacidade = getattr(instancia, "capacidade_caminhao", None)
    demandas = getattr(instancia, "demandas", None)
    if capacidade is None or demandas is None:
        return

    utilizacoes = [
        100.0 * sum(float(demandas[n]) for n in rota if n != 0) / capacidade
        for rota in solucao.rotas
    ]
    if not utilizacoes:
        return

    cores = [
        "#e74c3c" if u > 100 else "#f39c12" if u > 80 else "#2ecc71"
        for u in utilizacoes
    ]

    fig, ax = plt.subplots(figsize=(max(5, len(utilizacoes) * 0.9), 5))
    ax.bar(range(len(utilizacoes)), utilizacoes, color=cores, edgecolor="white")
    ax.axhline(100.0, color="red", linestyle="--", linewidth=1.5, label="Limite de capacidade")
    ax.set_xlabel("Veículo")
    ax.set_ylabel("Utilização de capacidade (%)")
    nome_fig = os.path.splitext(os.path.basename(output_path))[0]
    ax.set_title(f"Utilização de capacidade — {nome_fig}")
    ax.set_xticks(range(len(utilizacoes)))
    ax.set_xticklabels([f"V{i+1}" for i in range(len(utilizacoes))])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# C3 — Barras de nº médio de veículos por algoritmo × instância
# ---------------------------------------------------------------------------

def plotar_stacked_veiculos(resumo_rows, output_path):
    """Barras agrupadas de média de veículos por algoritmo e instância."""
    if not resumo_rows:
        return

    df = pd.DataFrame(resumo_rows)[["instancia", "algoritmo", "media_veiculos"]]
    instancias = df["instancia"].unique()
    algoritmos = df["algoritmo"].unique()

    x = np.arange(len(instancias))
    largura = 0.8 / max(len(algoritmos), 1)

    fig, ax = plt.subplots(figsize=(max(7, len(instancias) * 2), 5))
    for i, alg in enumerate(algoritmos):
        valores = []
        for inst in instancias:
            row = df[(df["instancia"] == inst) & (df["algoritmo"] == alg)]
            valores.append(float(row["media_veiculos"].values[0]) if len(row) > 0 else 0.0)
        ax.bar(x + i * largura - 0.4 + largura / 2, valores, largura, label=alg)

    ax.set_xticks(x)
    ax.set_xticklabels(instancias)
    ax.set_xlabel("Instância")
    ax.set_ylabel("Média de veículos usados")
    ax.set_title("Número médio de veículos por algoritmo e instância")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# D1 — Heatmap triangular de p-values Mann-Whitney
# ---------------------------------------------------------------------------

def plotar_heatmap_pvalores(estatisticas_rows, instancia_nome, output_path):
    """Heatmap triangular superior dos p-values ajustados (Bonferroni) Mann-Whitney.

    Anota *, ** ou *** conforme o nível de significância.
    """
    linhas = [r for r in estatisticas_rows if r.get("teste") == "mann_whitney"]
    if not linhas:
        return

    algs = sorted({r["grupo_a"] for r in linhas} | {r["grupo_b"] for r in linhas})
    if len(algs) < 2:
        return

    n = len(algs)
    idx = {a: i for i, a in enumerate(algs)}
    matriz = np.full((n, n), np.nan)

    for r in linhas:
        i, j = idx[r["grupo_a"]], idx[r["grupo_b"]]
        p = float(r["p_value_ajustado"])
        matriz[i][j] = p
        matriz[j][i] = p

    # Máscara: mostrar apenas triângulo superior (sem diagonal)
    mask = np.tril(np.ones((n, n), dtype=bool))

    df_mat = pd.DataFrame(matriz, index=algs, columns=algs)

    fig, ax = plt.subplots(figsize=(max(4, n * 1.5), max(3, n * 1.2)))
    sns.heatmap(
        df_mat, mask=mask, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0, vmax=1, ax=ax, linewidths=0.5,
        cbar_kws={"label": "p-value ajustado (Bonferroni)"},
    )

    for r in linhas:
        i, j = idx[r["grupo_a"]], idx[r["grupo_b"]]
        if j > i:
            p = float(r["p_value_ajustado"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if sig:
                ax.text(j + 0.5, i + 0.75, sig, ha="center", va="center",
                        fontsize=9, color="black", fontweight="bold")

    ax.set_title(f"p-values Mann-Whitney (Bonferroni) — instância {instancia_nome}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F — Grids consolidados ("small multiples") para múltiplas instâncias
# ---------------------------------------------------------------------------

def _criar_grid_subplots(n_inst, figsize_unit=(5, 4), nrows=None, ncols=None):
    """Cria figura com grid de subplots dimensionado para n_inst painéis.

    Layout padrão: 2x3 (6 slots) para 5-6 instâncias. Última posição reservada
    para legenda compartilhada quando sobra.
    """
    if nrows is None or ncols is None:
        if n_inst <= 2:
            nrows, ncols = 1, n_inst
        elif n_inst <= 4:
            nrows, ncols = 2, 2
        elif n_inst <= 6:
            nrows, ncols = 2, 3
        else:
            ncols = 3
            nrows = (n_inst + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_unit[0] * ncols, figsize_unit[1] * nrows),
        squeeze=False,
    )
    return fig, axes, nrows, ncols


def _finalizar_grid(fig, axes, nrows, ncols, n_inst, handles_labels=None, suptitle=None):
    """Esconde subplots não usados, adiciona legenda compartilhada e título."""
    for idx in range(n_inst, nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("off")

    if handles_labels is not None and handles_labels[0]:
        handles, labels = handles_labels
        # Tenta colocar legenda no primeiro slot vazio; senão usa figura inteira.
        if n_inst < nrows * ncols:
            slot_idx = n_inst
            ax_leg = axes[slot_idx // ncols][slot_idx % ncols]
            ax_leg.axis("off")
            ax_leg.legend(handles, labels, loc="center", fontsize=11, frameon=False,
                          title="Algoritmo")
        else:
            fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                       bbox_to_anchor=(0.5, -0.02), frameon=False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.00)
    fig.tight_layout()


def plotar_convergencia_grid(conv_por_inst, output_path, n_pontos=100):
    """Grid de curvas de convergência (uma instância por subplot).

    Parameters
    ----------
    conv_por_inst : dict[str, dict[str, list[list]]]
        {instancia: {algoritmo: [historicos por run]}}.
    """
    instancias = list(conv_por_inst.keys())
    n_inst = len(instancias)
    if n_inst == 0:
        return

    fig, axes, nrows, ncols = _criar_grid_subplots(n_inst)
    x = np.linspace(0.0, 1.0, n_pontos)
    handles_legenda, labels_legenda = [], []

    for i, inst in enumerate(instancias):
        ax = axes[i // ncols][i % ncols]
        for alg, historicos in conv_por_inst[inst].items():
            if not historicos:
                continue
            curvas = np.vstack([_interp_historico(h, n_pontos) for h in historicos])
            mediana = np.nanmedian(curvas, axis=0)
            q1 = np.nanpercentile(curvas, 25, axis=0)
            q3 = np.nanpercentile(curvas, 75, axis=0)
            linha, = ax.plot(x, mediana, label=alg, linewidth=1.8)
            ax.fill_between(x, q1, q3, alpha=0.2, color=linha.get_color())
            if alg not in labels_legenda:
                labels_legenda.append(alg)
                handles_legenda.append(linha)
        ax.set_yscale("log")
        ax.set_xlabel("Fração de iterações", fontsize=9)
        ax.set_ylabel("Custo (log)", fontsize=9)
        ax.set_title(inst, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    _finalizar_grid(fig, axes, nrows, ncols, n_inst,
                    handles_labels=(handles_legenda, labels_legenda),
                    suptitle="Convergência por iteração — instâncias grandes")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plotar_convergencia_tempo_grid(conv_por_inst, output_path, n_pontos=100):
    """Grid de convergência por tempo real (uma instância por subplot)."""
    instancias = list(conv_por_inst.keys())
    n_inst = len(instancias)
    if n_inst == 0:
        return

    fig, axes, nrows, ncols = _criar_grid_subplots(n_inst)
    handles_legenda, labels_legenda = [], []
    algum_dado = False

    for i, inst in enumerate(instancias):
        ax = axes[i // ncols][i % ncols]
        conv_por_alg = conv_por_inst[inst]
        t_max = 0.0
        for historicos in conv_por_alg.values():
            for h in historicos:
                t = _extrair_tempos(h)
                if t is not None and len(t) > 0:
                    t_max = max(t_max, float(t[-1]))
        if t_max <= 0:
            ax.set_visible(False)
            continue
        t_grid = np.linspace(0.0, t_max, n_pontos)
        for alg, historicos in conv_por_alg.items():
            curvas = []
            for h in historicos:
                if h and _extrair_tempos(h) is not None:
                    c = _interp_historico_por_tempo(h, t_grid)
                    if not np.all(np.isnan(c)):
                        curvas.append(c)
            if not curvas:
                continue
            curvas = np.vstack(curvas)
            mediana = np.nanmedian(curvas, axis=0)
            q1 = np.nanpercentile(curvas, 25, axis=0)
            q3 = np.nanpercentile(curvas, 75, axis=0)
            linha, = ax.plot(t_grid, mediana, label=alg, linewidth=1.8)
            ax.fill_between(t_grid, q1, q3, alpha=0.2, color=linha.get_color())
            if alg not in labels_legenda:
                labels_legenda.append(alg)
                handles_legenda.append(linha)
            algum_dado = True
        ax.set_xlabel("Tempo (s)", fontsize=9)
        ax.set_ylabel("Custo", fontsize=9)
        ax.set_title(inst, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    if not algum_dado:
        plt.close(fig)
        return

    _finalizar_grid(fig, axes, nrows, ncols, n_inst,
                    handles_labels=(handles_legenda, labels_legenda),
                    suptitle="Convergência por tempo — instâncias grandes")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plotar_violino_grid(custos_por_inst, output_path):
    """Grid de violinos (distribuição de custo final por algoritmo) por instância."""
    instancias = list(custos_por_inst.keys())
    n_inst = len(instancias)
    if n_inst == 0:
        return

    fig, axes, nrows, ncols = _criar_grid_subplots(n_inst)

    for i, inst in enumerate(instancias):
        ax = axes[i // ncols][i % ncols]
        runs = custos_por_inst[inst]
        registros = [
            {"algoritmo": alg, "custo": v}
            for alg, vals in runs.items()
            for v in vals
        ]
        if not registros:
            ax.set_visible(False)
            continue
        df = pd.DataFrame(registros)
        sns.violinplot(data=df, x="algoritmo", y="custo", inner="box", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Custo objetivo", fontsize=9)
        ax.set_title(inst, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(labelsize=8)

    _finalizar_grid(fig, axes, nrows, ncols, n_inst,
                    suptitle="Distribuição do custo final — instâncias grandes")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plotar_scatter_custo_tempo_grid(tempos_por_inst, custos_por_inst, output_path):
    """Grid de scatter (tempo, custo) por instância."""
    instancias = list(tempos_por_inst.keys())
    n_inst = len(instancias)
    if n_inst == 0:
        return

    fig, axes, nrows, ncols = _criar_grid_subplots(n_inst)
    handles_legenda, labels_legenda = [], []

    for i, inst in enumerate(instancias):
        ax = axes[i // ncols][i % ncols]
        tempos_por_alg = tempos_por_inst[inst]
        custos_por_alg = custos_por_inst.get(inst, {})
        for alg in tempos_por_alg:
            tempos = tempos_por_alg[alg]
            custos = custos_por_alg.get(alg, [])
            if tempos and custos:
                sc = ax.scatter(tempos, custos, label=alg, alpha=0.7, s=30)
                if alg not in labels_legenda:
                    labels_legenda.append(alg)
                    handles_legenda.append(sc)
        ax.set_xlabel("Tempo (s)", fontsize=9)
        ax.set_ylabel("Custo objetivo", fontsize=9)
        ax.set_title(inst, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    _finalizar_grid(fig, axes, nrows, ncols, n_inst,
                    handles_labels=(handles_legenda, labels_legenda),
                    suptitle="Custo vs tempo — instâncias grandes")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plotar_time_to_target_grid(conv_por_inst, alvo_por_inst, output_path):
    """Grid de ECDF time-to-target por instância.

    Parameters
    ----------
    alvo_por_inst : dict[str, float]
        {instancia: custo_alvo}.
    """
    instancias = list(conv_por_inst.keys())
    n_inst = len(instancias)
    if n_inst == 0:
        return

    fig, axes, nrows, ncols = _criar_grid_subplots(n_inst)
    handles_legenda, labels_legenda = [], []
    algum_dado = False

    for i, inst in enumerate(instancias):
        ax = axes[i // ncols][i % ncols]
        alvo = alvo_por_inst.get(inst)
        if alvo is None:
            ax.set_visible(False)
            continue
        tem_dados = False
        for alg, historicos in conv_por_inst[inst].items():
            n_total = len(historicos)
            tts = []
            for h in historicos:
                tempos = _extrair_tempos(h)
                custos = _extrair_custos(h)
                if tempos is None or len(custos) == 0:
                    continue
                reached = np.where(custos <= alvo)[0]
                if len(reached) > 0:
                    tts.append(float(tempos[reached[0]]))
            if not tts:
                continue
            tts_sorted = np.sort(tts)
            frac = np.arange(1, len(tts_sorted) + 1) / n_total
            linha, = ax.step(tts_sorted, frac, where="post", label=alg, linewidth=1.8)
            if alg not in labels_legenda:
                labels_legenda.append(alg)
                handles_legenda.append(linha)
            tem_dados = True
            algum_dado = True
        if not tem_dados:
            ax.set_visible(False)
            continue
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Tempo (s)", fontsize=9)
        ax.set_ylabel("Fração runs", fontsize=9)
        ax.set_title(f"{inst} (alvo≤{alvo:.0f})", fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    if not algum_dado:
        plt.close(fig)
        return

    _finalizar_grid(fig, axes, nrows, ncols, n_inst,
                    handles_labels=(handles_legenda, labels_legenda),
                    suptitle="Time-to-target — instâncias grandes")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plotar_pvalores_grid(estatisticas_por_inst, output_path):
    """Grid de heatmaps de p-values Mann-Whitney (Bonferroni) por instância."""
    instancias = list(estatisticas_por_inst.keys())
    n_inst = len(instancias)
    if n_inst == 0:
        return

    fig, axes, nrows, ncols = _criar_grid_subplots(n_inst, figsize_unit=(4.5, 4))
    algum_dado = False

    for i, inst in enumerate(instancias):
        ax = axes[i // ncols][i % ncols]
        linhas = [r for r in estatisticas_por_inst[inst] if r.get("teste") == "mann_whitney"]
        if not linhas:
            ax.set_visible(False)
            continue
        algs = sorted({r["grupo_a"] for r in linhas} | {r["grupo_b"] for r in linhas})
        if len(algs) < 2:
            ax.set_visible(False)
            continue

        n = len(algs)
        idx = {a: k for k, a in enumerate(algs)}
        matriz = np.full((n, n), np.nan)
        for r in linhas:
            ia, ib = idx[r["grupo_a"]], idx[r["grupo_b"]]
            p = float(r["p_value_ajustado"])
            matriz[ia][ib] = p
            matriz[ib][ia] = p

        mask = np.tril(np.ones((n, n), dtype=bool))
        df_mat = pd.DataFrame(matriz, index=algs, columns=algs)
        sns.heatmap(
            df_mat, mask=mask, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
            cbar=False, annot_kws={"fontsize": 8},
        )
        for r in linhas:
            ia, ib = idx[r["grupo_a"]], idx[r["grupo_b"]]
            if ib > ia:
                p = float(r["p_value_ajustado"])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if sig:
                    ax.text(ib + 0.5, ia + 0.78, sig, ha="center", va="center",
                            fontsize=8, color="black", fontweight="bold")
        ax.set_title(inst, fontsize=10)
        ax.tick_params(labelsize=8)
        algum_dado = True

    if not algum_dado:
        plt.close(fig)
        return

    _finalizar_grid(fig, axes, nrows, ncols, n_inst,
                    suptitle="p-values Mann-Whitney (Bonferroni) — instâncias grandes")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E1 — Perfil de desempenho de Dolan-Moré
# ---------------------------------------------------------------------------

def plotar_dolan_more(resumo_rows, output_path, metrica="melhor_objetivo",
                      instancias_excluir=None):
    """Perfil de desempenho de Dolan & Moré (2002) por algoritmo.

    Parameters
    ----------
    resumo_rows : list[dict]
        Linhas de resultados_resumo.csv (uma por instância × algoritmo).
    metrica : str
        Coluna a usar como custo por run. Padrão: 'melhor_objetivo'.
    instancias_excluir : set[str] | None
        Nomes de instâncias a excluir (ex: {'pequena'} para focar só nas grandes).
    """
    instancias_excluir = instancias_excluir or set()
    linhas = [
        r for r in resumo_rows
        if r.get("algoritmo") != "Exato" and r.get("instancia") not in instancias_excluir
    ]
    if not linhas:
        return

    df = pd.DataFrame(linhas)[["instancia", "algoritmo", metrica]]
    df[metrica] = pd.to_numeric(df[metrica], errors="coerce")
    df = df.dropna(subset=[metrica])
    if df.empty:
        return

    instancias = df["instancia"].unique()
    algoritmos = sorted(df["algoritmo"].unique())
    n_p = len(instancias)

    # Monta matriz M[algoritmo, instância]
    pivot = df.pivot(index="instancia", columns="algoritmo", values=metrica)

    # Performance ratios r[algoritmo, instância] = M / min_alg(M)
    best_per_inst = pivot.min(axis=1)
    ratios = pivot.div(best_per_inst, axis=0)

    tau_max = float(ratios.max().max())
    tau_vals = np.linspace(1.0, tau_max * 1.05, 2000)

    fig, ax = plt.subplots(figsize=(9, 5))
    for alg in algoritmos:
        if alg not in ratios.columns:
            continue
        r_alg = ratios[alg].dropna().values
        rho = np.array([np.mean(r_alg <= tau) for tau in tau_vals])
        ax.step(tau_vals, rho, where="post", label=alg, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("τ (razão de desempenho)")
    ax.set_ylabel("ρ(τ) — fração de instâncias resolvidas")
    ax.set_title("Perfil de Dolan-Moré (Dolan & Moré, 2002)")
    ax.set_xlim(left=1.0)
    ax.set_ylim(-0.05, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# E2 — Teste de Friedman + diagrama de distância crítica (Nemenyi)
# ---------------------------------------------------------------------------

# Valores críticos q_α para Nemenyi (Demšar 2006, Tabela 5), α=0.05
# q = (valor da tabela do studentized range) / sqrt(2)
_NEMENYI_Q = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def plotar_friedman_cd(runs_raw_rows, output_path, alpha=0.05, instancias_excluir=None):
    """Teste de Friedman + diagrama de distância crítica de Nemenyi (Demšar 2006).

    Parameters
    ----------
    runs_raw_rows : list[dict]
        Linhas de resultados_runs.csv (uma por instância × algoritmo × run).
    alpha : float
        Nível de significância para o CD de Nemenyi.
    instancias_excluir : set[str] | None
        Nomes de instâncias a excluir (ex: {'pequena'}).

    Returns
    -------
    dict com chaves: chi2, p, cd, ranks, n_instancias, n_algoritmos
    """
    instancias_excluir = instancias_excluir or set()
    linhas = [
        r for r in runs_raw_rows
        if r.get("algoritmo") != "Exato" and r.get("instancia") not in instancias_excluir
    ]

    resultado_vazio = {"chi2": None, "p": None, "cd": None, "ranks": {}, "n_instancias": 0, "n_algoritmos": 0}
    if not linhas:
        return resultado_vazio

    df = pd.DataFrame(linhas)[["instancia", "algoritmo", "custo_objetivo"]]
    df["custo_objetivo"] = pd.to_numeric(df["custo_objetivo"], errors="coerce")
    df = df.dropna(subset=["custo_objetivo"])

    # Estatística representativa por (instância, algoritmo) = mediana sobre runs
    resumo = (df.groupby(["instancia", "algoritmo"])["custo_objetivo"]
                .median().reset_index().rename(columns={"custo_objetivo": "mediana"}))

    instancias = sorted(resumo["instancia"].unique())
    algoritmos = sorted(resumo["algoritmo"].unique())
    n_i = len(instancias)
    n_k = len(algoritmos)

    if n_i < 2 or n_k < 2:
        return resultado_vazio

    pivot = resumo.pivot(index="instancia", columns="algoritmo", values="mediana")
    pivot = pivot.reindex(index=instancias, columns=algoritmos)
    if pivot.isnull().any().any():
        return resultado_vazio

    # Ranks por linha (instância): 1 = melhor custo
    rank_matrix = pivot.apply(lambda row: scipy_stats.rankdata(row), axis=1, result_type="expand")
    rank_matrix.columns = algoritmos
    rank_medios = rank_matrix.mean()

    # Teste de Friedman
    amostras = [pivot[alg].values for alg in algoritmos]
    chi2, p_valor = scipy_stats.friedmanchisquare(*amostras)

    # CD de Nemenyi
    q_alpha = _NEMENYI_Q.get(n_k)
    cd = q_alpha * np.sqrt(n_k * (n_k + 1) / (6.0 * n_i)) if q_alpha else None

    # ---- Diagrama CD ----
    fig, ax = plt.subplots(figsize=(max(7, n_k * 2), 3.5))

    # Eixo de ranks (1 = melhor à esquerda)
    ax.set_xlim(0.5, n_k + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, n_k + 1))
    ax.set_xlabel("Rank médio (1 = melhor)")
    ax.set_title(
        f"Diagrama de Distância Crítica — Friedman χ²={chi2:.3f}, p={p_valor:.4f}"
        + (f" | CD (Nemenyi, α={alpha}) = {cd:.3f}" if cd else "")
    )
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_ticks_position("bottom")

    # Posiciona cada algoritmo
    y_mid = 0.55
    for alg, r in rank_medios.items():
        ax.plot(r, y_mid, "o", ms=10, color="steelblue", zorder=3)
        ax.text(r, y_mid + 0.18, alg, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(r, y_mid - 0.20, f"{r:.2f}", ha="center", va="top", fontsize=8, color="gray")

    # Barra CD conectando grupos não significativamente diferentes
    if cd is not None:
        ranks_sorted = sorted(rank_medios.items(), key=lambda x: x[1])
        i = 0
        while i < len(ranks_sorted):
            alg_i, r_i = ranks_sorted[i]
            # encontra o grupo que não difere mais que CD
            j = i
            while j + 1 < len(ranks_sorted) and ranks_sorted[j + 1][1] - r_i <= cd:
                j += 1
            if j > i:  # grupo com ≥2 membros
                r_start = ranks_sorted[i][1]
                r_end = ranks_sorted[j][1]
                ax.plot([r_start, r_end], [y_mid - 0.08, y_mid - 0.08],
                        linewidth=4, color="steelblue", alpha=0.4, solid_capstyle="round")
            i += 1

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    return {
        "chi2": float(chi2),
        "p": float(p_valor),
        "cd": float(cd) if cd is not None else None,
        "ranks": {alg: float(r) for alg, r in rank_medios.items()},
        "n_instancias": n_i,
        "n_algoritmos": n_k,
    }
