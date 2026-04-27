"""Geração de gráficos para o experimento comparativo.

Funções de convergência (por iteração e por tempo), distribuição de custos,
análise estatística, estrutura de rotas (Gantt, capacidade) e comparações
agregadas (gap ao ótimo, número de veículos).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd


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

    ax.set_xlabel("Fração de iterações")
    ax.set_ylabel("Melhor custo objetivo")
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

    Parameters
    ----------
    tempos_por_inst_alg : dict[str, dict[str, float]]
        {instancia: {algoritmo: tempo_medio}}.
    """
    instancias = list(tempos_por_inst_alg.keys())
    if not instancias:
        return
    algoritmos = sorted({a for d in tempos_por_inst_alg.values() for a in d})

    x = np.arange(len(instancias))
    largura = 0.8 / max(len(algoritmos), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, alg in enumerate(algoritmos):
        valores = [tempos_por_inst_alg[inst].get(alg, 0.0) for inst in instancias]
        ax.bar(x + i * largura - 0.4 + largura / 2, valores, largura, label=alg)

    ax.set_xticks(x)
    ax.set_xticklabels(instancias)
    ax.set_ylabel("Tempo médio de execução (s)")
    ax.set_title("Tempo médio por algoritmo e instância")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
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
