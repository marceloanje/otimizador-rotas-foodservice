"""Geração de gráficos para o experimento comparativo.

- plotar_convergencia: curva média + banda IQR por algoritmo, eixo x normalizado.
- plotar_boxplot_custos: boxplot do custo_objetivo final por algoritmo.
- plotar_tempo_execucao: barras de tempo médio por algoritmo x instância.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _interp_historico(historico, n_pontos=100):
    """Reamostra uma curva de convergência para n_pontos no eixo [0,1]."""
    if not historico:
        return np.full(n_pontos, np.nan)
    hist = np.asarray(historico, dtype=float)
    if len(hist) == 1:
        return np.full(n_pontos, hist[0])
    x_original = np.linspace(0.0, 1.0, len(hist))
    x_target = np.linspace(0.0, 1.0, n_pontos)
    return np.interp(x_target, x_original, hist)


def plotar_convergencia(convergencia_por_alg, instancia_nome, output_path, n_pontos=100):
    """Gera figura de convergência.

    Parameters
    ----------
    convergencia_por_alg : dict[str, list[list[float]]]
        Mapeamento algoritmo -> lista de históricos (um por run).
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


def plotar_boxplot_custos(runs_por_alg, instancia_nome, output_path):
    """Gera boxplot do custo_objetivo final por algoritmo.

    Parameters
    ----------
    runs_por_alg : dict[str, list[float]]
        Mapeamento algoritmo -> lista de custo_objetivo finais.
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
    """Gera barras de tempo médio por algoritmo x instância.

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
