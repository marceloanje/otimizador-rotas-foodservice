"""Framework experimental comparativo de meta-heurísticas para CVRP/VRPTW.

Roda ACO, Busca Tabu e PSO (e opcionalmente o solver exato como baseline)
em múltiplas instâncias, coleta histórico de convergência por execução,
aplica testes estatísticos não-paramétricos (Kruskal-Wallis + Mann-Whitney U
com correção de Bonferroni) e exporta quatro CSVs + gráficos.

Saída em `resultados/`:
    resultados_runs.csv          — uma linha por run
    resultados_resumo.csv        — agregação por (instância, algoritmo)
    resultados_convergencia.csv  — formato longo (iter, melhor_custo)
    resultados_estatisticas.csv  — Kruskal-Wallis + Mann-Whitney pós-hoc
    figuras/                     — PNGs (convergência, boxplots, tempos)
"""

import time
import numpy as np
import csv
import statistics
import os
import random
import pickle

from scipy import stats

from config_experimento import INSTANCIAS, N_RUNS, SEED_BASE
from modelos.instancia import Instancia
from algoritmos.colonia_formigas import ACO
from algoritmos.busca_tabu import BuscaTabu
from algoritmos.enxame_particulas import PSO
from utilitarios.graficos_experimento import (
    plotar_convergencia,
    plotar_tempo_execucao,
    plotar_convergencia_tempo,
    plotar_time_to_target,
    plotar_violino_custos,
    plotar_scatter_custo_tempo,
    plotar_heatmap_gap,
    plotar_gantt_rota,
    plotar_utilizacao_capacidade,
    plotar_stacked_veiculos,
    plotar_heatmap_pvalores,
    plotar_dolan_more,
    plotar_friedman_cd,
    plotar_convergencia_grid,
    plotar_convergencia_tempo_grid,
    plotar_violino_grid,
    plotar_scatter_custo_tempo_grid,
    plotar_time_to_target_grid,
    plotar_pvalores_grid,
)


ALGORITMOS_HEURISTICOS = [
    ("ACO", ACO),
    ("Tabu", BuscaTabu),
    ("PSO", PSO),
]

# Instâncias para as quais Gantt e utilização de capacidade são geradas
# individualmente. Demais instâncias têm seus gráficos consolidados em grids.
INSTANCIAS_REPRESENTATIVAS_GANTT = {"pequena", "grande", "grande_5"}


def executar_algoritmo(name, solver_cls, instancia, runs, seed_base, solver_kwargs=None):
    """Executa um algoritmo `runs` vezes e retorna uma lista de dicts por run.

    Cada dict contém as métricas padrão + o histórico de convergência
    extraído de `sol.meta["historico_convergencia"]`.
    """
    results = []
    solver_kwargs = solver_kwargs or {}

    for r in range(runs):
        random.seed(seed_base + r)
        np.random.seed(seed_base + r)

        solver = solver_cls(instancia, **solver_kwargs)
        t0 = time.time()
        sol = solver.run()
        t1 = time.time()
        tempo = t1 - t0

        if sol.custo_objetivo is None:
            sol.avaliar(instancia)

        eh_valida = sol.eh_valida(instancia)
        n_violacoes = sol.violacoes.get("capacidade", 0)
        clientes_faltando = sol.violacoes.get("cobertura", 0)
        frota_excedida = sol.violacoes.get("frota_excedida", 0)

        n_violacoes_jt, total_atraso_jt = 0, 0.0
        if getattr(instancia, "matriz_tempos", None) is not None:
            vjt = sol.verificar_janelas_tempo(instancia)
            n_violacoes_jt = sum(len(v) for v in vjt.values())
            total_atraso_jt = sum(a for lst in vjt.values() for (_, a) in lst)

        meta = getattr(sol, "meta", {}) or {}
        historico = list(meta.get("historico_convergencia", []))
        max_iter = meta.get("max_iter", len(historico))

        results.append({
            "algoritmo": name,
            "run": r,
            "seed": seed_base + r,
            "custo": sol.custo,
            "custo_objetivo": sol.custo_objetivo,
            "n_veiculos": sol.n_veiculos,
            "eh_valida": eh_valida,
            "n_violacoes": n_violacoes,
            "clientes_faltando": clientes_faltando,
            "frota_excedida": frota_excedida,
            "n_violacoes_jt": n_violacoes_jt,
            "total_atraso_jt": total_atraso_jt,
            "solucao": sol,
            "tempo": tempo,
            "historico": historico,
            "max_iter": max_iter,
        })
    return results


def resumir_results(results):
    """Agrega um conjunto de runs em estatísticas descritivas."""
    custos = [r["custo"] for r in results]
    custos_objetivo = [r["custo_objetivo"] for r in results]
    tempos = [r["tempo"] for r in results]
    n_veiculos_list = [r["n_veiculos"] for r in results]
    validas = [r["eh_valida"] for r in results]
    violacoes = [r["n_violacoes"] for r in results]
    clientes_faltando = [r["clientes_faltando"] for r in results]
    frota_excedida = [r["frota_excedida"] for r in results]
    violacoes_jt_list = [r["n_violacoes_jt"] for r in results]
    atraso_jt_list = [r["total_atraso_jt"] for r in results]

    taxa_factivel = sum(validas) / len(validas) if validas else 0.0

    return {
        "n_runs": len(results),
        "melhor_custo": float(min(custos)),
        "media_custo": float(statistics.mean(custos)),
        "mediana_custo": float(statistics.median(custos)),
        "desvio_custo": float(statistics.pstdev(custos)),
        "q1_custo": float(np.percentile(custos, 25)),
        "q3_custo": float(np.percentile(custos, 75)),
        "min_custo": float(min(custos)),
        "max_custo": float(max(custos)),
        "melhor_objetivo": float(min(custos_objetivo)),
        "media_objetivo": float(statistics.mean(custos_objetivo)),
        "mediana_objetivo": float(statistics.median(custos_objetivo)),
        "desvio_objetivo": float(statistics.pstdev(custos_objetivo)),
        "media_veiculos": float(statistics.mean(n_veiculos_list)),
        "taxa_factivel": float(taxa_factivel),
        "media_violacoes": float(statistics.mean(violacoes)),
        "media_clientes_faltando": float(statistics.mean(clientes_faltando)),
        "media_frota_excedida": float(statistics.mean(frota_excedida)),
        "media_violacoes_jt": float(statistics.mean(violacoes_jt_list)),
        "media_atraso_jt": float(statistics.mean(atraso_jt_list)),
        "tempo_med": float(statistics.mean(tempos)),
    }


def analise_estatistica(runs_por_alg, instancia_nome):
    """Executa Kruskal-Wallis global + Mann-Whitney pós-hoc com Bonferroni.

    Usa `custo_objetivo` como variável de resposta.
    Retorna uma lista de dicts prontos para escrever em CSV.
    """
    linhas = []
    algs = [a for a in runs_por_alg if len(runs_por_alg[a]) >= 2]
    if len(algs) < 2:
        return linhas

    amostras = {a: [r["custo_objetivo"] for r in runs_por_alg[a]] for a in algs}

    if len(algs) >= 3:
        h_stat, p_kw = stats.kruskal(*[amostras[a] for a in algs])
        linhas.append({
            "instancia": instancia_nome,
            "teste": "kruskal_wallis",
            "grupo_a": "+".join(algs),
            "grupo_b": "",
            "estatistica": float(h_stat),
            "p_value": float(p_kw),
            "p_value_ajustado": float(p_kw),
            "significativo": bool(p_kw < 0.05),
            "melhor": "",
            "diferenca_medianas": "",
        })

    pares = [(algs[i], algs[j]) for i in range(len(algs)) for j in range(i + 1, len(algs))]
    n_pares = len(pares)

    for a, b in pares:
        u_stat, p_raw = stats.mannwhitneyu(amostras[a], amostras[b], alternative="two-sided")
        p_adj = min(1.0, float(p_raw) * n_pares)
        med_a = float(statistics.median(amostras[a]))
        med_b = float(statistics.median(amostras[b]))
        melhor = a if med_a < med_b else (b if med_b < med_a else "empate")
        linhas.append({
            "instancia": instancia_nome,
            "teste": "mann_whitney",
            "grupo_a": a,
            "grupo_b": b,
            "estatistica": float(u_stat),
            "p_value": float(p_raw),
            "p_value_ajustado": p_adj,
            "significativo": bool(p_adj < 0.05),
            "melhor": melhor,
            "diferenca_medianas": med_a - med_b,
        })

    return linhas


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _escrever_csv(path, linhas, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for linha in linhas:
            writer.writerow({k: linha.get(k, "") for k in fieldnames})


def _formatar_duracao(segundos):
    """Converte segundos em string 'HhMMmSSs' (formato compacto)."""
    s = int(round(segundos))
    h, resto = divmod(s, 3600)
    m, s = divmod(resto, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{segundos:.2f}s"


def _limpar_figuras_orfas(fig_dir, instancias):
    """Remove PNGs antigos de instâncias que deixaram de ser representativas.

    Mantém figuras do conjunto atual de instâncias para que execuções repetidas
    não acumulem PNGs obsoletos (boxplot_*, figuras de Gantt/utilização de
    instâncias não-representativas, etc.).
    """
    import glob
    if not os.path.isdir(fig_dir):
        return

    # Prefixos por-instância considerados obsoletos com a nova estratégia
    prefixos_obsoletos = ("boxplot_",)
    for fname in os.listdir(fig_dir):
        if any(fname.startswith(p) for p in prefixos_obsoletos):
            try:
                os.remove(os.path.join(fig_dir, fname))
            except OSError:
                pass

    # Gantt/utilização de instâncias não-representativas
    nomes_atuais = {i["nome"] for i in instancias}
    representativas = INSTANCIAS_REPRESENTATIVAS_GANTT & nomes_atuais
    for tipo in ("gantt", "utilizacao"):
        for fpath in glob.glob(os.path.join(fig_dir, f"{tipo}_*.png")):
            base = os.path.basename(fpath)
            # padrão: <tipo>_<instancia>__<alg>.png
            sem_ext = os.path.splitext(base)[0]
            if "__" not in sem_ext:
                continue
            inst_nome = sem_ext.split(f"{tipo}_", 1)[1].split("__", 1)[0]
            if inst_nome in nomes_atuais and inst_nome not in representativas:
                try:
                    os.remove(fpath)
                except OSError:
                    pass


def comparar_multi_instancia(instancias=None):
    """Executa o experimento comparativo para as instâncias fornecidas.

    Parameters
    ----------
    instancias : list[dict] | None
        Lista de configs de instância (mesmo formato de INSTANCIAS em
        config_experimento.py). Se None, usa todas as instâncias do config.
    """
    if instancias is None:
        instancias = INSTANCIAS

    n_inst_estatistica = sum(1 for i in instancias if not i.get("usar_exato"))
    if n_inst_estatistica < 5:
        print(
            f"[AVISO] Apenas {n_inst_estatistica} instância(s) sem solver exato. "
            "Dolan-Moré e Friedman+CD são mais informativos com ≥5 instâncias (Demšar 2006)."
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_dir = os.path.join(project_root, "resultados")
    fig_dir = os.path.join(out_dir, "figuras")
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)
    sol_dir = os.path.join(out_dir, "solucoes")
    _ensure_dir(sol_dir)

    _limpar_figuras_orfas(fig_dir, instancias)

    runs_raw_rows = []
    resumo_rows = []
    convergencia_rows = []
    estatisticas_rows = []

    tempos_por_inst_alg = {}
    tempo_por_instancia = {}     # tempo total de wall-clock por instância
    melhor_sol_por_inst_alg = {} # (instancia, algoritmo) -> Solucao para Gantt/utilização
    dados_grid = {}              # instancia -> dict com tudo para plotagem agregada

    t_experimento_inicio = time.time()

    for inst_cfg in instancias:
        nome = inst_cfg["nome"]
        path = inst_cfg["path"]
        usar_exato = inst_cfg.get("usar_exato", False)
        tempo_limite_exato = inst_cfg.get("tempo_limite_exato", 60)

        if not os.path.exists(path):
            print(f"[AVISO] Instância '{nome}' não encontrada em {path}. Pulando.")
            continue

        print(f"\n{'='*60}\nInstância: {nome} ({path})\n{'='*60}")
        t_instancia_inicio = time.time()
        instancia = Instancia.do_csv(
            path,
            capacidade_caminhao=inst_cfg["capacidade_caminhao"],
            numero_caminhoes=inst_cfg["numero_caminhoes"],
        )
        instancia.verificar_factibilidade()
        print(f"  n_clientes = {instancia.n_clientes}")

        runs_por_alg = {}
        tempos_por_inst_alg[nome] = {}

        for alg_name, alg_cls in ALGORITMOS_HEURISTICOS:
            print(f"\n  Executando {alg_name} ({N_RUNS} runs)...")
            results = executar_algoritmo(alg_name, alg_cls, instancia,
                                         runs=N_RUNS, seed_base=SEED_BASE)
            runs_por_alg[alg_name] = results
            melhor_run = min(results, key=lambda r: r["custo_objetivo"])
            with open(os.path.join(sol_dir, f"{nome}__{alg_name}.pkl"), "wb") as _f:
                pickle.dump(melhor_run["solucao"], _f)
            melhor_sol_por_inst_alg[(nome, alg_name)] = melhor_run["solucao"]
            s = resumir_results(results)
            s["instancia"] = nome
            s["algoritmo"] = alg_name
            s["gap_exato"] = ""
            resumo_rows.append(s)
            tempos_por_inst_alg[nome][alg_name] = s["tempo_med"]
            print(f"    melhor={s['melhor_objetivo']:.2f}  "
                  f"mediana={s['mediana_objetivo']:.2f}  "
                  f"tempo_med={s['tempo_med']:.3f}s")

        if usar_exato:
            print(f"\n  Executando SolverExato (1 run, limite={tempo_limite_exato}s)...")
            from algoritmos.solver_exato import SolverExato
            ex_results = executar_algoritmo(
                "Exato", SolverExato, instancia,
                runs=1, seed_base=SEED_BASE,
                solver_kwargs={"tempo_limite": tempo_limite_exato},
            )
            runs_por_alg["Exato"] = ex_results
            melhor_ex = min(ex_results, key=lambda r: r["custo_objetivo"])
            with open(os.path.join(sol_dir, f"{nome}__Exato.pkl"), "wb") as _f:
                pickle.dump(melhor_ex["solucao"], _f)
            melhor_sol_por_inst_alg[(nome, "Exato")] = melhor_ex["solucao"]
            s_ex = resumir_results(ex_results)
            s_ex["instancia"] = nome
            s_ex["algoritmo"] = "Exato"
            s_ex["gap_exato"] = 0.0
            resumo_rows.append(s_ex)
            tempos_por_inst_alg[nome]["Exato"] = s_ex["tempo_med"]
            print(f"    custo_exato={s_ex['melhor_objetivo']:.2f}  "
                  f"tempo={s_ex['tempo_med']:.2f}s")

            # Gap em relação ao ótimo usa `melhor_custo` (distância pura):
            # é a métrica que o solver exato otimiza
            custo_exato = s_ex["melhor_custo"]
            if custo_exato and custo_exato > 0:
                for row in resumo_rows:
                    if row["instancia"] == nome and row["algoritmo"] != "Exato":
                        row["gap_exato"] = 100.0 * (row["melhor_custo"] - custo_exato) / custo_exato

        for alg_name, results in runs_por_alg.items():
            for r in results:
                runs_raw_rows.append({
                    "instancia": nome,
                    "algoritmo": alg_name,
                    "run": r["run"],
                    "seed": r["seed"],
                    "custo": r["custo"],
                    "custo_objetivo": r["custo_objetivo"],
                    "n_veiculos": r["n_veiculos"],
                    "eh_valida": int(bool(r["eh_valida"])),
                    "n_violacoes": r["n_violacoes"],
                    "clientes_faltando": r["clientes_faltando"],
                    "frota_excedida": r["frota_excedida"],
                    "n_violacoes_jt": r["n_violacoes_jt"],
                    "total_atraso_jt": r["total_atraso_jt"],
                    "tempo": r["tempo"],
                })

                hist = r["historico"]
                max_it = r["max_iter"] or len(hist)
                for it_idx, item in enumerate(hist):
                    if isinstance(item, (tuple, list)):
                        iter_num, melhor, t_s = item[0], item[1], item[2]
                    else:
                        iter_num = it_idx + 1
                        melhor = item
                        t_s = ""
                    iter_frac = iter_num / max_it if max_it else 0.0
                    convergencia_rows.append({
                        "instancia": nome,
                        "algoritmo": alg_name,
                        "run": r["run"],
                        "iteracao": iter_num,
                        "iter_frac": iter_frac,
                        "melhor_custo_objetivo": melhor,
                        "tempo_s": t_s,
                    })

        runs_heur = {a: runs_por_alg[a] for a, _ in ALGORITMOS_HEURISTICOS if a in runs_por_alg}
        estatisticas_inst = analise_estatistica(runs_heur, nome)
        estatisticas_rows.extend(estatisticas_inst)

        # Coleta os dados desta instância; a plotagem é feita em batch ao final.
        convergencia_por_alg = {
            alg: [r["historico"] for r in results if r["historico"]]
            for alg, results in runs_heur.items()
        }
        custos_finais = {
            alg: [r["custo_objetivo"] for r in results]
            for alg, results in runs_heur.items()
        }
        tempos_finais = {
            alg: [r["tempo"] for r in results]
            for alg, results in runs_heur.items()
        }
        todos_objetivos = [r["custo_objetivo"] for res in runs_por_alg.values() for r in res]
        alvo_custo = 1.05 * min(todos_objetivos) if todos_objetivos else None

        dados_grid[nome] = {
            "instancia": instancia,
            "convergencia": convergencia_por_alg,
            "custos_finais": custos_finais,
            "tempos_finais": tempos_finais,
            "alvo_custo": alvo_custo,
            "estatisticas": estatisticas_inst,
            "algoritmos": list(runs_por_alg.keys()),
        }
        tempo_por_instancia[nome] = time.time() - t_instancia_inicio

    print("\nEscrevendo CSVs...")
    _escrever_csv(
        os.path.join(out_dir, "resultados_runs.csv"),
        runs_raw_rows,
        fieldnames=[
            "instancia", "algoritmo", "run", "seed",
            "custo", "custo_objetivo", "n_veiculos", "eh_valida",
            "n_violacoes", "clientes_faltando", "frota_excedida",
            "n_violacoes_jt", "total_atraso_jt", "tempo",
        ],
    )
    _escrever_csv(
        os.path.join(out_dir, "resultados_resumo.csv"),
        resumo_rows,
        fieldnames=[
            "instancia", "algoritmo", "n_runs",
            "melhor_custo", "media_custo", "mediana_custo", "desvio_custo",
            "q1_custo", "q3_custo", "min_custo", "max_custo",
            "melhor_objetivo", "media_objetivo", "mediana_objetivo", "desvio_objetivo",
            "media_veiculos", "taxa_factivel",
            "media_violacoes", "media_clientes_faltando", "media_frota_excedida",
            "media_violacoes_jt", "media_atraso_jt",
            "tempo_med", "gap_exato",
        ],
    )
    _escrever_csv(
        os.path.join(out_dir, "resultados_convergencia.csv"),
        convergencia_rows,
        fieldnames=["instancia", "algoritmo", "run", "iteracao", "iter_frac", "melhor_custo_objetivo", "tempo_s"],
    )
    _escrever_csv(
        os.path.join(out_dir, "resultados_estatisticas.csv"),
        estatisticas_rows,
        fieldnames=[
            "instancia", "teste", "grupo_a", "grupo_b",
            "estatistica", "p_value", "p_value_ajustado",
            "significativo", "melhor", "diferenca_medianas",
        ],
    )

    # -----------------------------------------------------------------------
    # Plotagem consolidada
    # -----------------------------------------------------------------------
    inst_pequenas = [i["nome"] for i in instancias if i.get("usar_exato")]
    inst_grandes = [i["nome"] for i in instancias if not i.get("usar_exato")]

    print("\nGerando figuras individuais (instâncias com solver exato)...")
    for nome in inst_pequenas:
        d = dados_grid.get(nome)
        if d is None:
            continue
        plotar_convergencia(d["convergencia"], nome,
                            os.path.join(fig_dir, f"convergencia_{nome}.png"))
        plotar_convergencia_tempo(d["convergencia"], nome,
                                  os.path.join(fig_dir, f"convergencia_tempo_{nome}.png"))
        plotar_violino_custos(d["custos_finais"], nome,
                              os.path.join(fig_dir, f"violino_{nome}.png"))
        plotar_scatter_custo_tempo(d["tempos_finais"], d["custos_finais"], nome,
                                   os.path.join(fig_dir, f"scatter_custo_tempo_{nome}.png"))
        if d["alvo_custo"] is not None:
            plotar_time_to_target(d["convergencia"], d["alvo_custo"], nome,
                                  os.path.join(fig_dir, f"time_to_target_{nome}.png"))
        plotar_heatmap_pvalores(d["estatisticas"], nome,
                                os.path.join(fig_dir, f"pvalores_{nome}.png"))

    if inst_grandes:
        print(f"Gerando grids consolidados para {len(inst_grandes)} instâncias grandes...")
        conv_grid = {n: dados_grid[n]["convergencia"] for n in inst_grandes if n in dados_grid}
        custos_grid = {n: dados_grid[n]["custos_finais"] for n in inst_grandes if n in dados_grid}
        tempos_grid = {n: dados_grid[n]["tempos_finais"] for n in inst_grandes if n in dados_grid}
        alvo_grid = {n: dados_grid[n]["alvo_custo"] for n in inst_grandes if n in dados_grid}
        estat_grid = {n: dados_grid[n]["estatisticas"] for n in inst_grandes if n in dados_grid}

        plotar_convergencia_grid(conv_grid, os.path.join(fig_dir, "convergencia_grandes_grid.png"))
        plotar_convergencia_tempo_grid(conv_grid, os.path.join(fig_dir, "convergencia_tempo_grandes_grid.png"))
        plotar_violino_grid(custos_grid, os.path.join(fig_dir, "violino_grandes_grid.png"))
        plotar_scatter_custo_tempo_grid(tempos_grid, custos_grid,
                                        os.path.join(fig_dir, "scatter_grandes_grid.png"))
        plotar_time_to_target_grid(conv_grid, alvo_grid,
                                   os.path.join(fig_dir, "time_to_target_grandes_grid.png"))
        plotar_pvalores_grid(estat_grid, os.path.join(fig_dir, "pvalores_grandes_grid.png"))

    print("Gerando Gantt e utilização para instâncias representativas: "
          f"{sorted(INSTANCIAS_REPRESENTATIVAS_GANTT)}")
    for nome in dados_grid:
        if nome not in INSTANCIAS_REPRESENTATIVAS_GANTT:
            continue
        instancia_obj = dados_grid[nome]["instancia"]
        for alg_n in dados_grid[nome]["algoritmos"]:
            sol = melhor_sol_por_inst_alg.get((nome, alg_n))
            if sol is None:
                continue
            plotar_gantt_rota(sol, instancia_obj,
                              os.path.join(fig_dir, f"gantt_{nome}__{alg_n}.png"))
            plotar_utilizacao_capacidade(sol, instancia_obj,
                                         os.path.join(fig_dir, f"utilizacao_{nome}__{alg_n}.png"))

    print("Gerando figuras cross-instance...")
    plotar_tempo_execucao(
        tempos_por_inst_alg,
        os.path.join(fig_dir, "tempos.png"),
    )
    plotar_heatmap_gap(resumo_rows, os.path.join(fig_dir, "heatmap_gap.png"))
    plotar_stacked_veiculos(resumo_rows, os.path.join(fig_dir, "stacked_veiculos.png"))

    inst_excluir = {i["nome"] for i in instancias if i.get("usar_exato")}
    plotar_dolan_more(
        resumo_rows,
        os.path.join(fig_dir, "dolan_more.png"),
        instancias_excluir=inst_excluir,
    )
    friedman_info = plotar_friedman_cd(
        runs_raw_rows,
        os.path.join(fig_dir, "friedman_cd.png"),
        instancias_excluir=inst_excluir,
    )
    if friedman_info.get("ranks"):
        ranks_rows = [
            {
                "algoritmo": alg,
                "rank_medio": r,
                "chi2": friedman_info["chi2"],
                "p_value": friedman_info["p"],
                "cd": friedman_info["cd"],
                "n_instancias": friedman_info["n_instancias"],
            }
            for alg, r in friedman_info["ranks"].items()
        ]
        _escrever_csv(
            os.path.join(out_dir, "resultados_ranks.csv"),
            ranks_rows,
            fieldnames=["algoritmo", "rank_medio", "chi2", "p_value", "cd", "n_instancias"],
        )

    # -----------------------------------------------------------------------
    # Tempos de execução: CSV detalhado + sumário no terminal
    # -----------------------------------------------------------------------
    tempo_total = time.time() - t_experimento_inicio
    tempos_rows = [
        {"escopo": "total", "instancia": "", "algoritmo": "",
         "tempo_segundos": tempo_total, "n_runs": ""},
    ]
    for nome, t in tempo_por_instancia.items():
        tempos_rows.append({
            "escopo": "instancia", "instancia": nome, "algoritmo": "",
            "tempo_segundos": t, "n_runs": "",
        })
    tempo_por_algoritmo = {}
    for nome, algs in tempos_por_inst_alg.items():
        for alg, t_med in algs.items():
            n_runs_alg = 1 if alg == "Exato" else N_RUNS
            tempo_total_alg_inst = t_med * n_runs_alg
            tempo_por_algoritmo[alg] = tempo_por_algoritmo.get(alg, 0.0) + tempo_total_alg_inst
            tempos_rows.append({
                "escopo": "instancia_algoritmo",
                "instancia": nome,
                "algoritmo": alg,
                "tempo_segundos": tempo_total_alg_inst,
                "n_runs": n_runs_alg,
            })
    for alg, t in tempo_por_algoritmo.items():
        tempos_rows.append({
            "escopo": "algoritmo", "instancia": "", "algoritmo": alg,
            "tempo_segundos": t, "n_runs": "",
        })

    _escrever_csv(
        os.path.join(out_dir, "resultados_tempos.csv"),
        tempos_rows,
        fieldnames=["escopo", "instancia", "algoritmo", "tempo_segundos", "n_runs"],
    )

    print(f"\nResultados salvos em: {out_dir}")
    print(f"Figuras salvas em:    {fig_dir}")
    print(f"\nTempo total do experimento: {_formatar_duracao(tempo_total)}")
    for nome, t in tempo_por_instancia.items():
        print(f"  {nome:>10s}: {_formatar_duracao(t)}")

    return {
        "runs_raw": runs_raw_rows,
        "resumo": resumo_rows,
        "convergencia": convergencia_rows,
        "estatisticas": estatisticas_rows,
        "tempos": tempos_rows,
    }


if __name__ == "__main__":
    comparar_multi_instancia()
