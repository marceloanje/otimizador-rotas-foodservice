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

from config import CAPACIDADE_CAMINHAO, NUMERO_CAMINHOES
from config_experimento import INSTANCIAS, N_RUNS, SEED_BASE
from modelos.instancia import Instancia
from algoritmos.colonia_formigas import ACO
from algoritmos.busca_tabu import BuscaTabu
from algoritmos.enxame_particulas import PSO
from utilitarios.graficos_experimento import (
    plotar_convergencia,
    plotar_boxplot_custos,
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
)


ALGORITMOS_HEURISTICOS = [
    ("ACO", ACO),
    ("Tabu", BuscaTabu),
    ("PSO", PSO),
]


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


def comparar_multi_instancia():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_dir = os.path.join(project_root, "resultados")
    fig_dir = os.path.join(out_dir, "figuras")
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)
    sol_dir = os.path.join(out_dir, "solucoes")
    _ensure_dir(sol_dir)

    runs_raw_rows = []
    resumo_rows = []
    convergencia_rows = []
    estatisticas_rows = []

    tempos_por_inst_alg = {}

    for inst_cfg in INSTANCIAS:
        nome = inst_cfg["nome"]
        path = inst_cfg["path"]
        usar_exato = inst_cfg.get("usar_exato", False)
        tempo_limite_exato = inst_cfg.get("tempo_limite_exato", 60)

        if not os.path.exists(path):
            print(f"[AVISO] Instância '{nome}' não encontrada em {path}. Pulando.")
            continue

        print(f"\n{'='*60}\nInstância: {nome} ({path})\n{'='*60}")
        instancia = Instancia.do_csv(path, capacidade_caminhao=CAPACIDADE_CAMINHAO)
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

        print(f"\n  Gerando figuras da instância '{nome}'...")
        convergencia_por_alg = {
            alg: [r["historico"] for r in results if r["historico"]]
            for alg, results in runs_heur.items()
        }
        plotar_convergencia(
            convergencia_por_alg, nome,
            os.path.join(fig_dir, f"convergencia_{nome}.png"),
        )

        custos_finais = {
            alg: [r["custo_objetivo"] for r in results]
            for alg, results in runs_heur.items()
        }
        plotar_boxplot_custos(
            custos_finais, nome,
            os.path.join(fig_dir, f"boxplot_{nome}.png"),
        )

        todos_objetivos = [r["custo_objetivo"] for res in runs_por_alg.values() for r in res]
        alvo_custo = 1.05 * min(todos_objetivos) if todos_objetivos else None

        plotar_convergencia_tempo(
            convergencia_por_alg, nome,
            os.path.join(fig_dir, f"convergencia_tempo_{nome}.png"),
        )
        if alvo_custo is not None:
            plotar_time_to_target(
                convergencia_por_alg, alvo_custo, nome,
                os.path.join(fig_dir, f"time_to_target_{nome}.png"),
            )
        plotar_violino_custos(
            custos_finais, nome,
            os.path.join(fig_dir, f"violino_{nome}.png"),
        )
        tempos_finais = {
            alg: [r["tempo"] for r in results]
            for alg, results in runs_heur.items()
        }
        plotar_scatter_custo_tempo(
            tempos_finais, custos_finais, nome,
            os.path.join(fig_dir, f"scatter_custo_tempo_{nome}.png"),
        )
        plotar_heatmap_pvalores(
            estatisticas_inst, nome,
            os.path.join(fig_dir, f"pvalores_{nome}.png"),
        )
        for alg_n in runs_por_alg:
            pkl_path = os.path.join(sol_dir, f"{nome}__{alg_n}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as _f:
                    melhor_sol = pickle.load(_f)
                plotar_gantt_rota(
                    melhor_sol, instancia,
                    os.path.join(fig_dir, f"gantt_{nome}__{alg_n}.png"),
                )
                plotar_utilizacao_capacidade(
                    melhor_sol, instancia,
                    os.path.join(fig_dir, f"utilizacao_{nome}__{alg_n}.png"),
                )

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

    plotar_tempo_execucao(
        tempos_por_inst_alg,
        os.path.join(fig_dir, "tempos.png"),
    )
    plotar_heatmap_gap(resumo_rows, os.path.join(fig_dir, "heatmap_gap.png"))
    plotar_stacked_veiculos(resumo_rows, os.path.join(fig_dir, "stacked_veiculos.png"))

    print(f"\nResultados salvos em: {out_dir}")
    print(f"Figuras salvas em:    {fig_dir}")

    return {
        "runs_raw": runs_raw_rows,
        "resumo": resumo_rows,
        "convergencia": convergencia_rows,
        "estatisticas": estatisticas_rows,
    }


if __name__ == "__main__":
    comparar_multi_instancia()
