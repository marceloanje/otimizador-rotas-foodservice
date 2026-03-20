import time
import numpy as np
import csv
import statistics
import os
import random

from modelos.instancia import Instancia
from algoritmos.colonia_formigas import ACO
from algoritmos.busca_tabu import BuscaTabu
from algoritmos.enxame_particulas import PSO

def executar_algoritmo(name, solver_cls, instancia, runs=10, seed_base=42):
    results = []
    for r in range(runs):
        random.seed(seed_base + r)
        np.random.seed(seed_base + r)

        solver = solver_cls(instancia)
        t0 = time.time()
        sol = solver.run()
        t1 = time.time()
        tempo = t1 - t0
        
        # Garantir que a solução foi avaliada
        if sol.custo_objetivo is None:
            sol.avaliar(instancia)
        
        # Verificar se é válida
        eh_valida = sol.eh_valida(instancia)
        n_violacoes = sol.violacoes.get("capacidade", 0)

        results.append({
            "run": r,
            "rotas": getattr(sol, "rotas", []),
            "custo": sol.custo,
            "custo_objetivo": sol.custo_objetivo,
            "n_veiculos": sol.n_veiculos,
            "eh_valida": eh_valida,
            "n_violacoes": n_violacoes,
            "tempo": tempo,
            "meta": getattr(sol, "meta", {})
        })
    return results

def resumir_results(results):
    custos = [r["custo"] for r in results]
    custos_objetivo = [r["custo_objetivo"] for r in results]
    tempos = [r["tempo"] for r in results]
    n_veiculos_list = [r["n_veiculos"] for r in results]
    validas = [r["eh_valida"] for r in results]
    violacoes = [r["n_violacoes"] for r in results]
    
    taxa_factivel = sum(validas) / len(validas) if validas else 0.0
    
    return {
        "melhor_custo": float(min(custos)),
        "media_custo": float(statistics.mean(custos)),
        "desvio_custo": float(statistics.pstdev(custos)),
        "melhor_objetivo": float(min(custos_objetivo)),
        "media_objetivo": float(statistics.mean(custos_objetivo)),
        "desvio_objetivo": float(statistics.pstdev(custos_objetivo)),
        "media_veiculos": float(statistics.mean(n_veiculos_list)),
        "taxa_factivel": float(taxa_factivel),
        "media_violacoes": float(statistics.mean(violacoes)),
        "tempo_med": float(statistics.mean(tempos))
    }

def comparar(instancia_path=None, runs=10, output_csv=None):
    # Resolver caminhos relativos baseados na localização deste arquivo
    if instancia_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        instancia_path = os.path.join(script_dir, "dados", "pedidos.csv")
    
    if output_csv is None:
        # Salvar no diretório raiz do projeto
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_csv = os.path.join(project_root, "resultados_comparacao.csv")
    
    print("Carregando instancia:", instancia_path)
    instancia = Instancia.do_csv(instancia_path)

    algs = [
        ("ACO", ACO),
        ("Tabu", BuscaTabu),
        ("PSO", PSO)
    ]

    summary = []

    for name, cls in algs:
        print(f"\nExecutando {name} por {runs} rodadas...")
        results = executar_algoritmo(name, cls, instancia, runs=runs)
        s = resumir_results(results)
        s["algoritmo"] = name
        summary.append(s)
        print(f"{name} — Melhor custo: {s['melhor_custo']:.2f}, Melhor objetivo: {s['melhor_objetivo']:.2f}")
        print(f"  Média custo: {s['media_custo']:.2f} (±{s['desvio_custo']:.2f})")
        print(f"  Média objetivo: {s['media_objetivo']:.2f} (±{s['desvio_objetivo']:.2f})")
        print(f"  Média veículos: {s['media_veiculos']:.1f}")
        print(f"  Taxa factível: {s['taxa_factivel']*100:.1f}% | Violações médias: {s['media_violacoes']:.1f}")
        print(f"  Tempo médio: {s['tempo_med']:.4f}s")

    keys = ["algoritmo", "melhor_custo", "media_custo", "desvio_custo", 
            "melhor_objetivo", "media_objetivo", "desvio_objetivo",
            "media_veiculos", "taxa_factivel", "media_violacoes", "tempo_med"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row[k] for k in keys})

    print("\nResumo salvo em:", os.path.abspath(output_csv))
    return summary

if __name__ == "__main__":
    comparar()
