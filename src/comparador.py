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

        results.append({
            "run": r,
            "rota": sol.rota,
            "custo": sol.custo,
            "tempo": tempo,
            "meta": getattr(sol, "meta", {})
        })
    return results

def resumir_results(results):
    custos = [r["custo"] for r in results]
    tempos = [r["tempo"] for r in results]
    return {
        "melhor": float(min(custos)),
        "media": float(statistics.mean(custos)),
        "desvio": float(statistics.pstdev(custos)),
        "tempo_med": float(statistics.mean(tempos))
    }

def comparar(instancia_path="src/dados/pedidos.csv", runs=10, output_csv="resultados_comparacao.csv"):
    print("Carregando instancia:", instancia_path)
    instancia = Instancia.from_csv(instancia_path)
    instancia.gerar_matriz_distancias_ficticia()

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
        print(f"{name} — Melhor: {s['melhor']:.4f}, Média: {s['media']:.4f}, Desvio: {s['desvio']:.4f}, Tempo médio: {s['tempo_med']:.4f}s")

    keys = ["algoritmo", "melhor", "media", "desvio", "tempo_med"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row[k] for k in keys})

    print("\nResumo salvo em:", os.path.abspath(output_csv))
    return summary

if __name__ == "__main__":
    comparar()
