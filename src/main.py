import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comparador import comparar_multi_instancia
from config_experimento import INSTANCIAS


def main():
    parser = argparse.ArgumentParser(description="Otimizador de Rotas Foodservice (CVRP)")
    parser.add_argument(
        "--instancia",
        choices=[i["nome"] for i in INSTANCIAS],
        default=None,
        metavar="NOME",
        help=f"Instância a executar ({', '.join(i['nome'] for i in INSTANCIAS)}). "
             "Padrão: todas.",
    )
    args = parser.parse_args()

    if args.instancia is not None:
        instancias_filtradas = [i for i in INSTANCIAS if i["nome"] == args.instancia]
    else:
        instancias_filtradas = None

    comparar_multi_instancia(instancias=instancias_filtradas)


if __name__ == "__main__":
    main()
