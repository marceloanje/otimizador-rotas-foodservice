import os
import pandas as pd


def limpar_pedidos(path_entrada: str, path_saida: str | None = None) -> pd.DataFrame:
    """Lê, limpa e normaliza o CSV de pedidos.

    Etapas:
    1. Leitura do CSV.
    2. Remoção de colunas totalmente vazias (artefato de exportação).
    3. Deduplicação por número de pedido.
    4. Descarte de linhas com demanda (valor_total) nula ou negativa.
    5. Normalização: valor_total convertido para float.
    6. Se path_saida fornecido, salva o resultado em CSV.

    Parâmetros
    ----------
    path_entrada : str
        Caminho para o CSV bruto de pedidos.
    path_saida : str | None
        Caminho destino para salvar o CSV limpo. Diretório criado automaticamente
        se não existir. Se None, nenhum arquivo é gerado.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpo, sem duplicatas e com demandas válidas.
    """
    df = pd.read_csv(path_entrada)
    n_original = len(df)

    # 1. Remover coluna vazia no final (artefato de exportação CSV)
    ultima = df.columns[-1]
    if (isinstance(ultima, str) and ultima.strip() == "") or df[ultima].isna().all():
        df = df.drop(columns=[ultima])

    # 2. Deduplicação por número de pedido
    if "pedido" in df.columns:
        n_antes = len(df)
        df = df.drop_duplicates(subset=["pedido"])
        n_dup = n_antes - len(df)
        if n_dup > 0:
            print(f"Limpeza: {n_dup} linha(s) duplicada(s) removida(s) (campo 'pedido').")

    # 3. Descarte de linhas com demanda inválida
    if "valor_total" in df.columns:
        df["valor_total"] = pd.to_numeric(df["valor_total"], errors="coerce")
        n_antes = len(df)
        df = df[df["valor_total"].notna() & (df["valor_total"] >= 0)]
        n_inv = n_antes - len(df)
        if n_inv > 0:
            print(f"Limpeza: {n_inv} linha(s) descartada(s) por demanda nula ou negativa.")

    df = df.reset_index(drop=True)

    n_final = len(df)
    n_removidos = n_original - n_final
    if n_removidos > 0:
        print(f"Limpeza concluída: {n_original} → {n_final} linhas ({n_removidos} removida(s)).")
    else:
        print(f"Limpeza concluída: {n_final} linhas, nenhuma removida.")

    if path_saida is not None:
        os.makedirs(os.path.dirname(path_saida), exist_ok=True)
        df.to_csv(path_saida, index=False)
        print(f"Dados limpos salvos em: {path_saida}")

    return df
