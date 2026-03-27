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


def gerar_janelas_tempo(df, seed=42):
    """Gera janelas de tempo sintéticas para VRPTW.

    O DataFrame deve ter o depósito na linha 0 já inserido.
    Turnos disponíveis (em minutos desde meia-noite):
      - Manhã : [480, 720]  (08:00–12:00)
      - Tarde  : [780, 1020] (13:00–17:00)
    Tempo de serviço por cliente: 15–30 minutos (aleatório).

    Idempotente: retorna df inalterado se as colunas já existirem.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com depósito em df.iloc[0].
    seed : int
        Semente para reprodutibilidade.

    Retorna
    -------
    pd.DataFrame
        df com colunas 'janela_inicio', 'janela_fim' e 'tempo_servico'.
    """
    if "janela_inicio" in df.columns:
        return df

    import random as _rnd
    rng = _rnd.Random(seed)
    n = len(df)
    turnos = [(480, 720), (780, 1020)]

    janela_inicio = [0]    # depósito
    janela_fim    = [1440]  # depósito (fim do dia)
    tempo_servico = [0]    # depósito

    for _ in range(n - 1):
        ini, fim = rng.choice(turnos)
        janela_inicio.append(ini)
        janela_fim.append(fim)
        tempo_servico.append(rng.randint(15, 30))

    df = df.copy()
    df["janela_inicio"] = janela_inicio
    df["janela_fim"]    = janela_fim
    df["tempo_servico"] = tempo_servico
    return df
