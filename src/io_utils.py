"""Leitura dos CSVs do InfoSiga com Polars (todas as colunas como texto)."""
import io
from pathlib import Path
import polars as pl
from . import config


def ler_csv_infosiga(caminho) -> pl.DataFrame:
    """Lê um CSV do InfoSiga (separador ';', codificação latin-1) sem alterar valores."""
    caminho = Path(caminho)
    texto = caminho.read_bytes().decode("latin-1")
    df = pl.read_csv(io.BytesIO(texto.encode("utf-8")), separator=";",
                     infer_schema=False, truncate_ragged_lines=True)
    return df.with_columns(pl.lit(caminho.name).alias("arquivo_origem"))


def listar_arquivos(prefixo: str) -> list[Path]:
    return sorted(config.DATA_RAW.glob(f"{prefixo}_*.csv"))


def carregar_tabela(prefixo: str) -> pl.DataFrame:
    arquivos = listar_arquivos(prefixo)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum '{prefixo}_*.csv' em {config.DATA_RAW}")
    partes = []
    for a in arquivos:
        df = ler_csv_infosiga(a)
        print(f"  {a.name:<28} {df.height:>9,} linhas × {df.width} colunas")
        partes.append(df)
    return pl.concat(partes, how="vertical_relaxed")
