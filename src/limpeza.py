"""Limpeza e criação de variáveis com Polars.

Cada função recebe/retorna pl.DataFrame. Nenhum valor é imputado: o que não
existe no InfoSiga fica nulo ou na categoria 'SEM VÍTIMA / N.D.'.
"""
import re
import unicodedata
import polars as pl
from . import config

DIAS = {1: "Seg", 2: "Ter", 3: "Qua", 4: "Qui", 5: "Sex", 6: "Sáb", 7: "Dom"}


def sem_acento(s: str | None) -> str | None:
    if s is None:
        return None
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _texto(col: str) -> pl.Expr:
    return (pl.col(col).str.strip_chars().str.to_uppercase()
            .str.replace_all(r"\s+", " "))


def _num(col: str) -> pl.Expr:
    return (pl.col(col).str.strip_chars().str.replace(",", ".", literal=True)
            .cast(pl.Float64, strict=False))


def _qtd(col: str) -> pl.Expr:
    return _num(col).fill_null(0)


# ── Nomes de vias ─────────────────────────────────────────────────────────────
_RE_ROD = re.compile(r"^(SPA|SPI|SP|BR)\s*-?\s*0*(\d{1,3})")
# Erros de digitação óbvios observados no top de logradouros (mesma via, grafias diferentes)
ALIAS_URBANO = {
    "AV. JOHN BOID DUNLOP": "AV. JOHN BOYD DUNLOP",
    "AV. RUY RODRIGUEZ": "AV. RUI RODRIGUES",
    "AV. RUY RODRIGUES": "AV. RUI RODRIGUES",
    "AV. RUI RODRIGUEZ": "AV. RUI RODRIGUES",
    "ESTR. PRESIDENTE JUSCELINO KUBITSCHECK DE OLIVEI": "ESTR. PRES. JUSCELINO K. DE OLIVEIRA",
    "ESTR. PRESIDENTE JUSCELINO KUBITSCHEK DE OLIVEIRA": "ESTR. PRES. JUSCELINO K. DE OLIVEIRA",
    "ESTR. PRESIDENTE JUSCELINO KUBITSCHECK DE OLIVEIRA": "ESTR. PRES. JUSCELINO K. DE OLIVEIRA",
}


def nome_rodovia(logr: str | None) -> str | None:
    if not logr:
        return None
    m = _RE_ROD.match(sem_acento(logr).upper().strip())
    if not m:
        return "OUTRA / N.D."
    cod = f"{m.group(1)}-{int(m.group(2)):03d}"
    return config.NOMES_RODOVIAS.get(cod, cod)


def nome_urbano(logr: str | None) -> str | None:
    if not logr:
        return None
    s = re.sub(r"\s+", " ", sem_acento(logr).upper().strip())
    for longo, curto in (("AVENIDA ", "AV. "), ("RUA ", "R. "), ("ESTRADA ", "ESTR. "),
                         ("PRACA ", "PC. "), ("ALAMEDA ", "AL. "), ("TRAVESSA ", "TV. ")):
        if s.startswith(longo):
            s = curto + s[len(longo):]
            break
    return ALIAS_URBANO.get(s, s)


# ── Sinistros ─────────────────────────────────────────────────────────────────
def preparar_sinistros(sin: pl.DataFrame) -> tuple[pl.DataFrame, list[dict]]:
    """Aplica o funil de seleção e cria as variáveis analíticas. Retorna (df, funil)."""
    funil = []

    def reg(etapa, df):
        funil.append({"etapa": etapa, "linhas": df.height})
        print(f"  {etapa:<52} {df.height:>9,}")

    df = sin.with_columns(
        _texto("municipio").alias("municipio"),
        _texto("tipo_registro").alias("tipo_registro"),
        _texto("tipo_via").alias("tipo_via"),
    )
    reg("Bruto (2022–2026, todos os municípios)", df)
    df = df.unique(subset="id_sinistro", keep="first", maintain_order=True)
    reg("Sem id_sinistro duplicado", df)
    df = df.filter(pl.col("municipio").is_in(config.MUNICIPIOS))
    reg("Campinas + Guarulhos", df)
    df = df.filter(pl.col("tipo_registro").is_in(config.REGISTROS_COM_VITIMA))
    reg("Somente sinistros com vítima (exclui NOTIFICACAO)", df)
    df = df.with_columns(pl.col("tipo_via").replace_strict(config.GRUPO_VIA, default=None)
                         .alias("grupo_via")).filter(pl.col("grupo_via").is_not_null())
    reg("Tipo de via = rodovia ou via urbana", df)

    df = df.with_columns(pl.col("data_sinistro").str.strptime(pl.Date, "%d/%m/%Y", strict=False)
                         .alias("data"))
    df = df.filter(pl.col("data").is_not_null() & pl.col("data").dt.year().is_in(config.ANOS))
    reg("Período 2022–2025 com data válida", df)

    hora = pl.col("hora_sinistro").str.extract(r"^(\d{1,2})", 1).cast(pl.Int8, strict=False)
    df = df.with_columns(
        pl.col("data").dt.year().alias("ano"),
        pl.col("data").dt.month().alias("mes"),
        pl.col("data").dt.strftime("%Y-%m").alias("ano_mes"),
        pl.col("data").dt.weekday().alias("dia_sem_n"),            # 1=seg … 7=dom
        pl.when(hora.is_between(0, 23)).then(hora).alias("hora_int"),
    ).with_columns(
        pl.col("dia_sem_n").replace_strict(DIAS).alias("dia_semana"),
        (pl.col("dia_sem_n") >= 6).alias("fim_semana"),
        pl.when(pl.col("hora_int").is_between(0, 5)).then(pl.lit("Madrugada"))
          .when(pl.col("hora_int").is_between(6, 11)).then(pl.lit("Manhã"))
          .when(pl.col("hora_int").is_between(12, 17)).then(pl.lit("Tarde"))
          .when(pl.col("hora_int").is_between(18, 23)).then(pl.lit("Noite"))
          .alias("periodo"),
    )

    # Quantidades e gravidade (FATAL > GRAVE > LEVE > sem vítima/N.D.)
    qtd_cols = [c for c in df.columns if c.startswith("qtd_")]
    df = df.with_columns([_qtd(c).alias(c) for c in qtd_cols])
    fatal = (pl.col("qtd_gravidade_fatal") > 0) | (pl.col("tipo_registro") == "SINISTRO FATAL")
    df = df.with_columns(
        pl.when(fatal).then(pl.lit("FATAL"))
          .when(pl.col("qtd_gravidade_grave") > 0).then(pl.lit("GRAVE"))
          .when(pl.col("qtd_gravidade_leve") > 0).then(pl.lit("LEVE"))
          .otherwise(pl.lit("SEM VÍTIMA / N.D.")).alias("gravidade"),
    ).with_columns(
        pl.col("gravidade").is_in(["GRAVE", "FATAL"]).alias("grave_fatal"),
        (pl.col("gravidade") == "FATAL").alias("fatal"),
    )

    # Flags de veículos/modos e de tipo de sinistro
    df = df.with_columns(
        [pl.any_horizontal([pl.col(c) > 0 for c in cols]).alias(nome)
         for nome, cols in config.FLAGS_VEICULO.items()]
        + [pl.any_horizontal([_texto(c) == "S" for c in cols]).fill_null(False).alias(nome)
           for nome, cols in config.FLAGS_TIPO.items()]
        + [_texto("tp_sinistro_primario").alias("tipo_sinistro")]
    )
    # Tipo detalhado exclusivo (prioridade: o mais grave/específico primeiro)
    expr = pl.lit("OUTROS / N.D.")
    for nome in reversed(["tp_atropelamento", "tp_capotamento", "tp_tombamento",
                          "tp_colisao_frontal", "tp_colisao_transv", "tp_colisao_lateral",
                          "tp_engavetamento", "tp_choque"]):
        expr = pl.when(pl.col(nome)).then(pl.lit(config.ROTULOS_TIPO[nome])).otherwise(expr)
    df = df.with_columns(expr.alias("tipo_detalhado"))

    # Coordenadas
    df = df.with_columns(_num("latitude").alias("latitude"), _num("longitude").alias("longitude"))
    valida = pl.lit(False)
    for mun, (la0, la1, lo0, lo1) in config.CAIXAS.items():
        valida = valida | ((pl.col("municipio") == mun)
                           & pl.col("latitude").is_between(la0, la1)
                           & pl.col("longitude").is_between(lo0, lo1))
    df = df.with_columns(valida.fill_null(False).alias("coord_valida"))

    # Nome padronizado da via e km (rodovias)
    df = df.with_columns(
        pl.when(pl.col("grupo_via") == "RODOVIA")
          .then(pl.col("logradouro").map_elements(nome_rodovia, return_dtype=pl.String))
          .otherwise(pl.col("logradouro").map_elements(nome_urbano, return_dtype=pl.String))
          .fill_null("N.D.").alias("via_nome"),
        pl.when(pl.col("grupo_via") == "RODOVIA").then(_num("numero_logradouro"))
          .alias("km"),
    )
    return df, funil


# ── Pessoas e veículos ────────────────────────────────────────────────────────
def categoria_vitima() -> pl.Expr:
    tv, vv = _texto("tipo_de_vitima"), _texto("tipo_veiculo_vitima")
    return (pl.when(tv == "PEDESTRE").then(pl.lit("PEDESTRE"))
              .when(vv == "MOTOCICLETA").then(pl.lit("MOTOCICLISTA"))
              .when(vv == "BICICLETA").then(pl.lit("CICLISTA"))
              .when(vv == "AUTOMOVEL").then(pl.lit("OCUPANTE AUTOMÓVEL"))
              .when(vv.is_in(["CAMINHAO", "ONIBUS"])).then(pl.lit("OCUPANTE VEÍC. PESADO"))
              .otherwise(pl.lit("OUTROS / N.D.")))


def preparar_pessoas(pes: pl.DataFrame, sin: pl.DataFrame) -> pl.DataFrame:
    chave = sin.select("id_sinistro", "municipio", "grupo_via", "ano", "gravidade")
    return (pes.drop("municipio", "tipo_via", strict=False)
               .join(chave.rename({"gravidade": "gravidade_sinistro"}), on="id_sinistro", how="inner")
               .with_columns(
                   _texto("tipo_de_vitima").alias("tipo_vitima"),
                   _texto("tipo_veiculo_vitima").fill_null("NAO DISPONIVEL").alias("veiculo_vitima"),
                   _texto("gravidade_lesao").fill_null("NAO DISPONIVEL").alias("gravidade_lesao"),
                   _texto("sexo").alias("sexo"),
                   _texto("faixa_etaria_demografica").alias("faixa_etaria"),
                   pl.col("idade").cast(pl.Int16, strict=False).alias("idade"),
                   categoria_vitima().alias("categoria_vitima"),
               ))


def preparar_veiculos(vei: pl.DataFrame, sin: pl.DataFrame) -> pl.DataFrame:
    chave = sin.select("id_sinistro", "municipio", "grupo_via", "ano", "gravidade")
    return (vei.join(chave, on="id_sinistro", how="inner")
               .with_columns(_texto("tipo_veiculo").fill_null("NAO DISPONIVEL").alias("tipo_veiculo")))


def resumo_qualidade(df: pl.DataFrame) -> pl.DataFrame:
    n = df.height
    return pl.DataFrame({
        "coluna": df.columns,
        "nulos": [df[c].null_count() for c in df.columns],
        "pct_nulos": [round(df[c].null_count() / n * 100, 2) for c in df.columns],
        "unicos": [df[c].n_unique() for c in df.columns],
        "exemplo": [str(df[c].drop_nulls().head(1).to_list()) for c in df.columns],
    })
