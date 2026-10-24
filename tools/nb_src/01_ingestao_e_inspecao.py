# %% [markdown]
# # 01 · Ingestão e inspeção dos dados brutos (InfoSiga SP)
#
# **Tema do TCC:** Análise comparativa dos acidentes de trânsito nas rodovias e vias urbanas de
# Campinas e Guarulhos: padrões e fatores de risco no InfoSiga SP (2022–2025).
#
# **Objetivo deste notebook:** ler os CSVs originais com **Polars**, entender a estrutura das três
# tabelas e salvar uma cópia fiel em Parquet (`data/interim/`) **sem alterar nenhum valor**.
#
# | Etapa | O que faz |
# |---|---|
# | 1 | Lê `sinistros`, `pessoas` e `veiculos` (2022–2024 e 2025–2026) de `data/raw/` |
# | 2 | Mostra colunas, tipos e exemplos |
# | 3 | Avalia qualidade: nulos e valores únicos |
# | 4 | Inspeciona as variáveis do recorte (município, tipo de via, tipo de registro) |
# | 5 | Verifica a chave `id_sinistro` e a cobertura temporal |
# | 6 | Gráficos de inspeção e gravação em `data/interim/` |
#
# > `data/raw/` é somente leitura. Toda transformação acontece a partir do notebook 02.

# %%
import sys
from pathlib import Path
RAIZ = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(RAIZ))

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from src import config, io_utils, limpeza, viz

viz.tema()
pl.Config.set_tbl_rows(25); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(40)
print("Raiz do projeto:", config.RAIZ)
print("Arquivos em data/raw:", [p.name for p in sorted(config.DATA_RAW.glob("*.csv"))])

# %% [markdown]
# ## 1. Leitura dos arquivos
# O InfoSiga distribui três tabelas ligadas pela chave `id_sinistro`. Os arquivos vêm em
# codificação **latin-1** e separador `;`. Tudo é lido como texto (`infer_schema=False`) para não
# perder zeros à esquerda nem converter valores silenciosamente.

# %%
print("SINISTROS"); sin = io_utils.carregar_tabela("sinistros")
print("PESSOAS");   pes = io_utils.carregar_tabela("pessoas")
print("VEÍCULOS");  vei = io_utils.carregar_tabela("veiculos")

dimensoes = pl.DataFrame({
    "tabela": ["sinistros", "pessoas", "veiculos"],
    "linhas": [sin.height, pes.height, vei.height],
    "colunas": [sin.width, pes.width, vei.width],
})
dimensoes

# %% [markdown]
# ## 2. Estrutura das tabelas

# %%
sin.head(3).transpose(include_header=True, header_name="coluna")

# %%
for nome, df in [("SINISTROS", sin), ("PESSOAS", pes), ("VEÍCULOS", vei)]:
    print(f"{nome} ({df.width} colunas):\n  {df.columns}\n")

# %% [markdown]
# ## 3. Qualidade: nulos, valores únicos e exemplos
# `pct_nulos` alto em colunas de flag (`tp_sinistro_*`, `qtd_*`) é esperado: o InfoSiga deixa a
# célula vazia quando a característica não ocorre.

# %%
q_sin = limpeza.resumo_qualidade(sin)
viz.salvar_tab(q_sin.to_pandas(), "01_qualidade_sinistros")
q_sin

# %%
q_pes = limpeza.resumo_qualidade(pes)
viz.salvar_tab(q_pes.to_pandas(), "01_qualidade_pessoas")
q_pes

# %% [markdown]
# ## 4. Variáveis-chave do recorte
# Conferimos como o InfoSiga escreve os valores que vamos filtrar no notebook 02.

# %%
print("Tipos de via (estado inteiro):")
display(sin.group_by("tipo_via").len().sort("len", descending=True))
print("Tipos de registro (estado inteiro):")
display(sin.group_by("tipo_registro").len().sort("len", descending=True))
print("Campinas e Guarulhos:")
display(sin.filter(pl.col("municipio").is_in(config.MUNICIPIOS))
           .group_by("municipio", "tipo_via").len().sort("municipio", "tipo_via"))

# %% [markdown]
# ## 5. Chave primária e cobertura temporal

# %%
dup = sin.height - sin["id_sinistro"].n_unique()
print(f"id_sinistro duplicados na tabela de sinistros: {dup:,}")
sem_par_p = pes.filter(~pl.col("id_sinistro").is_in(sin["id_sinistro"].implode())).height
sem_par_v = vei.filter(~pl.col("id_sinistro").is_in(sin["id_sinistro"].implode())).height
print(f"Pessoas sem sinistro correspondente: {sem_par_p:,} | Veículos sem sinistro: {sem_par_v:,}")

datas = sin.select(pl.col("data_sinistro").str.strptime(pl.Date, "%d/%m/%Y", strict=False))
print("Período:", datas.min().item(), "→", datas.max().item())

cg = sin.filter(pl.col("municipio").is_in(config.MUNICIPIOS))
por_ano = (cg.group_by("ano_sinistro", "municipio", "tipo_registro").len()
             .sort("ano_sinistro", "municipio", "tipo_registro"))
viz.salvar_tab(por_ano.to_pandas(), "01_registros_por_ano")
print("Último mês disponível em 2026:",
      cg.filter(pl.col("ano_sinistro") == "2026")["mes_sinistro"].max(),
      "→ 2026 é parcial e fica fora do recorte (2022–2025).")

# %% [markdown]
# ## 6. Gráficos de inspeção
# Registros por ano e tipo de registro nos dois municípios. Note que `NOTIFICACAO` (ocorrência sem
# vítima) muda de patamar ao longo dos anos — por isso a análise usa **somente sinistros com vítima**.

# %%
d = por_ano.to_pandas()
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
cores_reg = {"NOTIFICACAO": "#bdbdbd", "SINISTRO NAO FATAL": "#fdae6b", "SINISTRO FATAL": "#7f0000"}
for ax, mun in zip(axes, config.MUNICIPIOS):
    t = (d[d.municipio == mun].pivot(index="ano_sinistro", columns="tipo_registro", values="len")
           .fillna(0)[list(cores_reg)])
    t.plot(kind="bar", stacked=True, ax=ax, color=cores_reg.values(), width=0.75)
    ax.set(title=f"{mun.title()} – registros por ano", xlabel="Ano", ylabel="Registros")
    ax.tick_params(axis="x", rotation=0); ax.legend(title="Tipo de registro", fontsize=9)
fig.suptitle("InfoSiga – registros por ano e tipo (2022–2026*)", fontweight="bold", y=1.02)
plt.tight_layout(); viz.salvar_fig(fig, "01_registros_por_ano"); plt.show()

# %%
falt = (q_sin.filter(pl.col("pct_nulos") > 0).sort("pct_nulos").to_pandas())
fig, ax = plt.subplots(figsize=(9, 0.28 * len(falt) + 1.5))
ax.barh(falt.coluna, falt.pct_nulos, color="#1f77b4")
ax.set(title="% de valores vazios por coluna – sinistros (estado de SP)", xlabel="% vazio", ylabel="")
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout(); viz.salvar_fig(fig, "01_faltantes_sinistros"); plt.show()

# %% [markdown]
# ## 7. Gravação em `data/interim/` (Parquet, valores originais)

# %%
sin.write_parquet(config.DATA_INTERIM / "sinistros_bruto.parquet")
pes.write_parquet(config.DATA_INTERIM / "pessoas_bruto.parquet")
vei.write_parquet(config.DATA_INTERIM / "veiculos_bruto.parquet")
print("✔ Salvos:", sorted(p.name for p in config.DATA_INTERIM.glob("*.parquet")))
