# %% [markdown]
# # 02 · Limpeza, filtragem e criação de variáveis
#
# Transforma os dados brutos na **base analítica**: sinistros **com vítima** em **rodovias e vias
# urbanas** de **Campinas e Guarulhos**, **2022–2025**. Tudo em **Polars**.
#
# | Etapa | O que faz |
# |---|---|
# | 1 | Funil de seleção (cada filtro registra quantas linhas restam) |
# | 2 | Variáveis temporais: ano, mês, dia da semana, fim de semana, hora e período |
# | 3 | Gravidade do sinistro a partir de `qtd_gravidade_*` |
# | 4 | Flags de veículos/modos (moto, bicicleta, pedestre, automóvel, caminhão, ônibus, pesado) |
# | 5 | Flags de tipo de sinistro (atropelamento, capotamento, tombamento, colisões…) |
# | 6 | Nome padronizado de rodovias/vias e km; validação de coordenadas |
# | 7 | Pessoas e veículos vinculados aos sinistros selecionados |
# | 8 | Checagens de consistência e gravação em `data/processed/` |
#
# **Correções em relação à versão anterior do projeto:** as flags de moto/caminhão/bicicleta eram
# procuradas em colunas `tp_veiculo_*` que não existem no InfoSiga (ficavam sempre falsas); a
# gravidade procurava `gravidade_fatal` (o nome real é `qtd_gravidade_fatal`); e capotamento não
# existe em `tp_sinistro_primario` — ele vem da flag `tp_sinistro_capotamento`.

# %%
import sys
from pathlib import Path
RAIZ = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(RAIZ))

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from src import config, limpeza, viz

viz.tema()
pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(45)
sin_b = pl.read_parquet(config.DATA_INTERIM / "sinistros_bruto.parquet")
pes_b = pl.read_parquet(config.DATA_INTERIM / "pessoas_bruto.parquet")
vei_b = pl.read_parquet(config.DATA_INTERIM / "veiculos_bruto.parquet")

# %% [markdown]
# ## 1–6. Funil de seleção e variáveis derivadas
# A função `limpeza.preparar_sinistros` (em `src/limpeza.py`) executa, em ordem:
# remoção de `id_sinistro` duplicado → municípios → **somente `SINISTRO FATAL` e
# `SINISTRO NAO FATAL`** → tipo de via (`ESTRADAS E RODOVIAS` → RODOVIA, `VIAS URBANAS` → VIA URBANA;
# `NAO DISPONIVEL` sai) → anos 2022–2025. Depois cria as variáveis.

# %%
sin, funil = limpeza.preparar_sinistros(sin_b)
print(f"\nBase analítica: {sin.height:,} sinistros")

# %%
print("Sinistros por município × tipo de via × ano:")
tab = (sin.group_by("municipio", "grupo_via", "ano").len()
          .pivot(on="ano", index=["municipio", "grupo_via"], values="len", sort_columns=True)
          .sort("municipio", "grupo_via"))
viz.salvar_tab(tab.to_pandas(), "02_base_municipio_via_ano")
tab

# %% [markdown]
# ### Gravidade do sinistro
# Regra (hierárquica): **FATAL** se `qtd_gravidade_fatal > 0` ou registro `SINISTRO FATAL`;
# senão **GRAVE** se há ferido grave; senão **LEVE** se há ferido leve; senão
# **SEM VÍTIMA / N.D.** (vítimas com lesão "não disponível" ou só ilesos informados).

# %%
grav = (sin.group_by("municipio", "grupo_via", "gravidade").len()
           .pivot(on="gravidade", index=["municipio", "grupo_via"], values="len")
           .select("municipio", "grupo_via", *config.ORDEM_GRAVIDADE).sort("municipio", "grupo_via"))
grav

# %% [markdown]
# ### Flags de veículos e tipos de sinistro
# Cada flag indica se o sinistro **envolveu** aquele modo/tipo (um sinistro pode ter várias).

# %%
flags = list(config.FLAGS_VEICULO) + list(config.FLAGS_TIPO)
cont_flags = (sin.group_by("municipio", "grupo_via")
                 .agg([pl.col(f).sum() for f in flags]).sort("municipio", "grupo_via"))
viz.salvar_tab(cont_flags.to_pandas(), "02_contagem_flags")
cont_flags.transpose(include_header=True, header_name="flag",
                     column_names=[f"{m[:3]}-{v[:3]}" for m, v in cont_flags.select("municipio", "grupo_via").rows()])

# %%
print("Distribuição do tipo detalhado (exclusivo):")
sin.group_by("tipo_detalhado").len().sort("len", descending=True)

# %% [markdown]
# ### Nomes de vias e coordenadas
# Rodovias: o código (`SP 330`, `BR 116`…) é padronizado como `SP-330`; somente rodovias de nome
# inequívoco recebem o nome entre parênteses. Vias urbanas: acentos removidos, `AVENIDA` → `AV.`,
# e algumas grafias duplicadas óbvias unificadas (lista em `limpeza.ALIAS_URBANO`).

# %%
print("Rodovias encontradas:")
display(sin.filter(pl.col("grupo_via") == "RODOVIA").group_by("municipio", "via_nome").len()
           .sort("municipio", "len", descending=[False, True]))
print("Coordenadas válidas (dentro da caixa do município):")
sin.group_by("municipio", "grupo_via").agg((pl.col("coord_valida").mean() * 100).round(1)
                                           .alias("% coord válida")).sort("municipio", "grupo_via")

# %% [markdown]
# ## 7. Pessoas e veículos dos sinistros selecionados
# `categoria_vitima` combina o tipo de vítima e o veículo em que estava:
# PEDESTRE · MOTOCICLISTA · CICLISTA · OCUPANTE AUTOMÓVEL · OCUPANTE VEÍC. PESADO · OUTROS / N.D.

# %%
pes = limpeza.preparar_pessoas(pes_b, sin)
vei = limpeza.preparar_veiculos(vei_b, sin)
funil += [{"etapa": "Pessoas vinculadas aos sinistros selecionados", "linhas": pes.height},
          {"etapa": "Veículos vinculados aos sinistros selecionados", "linhas": vei.height}]
print(f"Pessoas: {pes.height:,} | Veículos: {vei.height:,}")
display(pes.group_by("categoria_vitima").len().sort("len", descending=True))
display(pes.group_by("gravidade_lesao").len().sort("len", descending=True))
vei.group_by("tipo_veiculo").len().sort("len", descending=True)

# %% [markdown]
# ## 8. Checagens de consistência
# - Nenhuma flag de veículo importante pode estar zerada (era o erro da versão anterior).
# - A soma por município × via × ano tem de bater com a contagem direta no CSV bruto.

# %%
for f in ["envolve_moto", "envolve_bicicleta", "envolve_pedestre", "envolve_pesado", "tp_capotamento"]:
    n = sin[f].sum()
    assert n > 0, f"flag {f} zerada"
    print(f"  {f:<20} {n:>6,} sinistros")

direto = (sin_b.filter(pl.col("municipio").is_in(config.MUNICIPIOS)
                       & pl.col("tipo_registro").is_in(config.REGISTROS_COM_VITIMA)
                       & pl.col("tipo_via").is_in(list(config.GRUPO_VIA))
                       & pl.col("ano_sinistro").is_in([str(a) for a in config.ANOS]))
                .unique("id_sinistro").height)
assert direto == sin.height, (direto, sin.height)
print(f"\n✔ Contagem direta no bruto = {direto:,} = base analítica")

# %% [markdown]
# ## 9. Funil e gravação

# %%
funil_df = pd.DataFrame(funil)
viz.salvar_tab(funil_df, "funil_selecao")
display(funil_df)

f = funil_df.iloc[:6]
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(f.etapa[::-1], f.linhas[::-1], color="#1f77b4")
ax.bar_label(ax.containers[0], labels=[f"{v:,}".replace(",", ".") for v in f.linhas[::-1]],
             padding=4, fontsize=9)
ax.set_xscale("log"); ax.set(title="Funil de seleção dos sinistros", xlabel="Linhas (escala log)")
plt.tight_layout(); viz.salvar_fig(fig, "02_funil_selecao"); plt.show()

# %%
sin.write_parquet(config.DATA_PROCESSED / "sinistros.parquet")
pes.write_parquet(config.DATA_PROCESSED / "pessoas.parquet")
vei.write_parquet(config.DATA_PROCESSED / "veiculos.parquet")
print(f"✔ Base final: {sin.height:,} sinistros | {pes.height:,} pessoas | {vei.height:,} veículos")
