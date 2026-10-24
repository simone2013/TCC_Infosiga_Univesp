# TCC – Acidentes de trânsito em Campinas e Guarulhos (InfoSiga SP, 2022–2025)

**Tema:** Análise comparativa dos acidentes de trânsito nas rodovias e vias urbanas de Campinas e
Guarulhos: padrões e fatores de risco no InfoSiga SP (2022–2025).

Versão 3: refeita **com os dados reais** do InfoSiga. A versão 2 (`Downloads\tcc_infosiga_v2_final`)
gerou o dashboard com dados sintéticos de demonstração, por isso os números dela não devem ir para o TCC.

## Estrutura

```
infosiga_profissional_24.10/
├── 01_ingestao_e_inspecao.ipynb          leitura dos CSVs (Polars), qualidade, cobertura
├── 02_limpeza_e_preparacao.ipynb         funil, variáveis, flags de veículos e tipos
├── 03_analise_exploratoria.ipynb         padrões temporais, gravidade, vias, modos (motos, pesados, bicicletas, pedestres)
├── 04_comparativo_e_fatores_de_risco.ipynb  tendência/crescimento, qui-quadrado, regressão logística (OR), capotamento × via
├── 05_espacial_vitimas_e_sintese.ipynb   mapas KDE, hotspots, perfil e letalidade das vítimas, radar
├── 06_random_forest_predicoes.ipynb      Random Forest, Gini, AUC-ROC, cenários calibrados
├── data/raw/          CSVs originais do InfoSiga (somente leitura)
├── data/interim/      Parquet com os valores brutos
├── data/processed/    Parquet da base analítica
├── src/               config, leitura, limpeza, estatística, gráficos, dashboard
├── reports/figures/   todos os gráficos (PNG 300 dpi) + série interativa Altair (HTML)
├── reports/tables/    todas as tabelas (CSV) + texto pronto para o TCC (06_texto_tcc.md)
├── reports/dashboard_tcc_v3.html   dashboard no mesmo padrão do v2
├── docs/metodologia.md
└── tools/             nb_src/ (fonte dos notebooks), py2nb.py, saidas.py
```

## Como executar (VS Code)

0. Os CSVs brutos não estão no GitHub (cada arquivo passa de 100 MB). Baixe no portal do
   InfoSiga SP os arquivos `sinistros`, `pessoas` e `veiculos` de 2022–2024 e 2025–2026 e salve em
   `data/raw/`. A base já processada (`data/processed/*.parquet`) está no repositório, então os
   notebooks 03–06 rodam sem os CSVs.
1. Abra a pasta no VS Code e selecione o interpretador `.venv` (Python 3.13).
2. Execute os notebooks **em ordem** (01 → 06). Cada um lê o que o anterior salvou.
3. Gere o dashboard:

```bash
.venv/Scripts/python -m src.dashboard
```

Para reinstalar o ambiente: `python -m venv .venv` e depois
`.venv/Scripts/python -m pip install -r requirements.txt`.

> **Polars:** este computador não tem as instruções AVX2, por isso é usado `polars[rtcompat]`
> (versão oficial compatível). O `polars` comum fecha com "Illegal instruction".

## Recorte

- Municípios: Campinas e Guarulhos.
- Registros: **somente sinistros com vítima** (`SINISTRO FATAL` e `SINISTRO NAO FATAL`); as
  `NOTIFICACAO` ficam fora e entram apenas na análise de sensibilidade do notebook 04.
- Tipo de via: `ESTRADAS E RODOVIAS` → RODOVIA; `VIAS URBANAS` → VIA URBANA.
- Período: 2022–2025 (2026 está incompleto).
