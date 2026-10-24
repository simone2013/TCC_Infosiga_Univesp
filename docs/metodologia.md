# Metodologia – passo a passo

Os números citados nos notebooks vêm de `reports/tables/`. Este documento descreve **como** cada
etapa foi feita; para os valores, consulte as tabelas (elas são regeneradas a cada execução).

## 1. Fonte e ingestão (notebook 01)
- InfoSiga SP, dados abertos: tabelas `sinistros`, `pessoas` e `veiculos` (arquivos 2022–2024 e
  2025–2026), ligadas por `id_sinistro`.
- Leitura com **Polars**, codificação latin-1, separador `;`, todas as colunas como texto.
- Qualidade: nulos, valores únicos, duplicidade de chave, cobertura temporal.
- Cópia fiel salva em Parquet (`data/interim/`).

## 2. Limpeza e variáveis (notebook 02, `src/limpeza.py`)
Funil: bruto → sem `id_sinistro` duplicado → Campinas + Guarulhos → somente sinistros com vítima →
tipo de via válido → 2022–2025 (`reports/tables/funil_selecao.csv`).

| Variável | Regra |
|---|---|
| `grupo_via` | `ESTRADAS E RODOVIAS` → RODOVIA; `VIAS URBANAS` → VIA URBANA |
| `periodo` | hora do sinistro: Madrugada 0–5, Manhã 6–11, Tarde 12–17, Noite 18–23 |
| `fim_semana` | sábado ou domingo |
| `gravidade` | FATAL (qtd_gravidade_fatal>0 ou registro fatal) > GRAVE > LEVE > SEM VÍTIMA / N.D. |
| `grave_fatal` | gravidade ∈ {GRAVE, FATAL} – variável-alvo dos modelos |
| `envolve_*` | `qtd_motocicleta`, `qtd_bicicleta`, `qtd_pedestre`, `qtd_automovel`, `qtd_caminhao`, `qtd_onibus` > 0; **veículo pesado** = caminhão ou ônibus |
| `tp_*` | flags `tp_sinistro_*` = 'S' (atropelamento, capotamento, tombamento, colisões, choque, engavetamento) |
| `via_nome` | rodovias padronizadas pelo código (SP-330 …); vias urbanas sem acento e abreviadas |
| `categoria_vitima` | pedestre, motociclista, ciclista, ocupante de automóvel, ocupante de veículo pesado |

Checagens automáticas: flags de moto/bicicleta/pedestre/pesado/capotamento não podem estar zeradas;
o total da base analítica tem de bater com uma contagem direta feita no CSV bruto.

## 3. Análise exploratória (notebook 03)
Comparações em % dentro de cada município × tipo de via: volume anual, série mensal, dia da semana,
hora × dia, período, gravidade, tipos de sinistro, top vias e modos envolvidos.

## 4. Tendência e fatores de risco (notebook 04)
- **Crescimento:** variação 2022→2025 e 2024→2025, CAGR, e regressão de Poisson na série mensal
  (`n ~ tempo + mês`, erro-padrão robusto HC0) → razão de taxas anual (IRR) com IC 95%.
  "Crescimento contínuo" = aumento em todos os anos consecutivos.
- **Meses anômalos:** < 50% da mediana dos 12 meses anteriores (possível sub-registro).
- **Sensibilidade:** 2022–2024 apenas; e incluindo `NOTIFICACAO`.
- **Qui-quadrado + V de Cramér:** Campinas × Guarulhos (por via) e rodovia × via urbana (por município).
- **Regressão logística (statsmodels):** OR, IC 95% e p-valor; referência Campinas, via urbana,
  manhã, dia útil, 2022. `envolve_pedestre` e `tp_atropelamento` são quase idênticos (correlação
  calculada no notebook), por isso só o primeiro entra. VIF verificado. AUC-ROC no teste 20%.
- **Capotamento × tipo de via:** taxas por segmento, modelos estratificados por via e termo de interação.

## 5. Espacial e vítimas (notebook 05)
KDE + pontos por gravidade; hotspots por km (rodovias) e por célula de ~500 m (vias urbanas);
perfil das vítimas (lesão leve, grave ou fatal); letalidade = fatais ÷ vítimas, só categorias com n ≥ 30.

## 6. Random Forest (notebook 06)
scikit-learn; 80/20 estratificado (semente 42); 300 árvores, profundidade 10, folha mínima 20,
pesos balanceados; AUC-ROC no teste e CV estratificada 5-fold; importância Gini e por permutação;
cenários com probabilidades calibradas (isotônica). O texto para o TCC é gerado em
`reports/tables/06_texto_tcc.md`.

## Limitações
Possível sub-registro em meses de 2025; parte dos sinistros com vítima sem gravidade informada
(proporção diferente por segmento); colisão traseira vazia no InfoSiga; ciclistas pouco
identificados como vítimas; ausência de dados de tráfego (VMD).
