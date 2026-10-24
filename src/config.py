"""Configurações centrais do projeto.

Escopo: sinistros COM VÍTIMA (fatal + não fatal) em Campinas e Guarulhos,
rodovias e vias urbanas, 2022–2025 – InfoSiga SP.
"""
from pathlib import Path

# ── Caminhos ─────────────────────────────────────────────────────────────────
RAIZ           = Path(__file__).resolve().parents[1]
DATA_RAW       = RAIZ / "data" / "raw"
DATA_INTERIM   = RAIZ / "data" / "interim"
DATA_PROCESSED = RAIZ / "data" / "processed"
FIG_DIR        = RAIZ / "reports" / "figures"
TAB_DIR        = RAIZ / "reports" / "tables"
for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, FIG_DIR, TAB_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ── Recorte ───────────────────────────────────────────────────────────────────
MUNICIPIOS  = ["CAMPINAS", "GUARULHOS"]
ANOS        = [2022, 2023, 2024, 2025]
SEED        = 42

# Tipos de registro do InfoSiga que entram na análise (NOTIFICACAO = sem vítima, fica de fora)
REGISTROS_COM_VITIMA = ["SINISTRO FATAL", "SINISTRO NAO FATAL"]

# tipo_via do InfoSiga → grupo usado no TCC
GRUPO_VIA = {"ESTRADAS E RODOVIAS": "RODOVIA", "VIAS URBANAS": "VIA URBANA"}
GRUPOS_VIA = ["RODOVIA", "VIA URBANA"]

# Caixas geográficas (lat_min, lat_max, lon_min, lon_max) para validar coordenadas
CAIXAS = {
    "CAMPINAS":  (-23.25, -22.65, -47.35, -46.80),
    "GUARULHOS": (-23.55, -23.25, -46.65, -46.30),
}

# População – Censo IBGE 2022
POPULACAO = {"CAMPINAS": 1_139_047, "GUARULHOS": 1_291_771}

# Nomes de rodovias (apenas os de identificação inequívoca; demais ficam só com o código)
NOMES_RODOVIAS = {
    "SP-330": "SP-330 (Anhanguera)",
    "SP-348": "SP-348 (Bandeirantes)",
    "SP-065": "SP-065 (D. Pedro I)",
    "SP-075": "SP-075 (Santos Dumont)",
    "BR-116": "BR-116 (Pres. Dutra)",
    "SP-070": "SP-070 (Ayrton Senna)",
    "BR-381": "BR-381 (Fernão Dias)",
    "SP-019": "SP-019 (Hélio Smidt)",
}

# ── Visual ────────────────────────────────────────────────────────────────────
CORES        = {"CAMPINAS": "#1f77b4", "GUARULHOS": "#e6550d"}
CORES_VIA    = {"RODOVIA": "#7f0000",  "VIA URBANA": "#2ca02c"}
CORES_GRAV   = {"SEM VÍTIMA / N.D.": "#d9d9d9", "LEVE": "#fdae6b",
                "GRAVE": "#e6550d", "FATAL": "#7f0000"}
ORDEM_DIAS       = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
ORDEM_PERIODOS   = ["Madrugada", "Manhã", "Tarde", "Noite"]
ORDEM_GRAVIDADE  = ["SEM VÍTIMA / N.D.", "LEVE", "GRAVE", "FATAL"]

# Flags de modo/veículo derivadas das colunas qtd_* da tabela de sinistros
FLAGS_VEICULO = {
    "envolve_moto":      ["qtd_motocicleta"],
    "envolve_bicicleta": ["qtd_bicicleta"],
    "envolve_pedestre":  ["qtd_pedestre"],
    "envolve_automovel": ["qtd_automovel"],
    "envolve_caminhao":  ["qtd_caminhao"],
    "envolve_onibus":    ["qtd_onibus"],
    "envolve_pesado":    ["qtd_caminhao", "qtd_onibus"],   # veículo pesado = caminhão ou ônibus
}
ROTULOS_MODO = {
    "envolve_moto": "Motocicleta", "envolve_bicicleta": "Bicicleta",
    "envolve_pedestre": "Pedestre", "envolve_automovel": "Automóvel",
    "envolve_pesado": "Veículo pesado",
}

# Flags de tipo de sinistro (colunas tp_sinistro_* com 'S')
FLAGS_TIPO = {
    "tp_atropelamento":   ["tp_sinistro_atrop_pedestre", "tp_sinistro_atrop_vitima_fora_veic"],
    "tp_colisao_frontal": ["tp_sinistro_colisao_frontal"],
    "tp_colisao_lateral": ["tp_sinistro_colisao_lateral"],
    "tp_colisao_transv":  ["tp_sinistro_colisao_transversal"],
    "tp_choque":          ["tp_sinistro_choque"],
    "tp_capotamento":     ["tp_sinistro_capotamento"],
    "tp_tombamento":      ["tp_sinistro_tombamento"],
    "tp_engavetamento":   ["tp_sinistro_engavetamento"],
}
ROTULOS_TIPO = {
    "tp_atropelamento": "Atropelamento", "tp_colisao_frontal": "Colisão frontal",
    "tp_colisao_lateral": "Colisão lateral", "tp_colisao_transv": "Colisão transversal",
    "tp_choque": "Choque", "tp_capotamento": "Capotamento",
    "tp_tombamento": "Tombamento", "tp_engavetamento": "Engavetamento",
}
