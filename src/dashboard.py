"""Gera reports/dashboard_tcc_v3.html (mesmo layout do dashboard v2) a partir das figuras e
tabelas produzidas pelos notebooks 01–06. Nenhum número é digitado à mão: tudo vem dos CSVs.

Uso:  .venv/Scripts/python -m src.dashboard
"""
import base64
import html
import io
from datetime import date
import pandas as pd
from PIL import Image
from . import config

SECOES = {
    "exploratoria": ("03 · Análise Exploratória", [
        "03_volume_anual", "03_serie_mensal", "03_dia_semana", "03_heatmap_hora_dia", "03_periodo_dia",
        "03_gravidade_por_via", "03_evolucao_gravidade_anual", "03_tipos_sinistro", "03_top_vias",
        "03_modos_envolvidos", "03_modos_evolucao_anual"]),
    "comparativo": ("04 · Comparativo, Tendência e Fatores de Risco", [
        "04_tendencia_crescimento", "04_serie_mensal", "04_comparativo_via_gravidade",
        "04_taxa_gravidade_fatores", "04_odds_ratios", "04_odds_ratios_por_via"]),
    "espacial": ("05 · Espacial, Vítimas e Síntese", [
        "05_mapa_kde_2x2", "05_hotspots", "05_perfil_vitimas", "05_perfil_vitimas_urbanas",
        "05_fatalidade_tipo_vitima", "05_fatais_tipo_vitima_ano", "05_evolucao_mensal_gravidade",
        "05_radar_4segmentos"]),
    "ml": ("06 · Random Forest – Predições", [
        "06_rf_metricas", "06_importancia_permutacao", "06_predicoes_cenarios"]),
    "dados": ("01–02 · Dados e Funil", ["02_funil_selecao", "01_registros_por_ano"]),
}
TABELAS = ["funil_selecao", "03_visao_geral", "04_tendencia_crescimento", "04_meses_anomalos",
           "04_sensibilidade_tendencia", "04_taxa_100k", "04_odds_ratios", "04_odds_ratios_por_via",
           "04_auc_logistica", "04_capotamento_via", "04_qui_quadrado", "03_top_vias", "05_hotspots",
           "05_fatalidade_tipo_vitima", "05_sintese_indicadores", "06_metricas_resumo",
           "06_classification_report", "06_feature_importance", "06_predicoes_cenarios"]

CSS = """
:root{--az:#1f77b4;--la:#e6550d;--vd:#2ca02c;--bg:#f4f4f8;--bd:#dde;--tx:#1a1a2e}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:var(--bg);color:var(--tx);padding-bottom:50px}
header{background:linear-gradient(135deg,#0a3d62,var(--az));color:#fff;padding:30px 40px 22px;box-shadow:0 3px 12px #0004}
header h1{font-size:1.5rem;font-weight:700;line-height:1.3}
header p{margin-top:8px;opacity:.85;font-size:.93rem}
nav{background:#fff;border-bottom:2px solid var(--bd);padding:11px 40px;display:flex;gap:18px;flex-wrap:wrap;position:sticky;top:0;z-index:99}
nav a{text-decoration:none;color:var(--az);font-weight:600;font-size:.88rem;padding:4px 10px;border-radius:20px;transition:.2s}
nav a:hover{background:var(--az);color:#fff}
main{padding:34px 40px;max-width:1500px;margin:0 auto}
h2.section{font-size:1.2rem;color:var(--az);border-left:4px solid var(--az);padding-left:12px;margin:38px 0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px}
.card{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 10px #0002;transition:box-shadow .2s}
.card:hover{box-shadow:0 5px 20px #0003}
.card-title{font-size:.8rem;font-weight:700;color:#555;margin-bottom:9px;text-transform:uppercase;letter-spacing:.04em}
.card img{width:100%;border-radius:6px;display:block}
details{background:#fff;border:1px solid var(--bd);border-radius:8px;margin-bottom:9px;overflow:hidden}
summary{padding:13px 16px;cursor:pointer;font-weight:600;color:var(--az);list-style:none}
summary::after{content:" ▼";font-size:.7rem} details[open] summary::after{content:" ▲"}
.tbl-wrap{overflow-x:auto;padding:0 16px 14px}
table{border-collapse:collapse;font-size:.78rem;width:100%}
th{background:var(--az);color:#fff;padding:7px 9px;text-align:left;white-space:nowrap}
td{padding:5px 9px;border-bottom:1px solid #eee}
tr:nth-child(even){background:#f8f8fc}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0 6px}
.kpi{background:#fff;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;box-shadow:0 2px 8px #0001;border-top:4px solid var(--az)}
.kpi.la{border-color:var(--la)} .kpi.vd{border-color:var(--vd)}
.kv{font-size:1.75rem;font-weight:800;color:var(--az)}
.kpi.la .kv{color:var(--la)} .kpi.vd .kv{color:var(--vd)}
.kl{font-size:.76rem;color:#666;margin-top:3px}
.achados{background:#fff;border-radius:12px;padding:18px 24px;box-shadow:0 2px 10px #0002;line-height:1.6;font-size:.92rem}
.achados li{margin:4px 0 4px 18px}
footer{text-align:center;color:#999;font-size:.8rem;margin-top:44px}
.badge{display:inline-block;background:var(--la);color:#fff;border-radius:20px;padding:3px 12px;font-size:.76rem;margin-left:8px;font-weight:700}
@media (max-width:640px){header,nav,main{padding-left:16px;padding-right:16px}.grid{grid-template-columns:1fr}}
"""


def br(v, d=0):
    s = f"{v:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def img_b64(nome: str, largura_max: int = 2200) -> str | None:
    p = config.FIG_DIR / f"{nome}.png"
    if not p.exists():
        return None
    im = Image.open(p)
    if im.width > largura_max:
        im = im.resize((largura_max, int(im.height * largura_max / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.convert("RGB").save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def tab(nome: str) -> pd.DataFrame:
    return pd.read_csv(config.TAB_DIR / f"{nome}.csv", encoding="utf-8-sig")


def achados() -> str:
    tend = tab("04_tendencia_crescimento")
    t0 = tend[tend.indicador == "Todos os sinistros com vítima"]
    itens = []
    for _, r in t0.iterrows():
        sentido = "queda" if r.irr_anual < 1 else "crescimento"
        sig = "significativa" if r.p_valor < 0.05 else "não significativa"
        itens.append(f"<b>{r.municipio.title()} – {r.grupo_via}</b>: {br(r['2022'])} (2022) → "
                     f"{br(r['2025'])} (2025), {br(r.var_22_25_pct, 1)}%; tendência de {sentido} {sig} "
                     f"(IRR {br(r.irr_anual, 3)}); crescimento contínuo: <b>{'sim' if r.crescimento_continuo else 'não'}</b>.")
    fat = tend[tend.indicador == "Sinistros fatais"]
    n_sig = int((fat.p_valor < 0.05).sum())
    itens.append(f"Sinistros <b>fatais</b>: tendência significativa em {n_sig} de {len(fat)} segmentos.")
    anom = tab("04_meses_anomalos")
    if len(anom):
        lst = ", ".join(f"{a.municipio.title()}/{a.grupo_via} {a.ano_mes} (n={a.n})" for a in anom.itertuples())
        itens.append(f"<b>Atenção – possível sub-registro em 2025</b>: meses anômalos {lst}.")
    orr = tab("04_odds_ratios").query("significativo").sort_values("OR", ascending=False).head(5)
    itens.append("Maiores Odds Ratios (grave/fatal): " + "; ".join(
        f"{r.termo} {br(r.OR, 2)} (IC95% {br(r.IC95_inf, 2)}–{br(r.IC95_sup, 2)})" for r in orr.itertuples()) + ".")
    m = tab("06_metricas_resumo")
    itens.append(f"AUC-ROC (teste): regressão logística {br(m.auc_teste[0], 3)} · Random Forest "
                 f"{br(m.auc_teste[1], 3)} (CV 5-fold {br(m.cv_auc[1], 3)} ± {br(m.cv_dp[1], 3)}).")
    return "<ul>" + "".join(f"<li>{i}</li>" for i in itens) + "</ul>"


def gerar() -> str:
    geral = tab("03_visao_geral")
    m = tab("06_metricas_resumo")
    tot = int(geral.sinistros.sum())
    por_mun = geral.groupby("municipio").sinistros.sum()
    por_via = geral.groupby("grupo_via").sinistros.sum()
    kpis = [("", br(tot), "sinistros com vítima"), ("la", br(por_mun["CAMPINAS"]), "Campinas"),
            ("", br(por_mun["GUARULHOS"]), "Guarulhos"), ("la", br(por_via["RODOVIA"]), "em rodovias"),
            ("vd", br(por_via["VIA URBANA"]), "em vias urbanas"), ("", br(geral.fatais.sum()), "sinistros fatais"),
            ("la", "AUC " + br(m.auc_teste[1], 3), "Random Forest"), ("vd", "2022–2025", "período analisado")]
    h = [f"<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>Dashboard TCC v3 – InfoSiga SP 2022–2025</title><style>{CSS}</style></head><body>",
         "<header><h1>📊 Análise Comparativa de Acidentes de Trânsito</h1>"
         "<p>Campinas × Guarulhos · Rodovias e Vias Urbanas · InfoSiga SP · 2022–2025"
         "<span class='badge'>TCC · Ciência de Dados</span></p></header><nav>",
         "<a href='#achados'>Achados</a>"]
    h += [f"<a href='#{k}'>{t.split(' · ')[0]} · {t.split(' · ')[1].split(',')[0]}</a>" for k, (t, _) in SECOES.items()]
    h += ["<a href='#tabelas'>Tabelas</a></nav><main><div class='kpis'>"]
    h += [f"<div class='kpi {c}'><div class='kv'>{v}</div><div class='kl'>{l}</div></div>" for c, v, l in kpis]
    h += ["</div>", f"<section id='achados'><h2 class='section'>Principais achados (calculados)</h2>"
          f"<div class='achados'>{achados()}</div></section>"]
    for k, (titulo, figs) in SECOES.items():
        h.append(f"<section id='{k}'><h2 class='section'>{titulo}</h2><div class='grid'>")
        for f in figs:
            b = img_b64(f)
            if b:
                h.append(f"<div class='card'><div class='card-title'>{f.replace('_', ' ').title()}</div>"
                         f"<img src='data:image/png;base64,{b}' alt='{f}'></div>")
        h.append("</div></section>")
    h.append("<section id='tabelas'><h2 class='section'>📋 Tabelas de resultados</h2>")
    for t in TABELAS:
        p = config.TAB_DIR / f"{t}.csv"
        if p.exists():
            df = tab(t)
            h.append(f"<details><summary>{html.escape(t.replace('_', ' ').title())}</summary>"
                     f"<div class='tbl-wrap'>{df.to_html(index=False, border=0, float_format=lambda x: f'{x:.4g}')}</div></details>")
    h.append(f"</section></main><footer>Projeto TCC v3 · Dados reais InfoSiga SP (sinistros com vítima) · "
             f"gerado em {date.today():%d/%m/%Y}</footer></body></html>")
    destino = config.RAIZ / "reports" / "dashboard_tcc_v3.html"
    destino.write_text("\n".join(h), encoding="utf-8")
    return str(destino)


if __name__ == "__main__":
    print("✔ Dashboard:", gerar())
