"""Testes e tabelas estatísticas (pandas / scipy / statsmodels)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency


def qui_quadrado(df: pd.DataFrame, coluna: str, linha: str = "municipio") -> dict:
    """Qui-quadrado de independência + V de Cramér entre `linha` e `coluna`."""
    mat = pd.crosstab(df[linha], df[coluna])
    mat = mat.loc[:, mat.sum() > 0]
    if min(mat.shape) < 2:
        return {"variavel": coluna, "n": int(mat.values.sum()), "chi2": np.nan,
                "gl": 0, "p_valor": np.nan, "cramer_v": np.nan}
    chi2, p, gl, _ = chi2_contingency(mat.values)
    n = mat.values.sum(); k = min(mat.shape) - 1
    return {"variavel": coluna, "n": int(n), "chi2": round(float(chi2), 2), "gl": int(gl),
            "p_valor": float(p), "cramer_v": round(float(np.sqrt(chi2 / (n * k))), 4)}


def tabela_risco(df: pd.DataFrame, fator: str, alvo: str = "grave_fatal",
                 grupos=("municipio",), minimo: int = 30) -> pd.DataFrame:
    """% de sinistros graves/fatais por nível de `fator` dentro de cada grupo."""
    t = (df.dropna(subset=[fator])
           .groupby(list(grupos) + [fator], observed=True)
           .agg(n=(alvo, "size"), casos=(alvo, "sum")).reset_index())
    t["pct"] = (t["casos"] / t["n"] * 100).round(2)
    return t[t["n"] >= minimo]


def tendencia_poisson(mensal: pd.DataFrame, col_n: str = "n") -> dict:
    """Regressão de Poisson n ~ tempo (anos) + mês (sazonalidade) sobre a série mensal.

    Retorna a razão de taxas anual (IRR), IC 95% e p-valor do termo de tendência.
    IRR > 1 = crescimento; usa erro-padrão robusto à superdispersão (HC0).
    """
    d = mensal.copy()
    d["t"] = (d["ano"] - d["ano"].min()) + (d["mes"] - 1) / 12
    m = smf.glm(f"{col_n} ~ t + C(mes)", data=d, family=sm.families.Poisson()).fit(cov_type="HC0")
    b, (lo, hi) = m.params["t"], m.conf_int().loc["t"]
    return {"irr_anual": float(np.exp(b)), "ic_inf": float(np.exp(lo)),
            "ic_sup": float(np.exp(hi)), "p_valor": float(m.pvalues["t"])}


def cagr(v0: float, v1: float, anos: int) -> float:
    return (v1 / v0) ** (1 / anos) - 1 if v0 > 0 else np.nan
