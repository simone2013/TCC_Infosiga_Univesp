"""Tema e exportação de gráficos (matplotlib/seaborn) – mesmo padrão do dashboard v2."""
import matplotlib.pyplot as plt
import seaborn as sns
from . import config


def tema():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300,
        "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def salvar_fig(fig, nome: str):
    p = config.FIG_DIR / f"{nome}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  ✔ figura salva: reports/figures/{nome}.png")
    return p


def salvar_tab(df, nome: str, index: bool = False):
    p = config.TAB_DIR / f"{nome}.csv"
    df.to_csv(p, index=index, encoding="utf-8-sig")
    print(f"  ✔ tabela salva: reports/tables/{nome}.csv")
    return p


def rotulos_pct(ax, fmt="%.1f%%", fontsize=9):
    for c in ax.containers:
        ax.bar_label(c, fmt=fmt, padding=2, fontsize=fontsize)


def titulo_mun(m: str) -> str:
    return m.title()
