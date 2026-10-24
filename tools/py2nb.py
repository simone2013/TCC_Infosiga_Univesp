"""Converte scripts no formato 'percent' (# %% / # %% [markdown]) em notebooks .ipynb.

Uso: python tools/py2nb.py tools/nb_src/01_ingestao_e_inspecao.py  →  01_ingestao_e_inspecao.ipynb
"""
import sys
from pathlib import Path
import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]


def converter(src: Path) -> Path:
    cells, tipo, buf = [], None, []

    def fechar():
        if tipo is None:
            return
        txt = "\n".join(buf).strip("\n")
        if tipo == "md":
            txt = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in txt.splitlines())
            cells.append(nbf.v4.new_markdown_cell(txt))
        elif txt:
            cells.append(nbf.v4.new_code_cell(txt))

    for linha in src.read_text(encoding="utf-8").splitlines():
        if linha.startswith("# %%"):
            fechar()
            tipo, buf = ("md" if "[markdown]" in linha else "code"), []
        else:
            buf.append(linha)
    fechar()
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}
    destino = RAIZ / (src.stem + ".ipynb")
    nbf.write(nb, destino)
    return destino


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print("→", converter(Path(a)).name)
