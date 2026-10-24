"""Imprime as saídas de texto de um notebook executado (para conferência rápida)."""
import sys
import nbformat as nbf

for caminho in sys.argv[1:]:
    nb = nbf.read(caminho, as_version=4)
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            txt = o.get("text") or o.get("data", {}).get("text/plain", "")
            if o.get("output_type") == "error":
                txt = f"ERRO {o['ename']}: {o['evalue']}"
            if txt:
                print(f"--- célula {i} ---\n{txt[:3000]}")
