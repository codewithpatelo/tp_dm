"""Promueve un experimento a `notebook_campeon.ipynb`.

Toma el `<entrega>_<nombre>.executed.ipynb` (que el orquestador genera
en cada corrida con outputs y código exactos), le saca `outputs` y
`execution_count`, y lo guarda como `entregas/<entrega>/notebook_campeon.ipynb`.

Es la implementación manual de lo que `run_entrega.py::post_process`
debería disparar automáticamente cuando declara un nuevo campeón
Kaggle (junto con la regeneración del informe).

Uso:
    python entregas/promote_champion.py --entrega entrega_2 --nombre v4

Convención: hay 3 notebooks por entrega. El campeón es la fuente de
verdad para entregar al docente cuando lo pida; el "último" (en la raíz)
es el que el orquestador edita y corre. Ver `.claude/CONTEXT.md →
Convención de notebooks por entrega`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def promote(entrega: str, nombre: str) -> Path:
    src = ROOT / "entregas" / entrega / f"{entrega}_{nombre}.executed.ipynb"
    dst = ROOT / "entregas" / entrega / "notebook_campeon.ipynb"

    if not src.exists():
        raise FileNotFoundError(
            f"No existe {src}. Necesitás haber corrido al menos una vez "
            f"`python entregas/run_entrega.py --entrega {entrega} --nombre {nombre}` "
            f"para que el .executed.ipynb se genere."
        )

    nb = json.loads(src.read_text(encoding="utf-8"))

    n_code = n_md = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            n_code += 1
        else:
            n_md += 1

    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    src_kb = src.stat().st_size / 1024
    dst_kb = dst.stat().st_size / 1024
    print(
        f"[promote_champion] {dst.relative_to(ROOT)}\n"
        f"  fuente : {src.relative_to(ROOT)} ({src_kb:.1f} KB con outputs)\n"
        f"  destino: {dst_kb:.1f} KB sin outputs\n"
        f"  celdas : {len(nb['cells'])} ({n_code} code, {n_md} md)"
    )
    return dst


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entrega", required=True, help="ej. entrega_2")
    p.add_argument(
        "--nombre",
        required=True,
        help="nombre del experimento campeón a promover (ej. v4)",
    )
    args = p.parse_args()
    promote(args.entrega, args.nombre)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
