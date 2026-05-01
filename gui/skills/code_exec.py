"""Skills de ejecución de código y propuesta de cambios a archivos.

execute_python: sandbox restringido (gui/sandbox.py).
plot_chart: genera spec plotly (renderiza la UI, acá solo devolvemos JSON).
propose_file_change: emite proposal.pending para que la UI muestre diff y apruebe.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from . import register
from ..data_loader import DataRegistry
from ..events import EVT_PROPOSAL_PENDING, get_bus
from ..parsers import ROOT
from ..sandbox import execute_python as _exec


@register(
    "execute_python",
    "Ejecuta código Python en sandbox. Disponibles: pd, np, plt, df (train_filtered).",
    {"code": "string", "table": "string default train_filtered", "timeout_s": "float default 10"},
)
def execute_python(code: str, table: str = "train_filtered", timeout_s: float = 10.0):
    df = DataRegistry.get().table(table)
    res = _exec(code, df=df, timeout_s=timeout_s)
    return {
        "ok": res.ok,
        "stdout": res.stdout[-2000:],
        "error": res.error,
        "n_figures": len(res.figures),
        "locals": res.locals_after,
    }


# ---------- plot_chart ------------------------------------------------------

@register(
    "plot_chart",
    "Genera spec de gráfico Plotly. La UI lo renderiza. "
    "spec: {type: hist|scatter|box|line|bar|map, table, x?, y?, color?, group?}",
    {"spec": "dict"},
)
def plot_chart(spec: dict):
    if not isinstance(spec, dict) or "type" not in spec:
        return {"error": "spec necesita 'type'"}
    table = spec.get("table", "train_filtered")
    df = DataRegistry.get().table(table)
    cols_used = [c for c in (spec.get("x"), spec.get("y"), spec.get("color"), spec.get("group"))
                 if c]
    missing = [c for c in cols_used if c not in df.columns]
    if missing:
        return {"error": f"columnas no existen: {missing}"}
    sample = df
    if len(df) > 5000 and spec["type"] in {"scatter", "map"}:
        sample = df.sample(5000, random_state=42)
    return {
        "spec": spec,
        "n_rows": len(sample),
        "preview": sample[cols_used].head(3).to_dict(orient="records") if cols_used else [],
    }


# ---------- propose_file_change --------------------------------------------

@register(
    "propose_file_change",
    "Propone crear/modificar un archivo. Emite proposal.pending; la UI muestra "
    "diff y permite Apply/Reject. Hasta aprobación el archivo NO se toca.",
    {"path": "string relativo a la raíz", "content": "string", "rationale": "string"},
)
def propose_file_change(path: str, content: str, rationale: str = ""):
    target = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not str(target).startswith(str(ROOT.resolve())):
        return {"ok": False, "error": "path fuera del repo"}
    current = ""
    if target.exists():
        try:
            current = target.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"no pude leer {path}: {e}"}
    proposal_id = uuid.uuid4().hex[:8]
    payload = {
        "id": proposal_id,
        "path": str(target.relative_to(ROOT)),
        "current": current,
        "proposed": content,
        "rationale": rationale,
        "status": "pending",
    }
    get_bus().emit(EVT_PROPOSAL_PENDING, payload)
    return {
        "ok": True,
        "proposal_id": proposal_id,
        "path": payload["path"],
        "status": "pending_user_approval",
        "preview_diff_lines": _quick_diff_summary(current, content),
    }


def _quick_diff_summary(a: str, b: str) -> dict:
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    return {
        "current_lines": len(a_lines),
        "proposed_lines": len(b_lines),
        "delta": len(b_lines) - len(a_lines),
    }
