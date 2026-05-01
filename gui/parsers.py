"""Parsers de artifacts del repo (leaderboard, soluciones JSON, plans, CONTEXT)."""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_FILE = ROOT / ".claude" / "CONTEXT.md"
SOLUTIONS_GLOB = str(ROOT / "entregas" / "**" / "solucion-*.json")
LEADERBOARD_GLOB = str(ROOT / "entregas" / "**" / "leaderboard.md")


def latest_leaderboard() -> Path | None:
    """Devuelve el leaderboard.md más reciente (por entrega más alta)."""
    paths = [Path(p) for p in glob.glob(LEADERBOARD_GLOB, recursive=True)]
    if not paths:
        return None
    return max(paths, key=lambda p: (p.stat().st_mtime, p.as_posix()))


LB_FILE = latest_leaderboard() or (ROOT / "leaderboard.md")


# ---------- informe + notebook campeón (por entrega) --------------------------

@dataclass
class EntregaEntregables:
    """Rutas bajo `entregas/entrega_n/`."""
    carpeta: str
    ruta: Path
    informe_md: Optional[Path] = None
    notebook_campeon: Optional[Path] = None
    informe_pdf: list[Path] = field(default_factory=list)


def list_entregas_con_entregables() -> list[EntregaEntregables]:
    """Entregas con al menos un entregable (informe .md, .pdf o notebook campeón)."""
    out: list[EntregaEntregables] = []
    for d in sorted(ROOT.glob("entregas/entrega_*")):
        if not d.is_dir():
            continue
        mds = sorted(d.glob("*informe*.md"), key=lambda p: p.name)
        informe = mds[0] if mds else None
        nb: Optional[Path] = None
        for candidate in (d / "notebook_campeon.ipynb",):
            if candidate.exists():
                nb = candidate
                break
        if nb is None:
            for n in sorted(d.glob("notebook_campeon*.ipynb")):
                nb = n
                break
        # PDFs de entrega (nombre típico: "Entrega Parcial n - ...pdf")
        pdfs = sorted(
            [p for p in d.glob("*.pdf")],
            key=lambda p: p.name.lower(),
        )
        if informe is None and nb is None and not pdfs:
            continue
        out.append(
            EntregaEntregables(
                carpeta=d.name,
                ruta=d,
                informe_md=informe,
                notebook_campeon=nb,
                informe_pdf=pdfs,
            )
        )
    return out


# ---------- leaderboard ------------------------------------------------------

@dataclass
class LBRow:
    nombre: str
    rmse_cv5_mean: Optional[float]
    rmse_holdout_temporal: Optional[float]
    rmse_holdout: Optional[float]
    rmse_kaggle: Optional[float]
    raw: str


def _parse_num(s: str) -> Optional[float]:
    s = (s or "").replace("**", "").replace("_", "").strip()
    if not s or s in {"—", "-", "pendiente", "None"}:
        return None
    if "±" in s:
        s = s.split("±")[0].strip()
    try:
        return float(s.replace(" ", "").replace(",", ""))
    except ValueError:
        return None


def parse_leaderboard(path: Path | None = None) -> list[LBRow]:
    """Parser posicional: detecta header y mapea columnas por nombre."""
    if path is None:
        path = latest_leaderboard() or LB_FILE
    if not path.exists():
        return []

    header_idx: dict[str, int] | None = None
    rows: list[LBRow] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue

        if header_idx is None:
            low = [c.lower() for c in cells]
            if "nombre" in low or "experimento" in low or "version" in low:
                header_idx = {}
                for i, c in enumerate(low):
                    if "cv5" in c:
                        header_idx["cv5"] = i
                    elif "holdout temporal" in c:
                        header_idx["ho_temp"] = i
                    elif "holdout" in c and "temp" not in c:
                        header_idx["ho"] = i
                    elif "kaggle" in c:
                        header_idx["kg"] = i
                    elif i == 0:
                        header_idx["nombre"] = i
                continue

        if header_idx is None:
            continue

        nombre = cells[header_idx.get("nombre", 0)]
        if not nombre:
            continue

        def _at(key: str):
            i = header_idx.get(key)
            if i is None or i >= len(cells):
                return None
            return _parse_num(cells[i])

        rows.append(LBRow(
            nombre=nombre,
            rmse_cv5_mean=_at("cv5"),
            rmse_holdout_temporal=_at("ho_temp"),
            rmse_holdout=_at("ho"),
            rmse_kaggle=_at("kg"),
            raw=raw,
        ))
    return rows


def champion(rows: list[LBRow] | None = None) -> Optional[LBRow]:
    """Mejor fila por kaggle si hay; si no por holdout temporal; si no cv5."""
    if rows is None:
        rows = parse_leaderboard()
    if not rows:
        return None
    by_kaggle = [r for r in rows if r.rmse_kaggle is not None]
    if by_kaggle:
        return min(by_kaggle, key=lambda r: r.rmse_kaggle)
    by_ho = [r for r in rows if r.rmse_holdout_temporal is not None]
    if by_ho:
        return min(by_ho, key=lambda r: r.rmse_holdout_temporal)
    by_cv = [r for r in rows if r.rmse_cv5_mean is not None]
    if by_cv:
        return min(by_cv, key=lambda r: r.rmse_cv5_mean)
    return rows[0]


# ---------- soluciones JSON --------------------------------------------------

def list_solutions() -> list[Path]:
    return sorted(Path(p) for p in glob.glob(SOLUTIONS_GLOB, recursive=True))


def load_solution(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------- CONTEXT.md sections ---------------------------------------------

def context_sections() -> dict[str, str]:
    """Parsea CONTEXT.md como dict {heading: cuerpo}."""
    if not CONTEXT_FILE.exists():
        return {}
    text = CONTEXT_FILE.read_text(encoding="utf-8")
    # Headings nivel 2 y 3 como anchors
    parts = re.split(r"\n(?=#{1,3} )", text)
    out: dict[str, str] = {}
    for p in parts:
        m = re.match(r"(#{1,3})\s+(.+?)\n", p)
        if not m:
            continue
        title = m.group(2).strip()
        out[title] = p.strip()
    return out


def context_section(name: str) -> str:
    """Busca por substring case-insensitive."""
    secs = context_sections()
    needle = name.lower()
    for k, v in secs.items():
        if needle in k.lower():
            return v
    return ""


# ---------- plan TODOs (~/.cursor/plans/*.md) -------------------------------

@dataclass
class PlanTodo:
    plan_file: str
    content: str
    status: str


def parse_plan_todos(plan_dir: Path | None = None) -> list[PlanTodo]:
    """Recorre los .plan.md del usuario y extrae TODOs."""
    if plan_dir is None:
        home = Path(os.path.expanduser("~"))
        plan_dir = home / ".cursor" / "plans"
    if not plan_dir.exists():
        return []
    todos: list[PlanTodo] = []
    for f in sorted(plan_dir.glob("*.plan.md")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(
            r"id:\s*([^\n]+)\s*\n\s*content:\s*\"?([^\"\n]+)\"?\s*\n\s*status:\s*([a-z_]+)",
            text,
        ):
            todos.append(PlanTodo(
                plan_file=f.name,
                content=m.group(2).strip(),
                status=m.group(3).strip(),
            ))
    return todos
