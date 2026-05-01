"""Skills de acceso a datos: SQL, lectura de archivos, listado, RAG vectorless."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import register
from ..data_loader import DataRegistry
from ..parsers import (
    ROOT, champion, context_section, list_solutions, load_solution,
    parse_leaderboard, parse_plan_todos,
)


# ---------- SQL --------------------------------------------------------------

@register(
    "query_sql",
    "Consulta DuckDB sobre train_filtered, test, train_raw. Devuelve DataFrame.",
    {"sql": "string"},
)
def query_sql(sql: str):
    if not sql.strip().lower().startswith(("select", "with", "describe", "show")):
        raise ValueError("Solo SELECT/WITH/DESCRIBE/SHOW permitidos.")
    return DataRegistry.get().query(sql)


# ---------- file IO ---------------------------------------------------------

ALLOWED_EXTS = {".md", ".py", ".json", ".txt", ".csv", ".yml", ".yaml", ".ipynb", ".r"}


def _safe_path(path: str) -> Path:
    p = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not str(p).startswith(str(ROOT.resolve())):
        raise ValueError(f"path fuera del repo: {path}")
    return p


@register(
    "read_project_file",
    "Lee un archivo del repo (md/py/json/txt/csv/yml/ipynb/r). Devuelve string.",
    {"path": "string", "max_chars": "int (default 8000)"},
)
def read_project_file(path: str, max_chars: int = 8000):
    p = _safe_path(path)
    if p.suffix.lower() not in ALLOWED_EXTS:
        raise ValueError(f"extensión no permitida: {p.suffix}")
    if not p.exists():
        raise FileNotFoundError(str(p))
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if len(txt) > max_chars:
        return txt[:max_chars] + f"\n\n... [truncado a {max_chars} chars]"
    return txt


@register(
    "list_files",
    "Lista archivos del repo que matchean un glob (relativo a la raíz).",
    {"pattern": "string ej. 'entregas/**/*.md'"},
)
def list_files(pattern: str = "**/*"):
    matches = sorted(p.relative_to(ROOT) for p in ROOT.glob(pattern)
                     if p.is_file())
    return [str(p) for p in matches[:200]]


# ---------- vectorless RAG --------------------------------------------------

@register(
    "vectorless_rag",
    "Busca pasajes relevantes via grep sobre context|notebooks|reports|memory.",
    {"query": "string", "kind": "context|notebooks|reports|memory|all (default all)"},
)
def vectorless_rag(query: str, kind: str = "all", max_hits: int = 12):
    """Usa rg si está; fallback a python re.

    Devuelve lista de {file, line, snippet}.
    """
    targets = _kind_to_paths(kind)
    hits: list[dict] = []
    for target in targets:
        if not target.exists():
            continue
        try:
            out = subprocess.run(
                ["rg", "-n", "--no-heading", "-i", "-m", "3", query, str(target)],
                capture_output=True, text=True, timeout=10,
            )
            for ln in (out.stdout or "").splitlines()[:max_hits * 2]:
                m = re.match(r"([^:]+):(\d+):(.*)", ln)
                if m:
                    hits.append({
                        "file": str(Path(m.group(1)).relative_to(ROOT)),
                        "line": int(m.group(2)),
                        "snippet": m.group(3).strip()[:200],
                    })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            hits.extend(_python_grep(target, query, max_hits))
        if len(hits) >= max_hits:
            break
    return hits[:max_hits]


def _kind_to_paths(kind: str) -> list[Path]:
    mapping = {
        "context": [ROOT / ".claude", ROOT / "README.md"],
        "notebooks": [ROOT / "entregas"],
        "reports": [ROOT / "entregas"],
        "memory": [Path(__file__).resolve().parent.parent / "_memory"],
    }
    if kind == "all":
        return [p for ps in mapping.values() for p in ps]
    return mapping.get(kind, [])


def _python_grep(root: Path, query: str, max_hits: int) -> list[dict]:
    pat = re.compile(re.escape(query), re.IGNORECASE)
    out: list[dict] = []
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in ALLOWED_EXTS:
            continue
        try:
            for i, ln in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if pat.search(ln):
                    out.append({
                        "file": str(f.relative_to(ROOT)),
                        "line": i, "snippet": ln.strip()[:200],
                    })
                    if len(out) >= max_hits:
                        return out
        except Exception:  # noqa: BLE001
            continue
    return out


# ---------- meta-info del proyecto ------------------------------------------

@register("get_champion", "Devuelve la fila campeona del leaderboard.", {})
def get_champion():
    c = champion()
    return c.__dict__ if c else None


@register("list_experiments", "Lista todos los experimentos del leaderboard.", {})
def list_experiments():
    return [r.__dict__ for r in parse_leaderboard()]


@register("list_solution_jsons", "Lista paths de soluciones JSON.", {})
def list_solution_jsons():
    return [str(p.relative_to(ROOT)) for p in list_solutions()]


@register("load_solution_json", "Carga solucion-*.json.", {"path": "string"})
def load_solution_json(path: str):
    return load_solution(_safe_path(path))


@register("read_context_section", "Lee sección de CONTEXT.md por nombre.",
          {"name": "string"})
def read_context_section(name: str):
    sec = context_section(name)
    return sec or f"(no encontré sección que matchee '{name}')"


@register("plan_todos", "Lista TODOs del plan activo en ~/.cursor/plans/.", {})
def plan_todos():
    return [t.__dict__ for t in parse_plan_todos()]
