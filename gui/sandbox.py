"""Sandbox read-only para análisis Code-First.

El LLM genera Python que se ejecuta acá con un contexto controlado:

- Datasets disponibles: `train_filtered`, `test` (DataFrames).
- Librerías: `pd`, `np`, `px`, `go`, `math`, `statistics`, `re`, `json`,
  `datetime`, `collections`, `itertools`, `functools`, `unicodedata`, `random`.
- Helper de salida: `publish_artifact(kind, title=..., **kwargs)` para
  registrar tablas, métricas o gráficos que renderiza la UI.
- `print()` se captura como notas humanas (stdout).

Capas de seguridad:
1. AST allow-list: imports y llamadas peligrosas (`os`, `subprocess`,
   `eval`, `open`, etc.) abortan la ejecución.
2. Builtins reducidos.
3. Timeout por hilo (default 20s).
4. Sin filesystem write, sin red, sin shell.
"""
from __future__ import annotations

import ast
import io
import re
import threading
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

# ---------- whitelist / blacklist ------------------------------------------

# Sólo permitimos imports de estos roots.
ALLOWED_IMPORT_ROOTS = {
    "re", "math", "statistics", "json", "datetime", "collections",
    "itertools", "functools", "unicodedata", "random", "decimal",
    "fractions", "string", "operator",
    "numpy", "pandas",
    "plotly", "plotly.express", "plotly.graph_objects", "plotly.io",
}

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "globals", "locals", "vars", "breakpoint", "exit", "quit",
    "memoryview",
}


@dataclass
class AnalysisResult:
    ok: bool
    stdout: str = ""
    error: str = ""
    traceback: str = ""
    artifacts: list[dict] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_summary(self, max_chars: int = 1800) -> str:
        """Resumen textual breve para meter en el prompt de síntesis."""
        parts: list[str] = []
        if self.error:
            parts.append(f"ERROR: {self.error}")
            if self.traceback:
                parts.append(f"Traceback (resumen):\n{self.traceback[-800:]}")
        if self.stdout.strip():
            parts.append("STDOUT (notas):\n" + self.stdout.strip()[-1200:])
        if self.outputs:
            parts.append("OUTPUTS (variables seleccionadas):")
            for k, v in list(self.outputs.items())[:12]:
                parts.append(f"- {k}: {v}")
        if self.artifacts:
            parts.append(f"ARTIFACTS publicados: {len(self.artifacts)}")
            for i, art in enumerate(self.artifacts[:5]):
                head = (
                    f"  [{i}] kind={art.get('kind')} title={art.get('title')!r}"
                )
                summary = art.get("summary")
                if summary:
                    head += f" summary={str(summary)[:300]}"
                parts.append(head)
        text = "\n".join(parts)
        return text[:max_chars]


# ---------- AST guard -------------------------------------------------------


class _AstGuard(ast.NodeVisitor):
    """Aborta si encuentra imports/llamadas no permitidas."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                self.problems.append(f"import bloqueado: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                self.problems.append(f"from-import bloqueado: {node.module}")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
            self.problems.append(f"llamada bloqueada: {func.id}()")
        if isinstance(func, ast.Attribute) and func.attr.startswith("_") \
                and not func.attr.startswith("__"):
            # Acceso a métodos privados (_x) lo permitimos: pandas usa eso.
            pass
        if isinstance(func, ast.Attribute) and func.attr in {"system", "popen"}:
            self.problems.append(f"método bloqueado: .{func.attr}()")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Bloquear dunders peligrosos (subclasses, globals, etc.)
        if node.attr in {
            "__globals__", "__code__", "__builtins__", "__import__",
            "__subclasses__", "__class__", "__bases__", "__mro__",
        }:
            self.problems.append(f"acceso dunder bloqueado: .{node.attr}")
        self.generic_visit(node)


# ---------- builtins seguros ------------------------------------------------


_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "getattr", "hasattr", "hash", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "object", "ord", "pow", "print", "range", "repr", "reversed",
    "round", "set", "setattr", "slice", "sorted", "str", "sum", "tuple",
    "type", "zip", "True", "False", "None",
    # Excepciones que el código pueda atrapar.
    "Exception", "ValueError", "KeyError", "TypeError", "ZeroDivisionError",
    "AttributeError", "IndexError", "RuntimeError",
)


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    """Wrapper de `__import__` que sólo permite roots whitelisted.

    Es defensa en profundidad: el AST guard ya rechaza imports no permitidos
    en parse time, pero esta capa cubre cualquier camino dinámico residual
    (p. ej. importlib lo bloqueamos por root, pero re-validamos en runtime).
    """
    root = (name or "").split(".")[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import bloqueado por sandbox: {name}")
    import builtins as _b
    return _b.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    src = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    out: dict[str, Any] = {}
    for name in _SAFE_BUILTIN_NAMES:
        if name in src:
            out[name] = src[name]
    out["__import__"] = _safe_import
    return out


# ---------- runner with timeout --------------------------------------------


def _run_with_timeout(fn: Callable[[], Any], timeout_s: float) -> tuple[bool, Any]:
    """Ejecuta `fn` en daemon thread; si tarda demasiado, retorna TimeoutError."""
    box: dict[str, Any] = {"value": None, "exc": None, "done": False}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["exc"] = e
        finally:
            box["done"] = True

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout_s)
    if not box["done"]:
        return False, TimeoutError(f"código superó {timeout_s:.1f}s")
    if box["exc"] is not None:
        return False, box["exc"]
    return True, box["value"]


# ---------- helper de dominio: inferencia de m² y ambientes ---------------

# Regex compiladas (vectorizadas vía pandas.Series.str.extract).
_RE_M2_EXTRACT = re.compile(
    r"(\d{1,4})\s*(?:m2|m²|mts2|mt2|metros\b)",
    re.IGNORECASE,
)
_RE_AMB_EXTRACT = re.compile(
    r"(\d{1,2})\s*(?:ambientes?|amb\b)",
    re.IGNORECASE,
)
_RE_DORM_EXTRACT = re.compile(
    r"(\d{1,2})\s*(?:dormitorios?|habitaciones?)",
    re.IGNORECASE,
)


# Cache por id(df) para no recomputar si el LLM llama dos veces.
_INFER_CACHE: dict[int, tuple[int, pd.Series, pd.Series]] = {}


def infer_real_estate_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve copia de `df` con `_rooms_est` y `_surface_m2_est`.

    Vectorizado: concatena `features` + `description` y aplica regex con
    re.IGNORECASE (sin str.lower(), evita una pasada full). Cachea por
    id(df)+shape para no repetir trabajo en el mismo turno.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("infer_real_estate_fields espera un DataFrame")

    cache_key = id(df)
    cached = _INFER_CACHE.get(cache_key)
    if cached and cached[0] == len(df):
        out = df.copy()
        out["_rooms_est"] = cached[1]
        out["_surface_m2_est"] = cached[2]
        return out

    out = df.copy()

    parts: list[pd.Series] = []
    for col in ("features", "description"):
        if col in out.columns:
            parts.append(out[col].fillna(""))
    if not parts:
        out["_rooms_est"] = pd.array([pd.NA] * len(out), dtype="Int64")
        out["_surface_m2_est"] = pd.array([pd.NA] * len(out), dtype="Float64")
        return out
    text = parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=" ", na_rep="")

    m2_str = text.str.extract(_RE_M2_EXTRACT, expand=False)
    m2 = pd.to_numeric(m2_str, errors="coerce")
    m2 = m2.where(m2.between(10, 800), other=np.nan).astype("Float64")

    amb_str = text.str.extract(_RE_AMB_EXTRACT, expand=False)
    amb = pd.to_numeric(amb_str, errors="coerce")
    amb = amb.where(amb.between(1, 12), other=np.nan)

    dorm_str = text.str.extract(_RE_DORM_EXTRACT, expand=False)
    dorm = pd.to_numeric(dorm_str, errors="coerce")
    dorm = dorm.where(dorm.between(0, 10), other=np.nan)

    rooms = amb.combine_first(dorm + 1).astype("Int64")

    out["_rooms_est"] = rooms
    out["_surface_m2_est"] = m2

    _INFER_CACHE[cache_key] = (len(df), rooms, m2)
    if len(_INFER_CACHE) > 8:
        _INFER_CACHE.pop(next(iter(_INFER_CACHE)))

    return out


# ---------- artifact normalization -----------------------------------------


_ALLOWED_KINDS = {
    "metrics", "table", "chart", "markdown",
    # Aliases usables por el LLM; los normalizamos a los 4 anteriores.
    "bar_chart", "line_chart", "scatter", "histogram", "box_plot", "map",
    "dataframe",
}


def _coerce_dataframe(obj: Any) -> pd.DataFrame | None:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, list):
        if not obj:
            return pd.DataFrame()
        if isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        return pd.DataFrame({"value": obj})
    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except Exception:  # noqa: BLE001
            return None
    return None


def _pick(kwargs: dict, *keys: str) -> Any:
    """Devuelve el primer valor no-None entre las keys (sin usar `or` para
    evitar el bool ambiguo de DataFrame/Series)."""
    for k in keys:
        if k in kwargs and kwargs[k] is not None:
            return kwargs[k]
    return None


def _normalize_artifact(kind: str, title: str | None, kwargs: dict) -> dict:
    """Convierte cualquier artifact a un payload renderizable y serializable."""
    kind = (kind or "").strip().lower()
    art: dict[str, Any] = {
        "kind": kind if kind in _ALLOWED_KINDS else "table",
        "title": title or kwargs.get("title") or "Resultado",
    }

    # ---- Charts (plotly) ------------------------------------------------
    if kind in {"chart", "bar_chart", "line_chart", "scatter",
                "histogram", "box_plot", "map"}:
        fig = _pick(kwargs, "figure", "fig")
        if fig is not None:
            art["kind"] = "chart"
            art["figure"] = fig
        else:
            df = _coerce_dataframe(_pick(kwargs, "data"))
            if df is not None:
                art["kind"] = "chart"
                art["data"] = df.head(500).to_dict(orient="records")
                art["chart_type"] = (
                    "bar" if kind == "bar_chart"
                    else "line" if kind == "line_chart"
                    else "scatter" if kind == "scatter"
                    else "histogram" if kind == "histogram"
                    else "box" if kind == "box_plot"
                    else "bar"
                )
                art["x"] = kwargs.get("x")
                art["y"] = kwargs.get("y")
                art["color"] = kwargs.get("color")
        if "summary" in kwargs:
            art["summary"] = kwargs["summary"]
        return art

    # ---- Tables --------------------------------------------------------
    if kind in {"table", "dataframe"} or kind == "":
        df = _coerce_dataframe(_pick(kwargs, "data", "rows"))
        if df is None:
            df = pd.DataFrame()
        art["kind"] = "table"
        art["data"] = df.head(500).to_dict(orient="records")
        art["shape"] = list(df.shape)
        if kwargs.get("caption") is not None:
            art["caption"] = str(kwargs["caption"])[:600]
        if "summary" in kwargs:
            art["summary"] = kwargs["summary"]
        return art

    # ---- Metrics -------------------------------------------------------
    if kind == "metrics":
        raw = _pick(kwargs, "data", "metrics")
        if raw is None:
            raw = {}
        items: list[dict] = []
        if isinstance(raw, dict):
            for label, value in list(raw.items())[:8]:
                items.append({"label": str(label), "value": value})
        elif isinstance(raw, list):
            for entry in raw[:8]:
                if isinstance(entry, dict) and "label" in entry:
                    items.append(entry)
                else:
                    items.append({"label": str(entry), "value": ""})
        art["kind"] = "metrics"
        art["items"] = items
        if kwargs.get("caption") is not None:
            art["caption"] = str(kwargs["caption"])[:600]
        return art

    # ---- Markdown ------------------------------------------------------
    if kind == "markdown":
        art["kind"] = "markdown"
        art["text"] = str(_pick(kwargs, "text", "data") or "")[:4000]
        return art

    # Fallback: tratar como tabla si trae data, si no markdown vacío.
    df = _coerce_dataframe(_pick(kwargs, "data"))
    if df is not None:
        art["kind"] = "table"
        art["data"] = df.head(500).to_dict(orient="records")
        art["shape"] = list(df.shape)
    else:
        art["kind"] = "markdown"
        art["text"] = str(_pick(kwargs, "data") or "")[:2000]
    return art


_RESERVED_VARS = {
    "pd", "np", "px", "go", "publish_artifact", "infer_real_estate_fields",
    "train_filtered", "test", "__builtins__",
}


def _serializable_outputs(local_ns: dict[str, Any]) -> dict[str, Any]:
    """Extrae variables 'top-level' livianas y serializables para el LLM."""
    out: dict[str, Any] = {}
    for k, v in local_ns.items():
        if k.startswith("_") or k in _RESERVED_VARS:
            continue
        # Saltar módulos/funciones (no son data útil para el LLM).
        if hasattr(v, "__call__") or type(v).__name__ == "module":
            continue
        if isinstance(v, (int, float, str, bool)):
            out[k] = v
            continue
        if isinstance(v, (list, tuple)) and len(v) <= 30:
            try:
                out[k] = list(v)[:30]
            except Exception:  # noqa: BLE001
                pass
            continue
        if isinstance(v, dict) and len(v) <= 30:
            try:
                out[k] = {kk: vv for kk, vv in list(v.items())[:30]
                          if isinstance(vv, (int, float, str, bool, list, dict))}
            except Exception:  # noqa: BLE001
                pass
            continue
        if isinstance(v, pd.DataFrame):
            out[k] = {
                "type": "DataFrame",
                "shape": list(v.shape),
                "columns": list(v.columns),
                "head": v.head(8).to_dict(orient="records"),
            }
            continue
        if isinstance(v, pd.Series):
            out[k] = {
                "type": "Series",
                "name": v.name,
                "head": v.head(8).to_dict(),
            }
            continue
    # Cap cantidad total para no inflar el prompt.
    return dict(list(out.items())[:14])


# ---------- ejecución principal --------------------------------------------


def execute_analysis(
    code: str,
    datasets: dict[str, pd.DataFrame],
    *,
    timeout_s: float = 20.0,
) -> AnalysisResult:
    """Ejecuta `code` con datasets/helpers inyectados y devuelve outputs estructurados."""
    if not (code or "").strip():
        return AnalysisResult(ok=False, error="código vacío")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return AnalysisResult(ok=False, error=f"SyntaxError: {e}")

    guard = _AstGuard()
    guard.visit(tree)
    if guard.problems:
        return AnalysisResult(
            ok=False,
            error="Sandbox bloqueó la ejecución:\n" + "\n".join(guard.problems),
        )

    # Plotly es opcional: lo importamos lazy para no romper la GUI si falta.
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception:  # noqa: BLE001
        px = None  # type: ignore[assignment]
        go = None  # type: ignore[assignment]

    artifacts: list[dict] = []

    def publish_artifact(kind: str, title: str | None = None, **kwargs: Any) -> None:
        try:
            artifacts.append(_normalize_artifact(kind, title, kwargs))
        except Exception as e:  # noqa: BLE001
            artifacts.append({
                "kind": "markdown",
                "title": title or "Artifact (error)",
                "text": f"No pude normalizar el artifact: {e}",
            })

    g: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "publish_artifact": publish_artifact,
        "infer_real_estate_fields": infer_real_estate_fields,
    }
    # Datasets read-only (referencias; el agente debe usar .copy() si quiere editar).
    for name, df in (datasets or {}).items():
        if df is not None:
            g[name] = df

    # Ojo: usamos el mismo dict para globals y locals. Si los separamos,
    # las funciones definidas top-level no ven los `import` que el código
    # haga (porque imports irían al local_ns y `__globals__` de la función
    # quedaría apuntando al `g` original). Compartirlo evita ese bug clásico.
    local_ns: dict[str, Any] = g
    buf = io.StringIO()
    started = time.time()

    def runner() -> None:
        with redirect_stdout(buf):
            exec(  # noqa: S102
                compile(tree, "<analysis>", "exec"), g, g,
            )

    ok, val = _run_with_timeout(runner, timeout_s)
    elapsed = time.time() - started

    if not ok:
        import traceback as _tb
        tb_text = ""
        if isinstance(val, BaseException):
            tb_text = "".join(_tb.format_exception(type(val), val, val.__traceback__))
        return AnalysisResult(
            ok=False,
            stdout=buf.getvalue(),
            error=f"{type(val).__name__}: {val}",
            traceback=tb_text[-2000:],
            artifacts=artifacts,
            elapsed_s=elapsed,
        )

    return AnalysisResult(
        ok=True,
        stdout=buf.getvalue(),
        artifacts=artifacts,
        outputs=_serializable_outputs(local_ns),
        elapsed_s=elapsed,
    )


# ---------- compatibilidad con la skill `execute_python` legacy -----------


@dataclass
class SandboxResult:
    """Compat: shape vieja para `gui/skills/code_exec.py`. No usar en flujo nuevo."""

    ok: bool
    stdout: str = ""
    error: str = ""
    figures: list = field(default_factory=list)
    locals_after: dict[str, Any] = field(default_factory=dict)


def execute_python(
    code: str,
    df: pd.DataFrame | None = None,
    extra_locals: dict | None = None,
    timeout_s: float = 10.0,
) -> SandboxResult:
    """Compat layer para la skill legacy `execute_python`.

    Internamente ahora usa `execute_analysis` para mantener una sola
    superficie segura. `df` se expone como `df`; sin matplotlib.
    """
    datasets: dict[str, pd.DataFrame] = {}
    if df is not None:
        datasets["df"] = df
    if extra_locals:
        for k, v in extra_locals.items():
            if isinstance(v, pd.DataFrame):
                datasets[k] = v

    res = execute_analysis(code, datasets, timeout_s=timeout_s)
    locals_after: dict[str, Any] = {}
    for k, v in res.outputs.items():
        if isinstance(v, (int, float, str, bool, list, dict)):
            locals_after[k] = v
    return SandboxResult(
        ok=res.ok,
        stdout=res.stdout,
        error=res.error,
        figures=[],
        locals_after=locals_after,
    )
