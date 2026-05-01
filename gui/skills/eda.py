"""Primitivas genéricas de EDA.

Notas:

- El agente Code-First (`gui/cognitive.py`) NO consume estas skills por nombre:
  ahora genera código Python que corre en sandbox. Las funciones de acá
  se mantienen como utilidades reutilizables para otros panels y para
  inspección manual desde notebooks/REPL.
- Mantenemos sólo primitivas genéricas (`describe_data`, `group_summary`,
  correlaciones, outliers, comparaciones por grupo, drift temporal,
  feature importance). Eliminamos lo que era demasiado específico
  (p. ej. `market_comps`): ese tipo de análisis ahora lo construye el
  agente con un script ad-hoc en lugar de inflar el catálogo de skills.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import register
from ..data_loader import DataRegistry


def _get_df(table: str) -> pd.DataFrame:
    return DataRegistry.get().table(table)


# ---------- describe --------------------------------------------------------

@register(
    "describe_data",
    "Stats descriptivos sobre una tabla. Si no se pasan columnas, todas las numéricas.",
    {"table": "train_filtered|test|train_raw", "columns": "list[str]?"},
)
def describe_data(table: str = "train_filtered", columns: list[str] | None = None):
    df = _get_df(table)
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    desc = df.describe(include="all").transpose()
    desc["dtype"] = df.dtypes.astype(str)
    desc["nulls"] = df.isna().sum()
    desc["null_pct"] = (df.isna().mean() * 100).round(2)
    return desc.reset_index().rename(columns={"index": "column"})


# ---------- correlations ----------------------------------------------------

@register(
    "compute_correlations",
    "Correlación de Spearman entre features numéricas (o vs target).",
    {"table": "string", "target": "string?", "top_k": "int (default 20)"},
)
def compute_correlations(
    table: str = "train_filtered", target: str | None = "price", top_k: int = 20,
):
    df = _get_df(table)
    num = df.select_dtypes(include=[np.number])
    if target and target in num.columns:
        corr = num.corr(method="spearman")[target].drop(target).sort_values(
            key=lambda s: s.abs(), ascending=False
        )
        out = corr.head(top_k).reset_index()
        out.columns = ["feature", f"spearman_vs_{target}"]
        return out
    return num.corr(method="spearman").round(3)


# ---------- outliers --------------------------------------------------------

@register(
    "detect_outliers",
    "Detecta outliers (IQR o zscore) sobre una columna numérica.",
    {"table": "string", "column": "string", "method": "iqr|zscore"},
)
def detect_outliers(table: str, column: str, method: str = "iqr"):
    df = _get_df(table)
    s = df[column].dropna()
    if method == "iqr":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
    elif method == "zscore":
        z = (s - s.mean()) / s.std(ddof=0)
        lo, hi = -3.0, 3.0
        mask = z.abs() > 3
    else:
        raise ValueError("method debe ser iqr o zscore")
    return {
        "method": method,
        "lo": float(lo), "hi": float(hi),
        "n_total": int(len(s)),
        "n_outliers": int(mask.sum()),
        "pct_outliers": round(mask.mean() * 100, 2),
        "examples": s[mask].head(5).tolist(),
    }


# ---------- compare distributions ------------------------------------------

@register(
    "compare_distributions",
    "Compara distribución de `column` entre subgrupos definidos por group_by. "
    "Devuelve stats por grupo + mediana/mean.",
    {"table": "string", "column": "string", "group_by": "string"},
)
def compare_distributions(table: str, column: str, group_by: str):
    df = _get_df(table)
    g = df.groupby(group_by)[column].agg(["count", "mean", "median", "std"])
    g["null_pct"] = (
        df.groupby(group_by)[column].apply(lambda s: s.isna().mean() * 100).round(2)
    )
    return g.sort_values("count", ascending=False).head(30).reset_index()


# ---------- group summary --------------------------------------------------

@register(
    "group_summary",
    "Agrega una métrica numérica por cualquier columna categórica y ordena el ranking.",
    {
        "table": "string default train_filtered",
        "value": "columna numérica, ej. price",
        "group_by": "columna categórica, ej. location_3",
        "top_k": "int default 15",
        "min_n": "int default 30",
        "ascending": "bool default false",
    },
)
def group_summary(
    table: str = "train_filtered",
    value: str = "price",
    group_by: str = "location_3",
    top_k: int = 15,
    min_n: int = 30,
    ascending: bool = False,
):
    """Agregación genérica por grupo: n, mean, median, std, q25, q75."""
    df = _get_df(table)
    if value not in df.columns:
        return {"error": f"{value} no está en {table}"}
    if group_by not in df.columns:
        return {"error": f"{group_by} no está en {table}"}

    work = df[[group_by, value]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    valid = work[work[group_by].notna() & work[value].notna()]
    if valid.empty:
        return {"error": f"sin datos válidos para {value} por {group_by}"}

    out = (
        valid.groupby(group_by)[value]
        .agg(
            n="size",
            mean="mean",
            median="median",
            std="std",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
        .rename(columns={group_by: "group"})
    )
    out = out[out["n"] >= int(min_n)].copy()
    out = out.sort_values("median", ascending=bool(ascending)).head(int(top_k))
    for col in ["mean", "median", "std", "q25", "q75"]:
        out[col] = out[col].round(2)
    return {
        "table": table,
        "value": value,
        "group_by": group_by,
        "min_n": min_n,
        "sorted_by": "median",
        "rows": out.to_dict(orient="records"),
    }


# ---------- temporal drift -------------------------------------------------

@register(
    "temporal_drift_check",
    "Compara distribución de `column` entre train (último período) y test. "
    "Devuelve stats agregadas y aviso de drift si pct_diff_mean > 15%.",
    {"column": "string"},
)
def temporal_drift_check(column: str):
    reg = DataRegistry.get()
    train = reg.table("train_filtered")
    test = reg.table("test")
    if column not in train.columns or column not in test.columns:
        return {"error": f"{column} no está en train+test"}
    s_tr = pd.to_numeric(train[column], errors="coerce").dropna()
    s_te = pd.to_numeric(test[column], errors="coerce").dropna()
    if s_tr.empty or s_te.empty:
        return {"error": "columna sin valores numéricos en alguno de los dos"}
    drift = {
        "train_mean": float(s_tr.mean()), "test_mean": float(s_te.mean()),
        "train_median": float(s_tr.median()), "test_median": float(s_te.median()),
        "train_std": float(s_tr.std()), "test_std": float(s_te.std()),
        "train_n": int(len(s_tr)), "test_n": int(len(s_te)),
    }
    drift["pct_diff_mean"] = round(
        (drift["test_mean"] - drift["train_mean"]) / max(abs(drift["train_mean"]), 1e-9) * 100, 2,
    )
    drift["alert"] = abs(drift["pct_diff_mean"]) > 15
    return drift


# ---------- feature importance ---------------------------------------------

@register(
    "feature_importance",
    "Random Forest rápido (small) para rankear importancia de features sobre price. "
    "Usa subsample para que sea rápido (<10s).",
    {"features": "list[str]", "n_samples": "int (default 20000)"},
)
def feature_importance(features: list[str], n_samples: int = 20000):
    from sklearn.ensemble import RandomForestRegressor
    df = _get_df("train_filtered")
    cols = [c for c in features if c in df.columns]
    if not cols or "price" not in df.columns:
        return {"error": "features inválidas o falta price"}
    sub = df[cols + ["price"]].dropna()
    if len(sub) > n_samples:
        sub = sub.sample(n_samples, random_state=42)
    X = sub[cols].select_dtypes(include=[np.number])
    if X.empty:
        return {"error": "ninguna feature numérica"}
    y = sub.loc[X.index, "price"]
    rf = RandomForestRegressor(n_estimators=80, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.reset_index().rename(columns={"index": "feature", 0: "importance"})
