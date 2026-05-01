"""Contexto REAL del dataset para el system prompt.

Lo que ve el LLM tiene que ser fiel al dataset que efectivamente está cargado
en memoria: nombre exacto de columnas, dtype, % de nulos, muestras de texto
libre, barrios disponibles, cobertura temporal y stats de precio.

Este módulo se ejecuta UNA vez al primer pedido (cacheado en memoria) y se
puede invalidar con `reset_data_context_cache()` si el dataset se recarga.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd

from .data_loader import DataRegistry

_CACHED: Optional[str] = None
_CACHED_FACTS: Optional[dict] = None


def reset_data_context_cache() -> None:
    global _CACHED, _CACHED_FACTS
    _CACHED = None
    _CACHED_FACTS = None


# ---------- helpers de inspección ------------------------------------------


def _schema_lines(df: pd.DataFrame, max_cols: int = 24) -> list[str]:
    out: list[str] = []
    for c in df.columns[:max_cols]:
        nulls_pct = df[c].isna().mean() * 100
        dtype = str(df[c].dtype)
        out.append(f"  - {c} ({dtype}, nulls={nulls_pct:.1f}%)")
    if len(df.columns) > max_cols:
        out.append(f"  - ... +{len(df.columns) - max_cols} columnas más")
    return out


def _short(s: str, n: int = 220) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _topn(s: pd.Series, n: int = 12) -> list[tuple[str, int]]:
    vc = s.dropna().astype(str).value_counts().head(n)
    return [(idx, int(cnt)) for idx, cnt in vc.items()]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip()


# ---------- detección semi-automática de "ambientes" / "m2" en texto ----


# Strings sin grupos para usar con str.contains (no .extract).
_PAT_M2 = r"\d{1,4}\s*(?:m2|m²|mts2|mt2|metros\b)"
_PAT_AMB = r"\d{1,2}\s*(?:ambientes?|amb\b)"
_PAT_DORM = r"\d{1,2}\s*(?:dormitorios?|habitaciones?)"


def _detect_text_field_coverage(df: pd.DataFrame, sample_n: int = 4000) -> dict:
    """% de filas con un patrón m2 / ambientes encontrable en `description+features`."""
    n = min(sample_n, len(df))
    if n == 0:
        return {"sample_n": 0, "pct_m2": 0.0, "pct_amb": 0.0, "pct_dorm": 0.0}
    sub = df.sample(n=n, random_state=0)
    text = pd.Series("", index=sub.index, dtype=object)
    for c in ("features", "description", "title"):
        if c in sub.columns:
            text = text.str.cat(sub[c].fillna("").astype(str), sep=" ")
    text = text.str.lower()
    return {
        "sample_n": int(n),
        "pct_m2": round(float(text.str.contains(_PAT_M2, na=False, regex=True).mean() * 100), 1),
        "pct_amb": round(float(text.str.contains(_PAT_AMB, na=False, regex=True).mean() * 100), 1),
        "pct_dorm": round(float(text.str.contains(_PAT_DORM, na=False, regex=True).mean() * 100), 1),
    }


# ---------- bloque principal ----------------------------------------------


def data_context_facts() -> dict:
    """Snapshot estructurado del estado actual de los datasets."""
    global _CACHED_FACTS
    if _CACHED_FACTS is not None:
        return _CACHED_FACTS

    reg = DataRegistry.get()
    train = reg.table("train_filtered")
    test = reg.table("test")

    facts: dict = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_columns": list(train.columns),
        "test_columns": list(test.columns),
    }

    # Price (USD) stats sobre train
    if "price" in train.columns:
        p = pd.to_numeric(train["price"], errors="coerce").dropna()
        facts["price"] = {
            "min": float(p.min()),
            "p10": float(p.quantile(0.10)),
            "p25": float(p.quantile(0.25)),
            "median": float(p.median()),
            "p75": float(p.quantile(0.75)),
            "p90": float(p.quantile(0.90)),
            "max": float(p.max()),
            "n": int(len(p)),
        }

    # Cobertura temporal
    if "pub_yyyymm" in train.columns:
        ym = pd.to_numeric(train["pub_yyyymm"], errors="coerce").dropna().astype(int)
        facts["pub_yyyymm_train"] = {
            "min": int(ym.min()) if len(ym) else None,
            "max": int(ym.max()) if len(ym) else None,
            "by_year": {int(y): int(c)
                        for y, c in (ym // 100).value_counts().sort_index().items()},
        }
    if "pub_yyyymm" in test.columns:
        ym = pd.to_numeric(test["pub_yyyymm"], errors="coerce").dropna().astype(int)
        facts["pub_yyyymm_test"] = {
            "min": int(ym.min()) if len(ym) else None,
            "max": int(ym.max()) if len(ym) else None,
        }

    # Categóricas relevantes
    for col in ("property_type", "operation_type", "currency_type",
                "location_1", "location_2", "location_3"):
        if col in train.columns:
            facts.setdefault("top_values", {})[col] = _topn(train[col], 12)

    # Muestras de texto libre
    samples: dict[str, list[str]] = {}
    for c in ("features", "description"):
        if c in train.columns:
            samples[c] = [_short(v, 220)
                          for v in train[c].dropna().head(3).tolist()]
    facts["text_samples"] = samples

    # Cobertura de patrones rooms/m2 en texto
    facts["text_extraction"] = _detect_text_field_coverage(train)

    _CACHED_FACTS = facts
    return facts


def data_context_block(max_chars: int = 3500) -> str:
    """Markdown compacto con TODO lo que el LLM debe saber del dataset."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    f = data_context_facts()

    parts: list[str] = []
    parts.append("=== ESTADO ACTUAL DE LOS DATASETS (snapshot real) ===")
    parts.append(
        f"- `train_filtered`: {f['train_shape'][0]:,} filas × "
        f"{f['train_shape'][1]} columnas (filtros E1: CABA + venta + USD + "
        f"property_type ok + price razonable)."
    )
    parts.append(
        f"- `test`: {f['test_shape'][0]:,} filas × {f['test_shape'][1]} columnas "
        "(a_predecir, sin price)."
    )

    parts.append("\nColumnas REALES de `train_filtered`:")
    reg = DataRegistry.get()
    train = reg.table("train_filtered")
    parts.extend(_schema_lines(train, max_cols=24))

    parts.append(
        "\n⚠️ Importante: NO existen columnas estructuradas como `rooms`, "
        "`bedrooms`, `bathrooms`, `surface_total` ni `surface_covered`.\n"
        "Esa información está dentro de `features` y `description` (texto libre)."
    )

    if "price" in f:
        p = f["price"]
        parts.append("\nDistribución de `price` (USD) en train:")
        parts.append(
            f"  n={p['n']:,} | min={p['min']:.0f} | p25={p['p25']:.0f} | "
            f"mediana={p['median']:.0f} | p75={p['p75']:.0f} | "
            f"p90={p['p90']:.0f} | max={p['max']:.0f}"
        )

    if "pub_yyyymm_train" in f:
        ym = f["pub_yyyymm_train"]
        by_year = ym.get("by_year") or {}
        last_5 = sorted(by_year.items())[-6:]
        parts.append(
            f"\nCobertura temporal `pub_yyyymm` train: {ym['min']} → {ym['max']}. "
            f"NB: ~87% de filas tienen pub_yyyymm null. Conteo por año (últimos): "
            + ", ".join(f"{y}={c:,}" for y, c in last_5)
        )
    if "pub_yyyymm_test" in f:
        ymt = f["pub_yyyymm_test"]
        parts.append(
            f"Cobertura temporal `pub_yyyymm` test: {ymt['min']} → {ymt['max']}."
        )

    if "top_values" in f:
        for col in ("property_type", "location_2", "location_3"):
            if col in f["top_values"]:
                pares = f["top_values"][col][:8]
                parts.append(
                    f"\nTop valores de `{col}`: "
                    + ", ".join(f"{v} ({n:,})" for v, n in pares)
                )

    if f.get("text_samples"):
        parts.append("\nMuestras de TEXTO libre (para entender formato):")
        for c, vals in f["text_samples"].items():
            parts.append(f"\n`{c}` (3 muestras):")
            for v in vals:
                parts.append(f"  · {v}")

    if "text_extraction" in f:
        e = f["text_extraction"]
        parts.append(
            f"\nCobertura de patrones en texto (sample n={e['sample_n']}): "
            f"`m2` aparece en ~{e['pct_m2']}%, `ambientes` en ~{e['pct_amb']}%, "
            f"`dormitorios` en ~{e['pct_dorm']}% de las filas."
        )
        parts.append(
            "El sandbox expone `infer_real_estate_fields(df)` que ya hace esa "
            "extracción de manera vectorizada y devuelve copia con columnas "
            "`_rooms_est` y `_surface_m2_est` (None si no detectó)."
        )

    parts.append(
        "\nGuía operativa para preguntas tipo 'cuánto vale / qué es buen precio':\n"
        "  1) Empezá llamando `df = infer_real_estate_fields(train_filtered)`.\n"
        "  2) Filtrá por barrio: usá normalización lower+sin tildes contra "
        "`location_2` y `location_3`. Match por `str.contains` con `regex=False`.\n"
        "  3) Filtrá `property_type` con `str.contains('departamento'|'casa'|'ph', "
        "regex=True, na=False)` según corresponda.\n"
        "  4) Filtrá ambientes/superficie usando `_rooms_est` y `_surface_m2_est`.\n"
        "  5) Si filtrás por año/`pub_yyyymm` y la muestra queda chica (<25), "
        "ampliá ventana o quitalo y comentalo en `print`.\n"
        "  6) Calculá `q25/median/q75` de `price`, mediana de `price/_surface_m2_est`, "
        "publicá `metrics`+`table`+`chart` y respondé el rango óptimo."
    )

    parts.append("=== FIN SNAPSHOT ===")

    block = "\n".join(parts)
    if len(block) > max_chars:
        block = block[: max_chars - 3] + "..."

    _CACHED = block
    return block


__all__ = [
    "data_context_block",
    "data_context_facts",
    "reset_data_context_cache",
]


_ = (np, re)  # mantener imports usados a futuro
