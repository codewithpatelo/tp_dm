"""drift_correct.py — Corrección de data drifting (starter para Entrega Final).

Adaptación a Python + regresión real-estate del workflow anti-drift visto en
Laboratorio de Implementación II (materia hermana de DM UBA Exactas).

Original (R): z1401_DR_corregir_drifting.r — corregía drift en variables
monetarias de un dataset bancario para clasificar churn (BAJA+2 a 2 meses).
Este port adapta los mismos 7 métodos a:

  - regresión continua de price USD (en vez de clasificación binaria),
  - período = pub_yyyymm parseado de publication_date (en vez de foto_mes),
  - Pandas + numpy (en vez de data.table),
  - sin leakage train↔test: estadísticas se ajustan SOLO en train y se
    aplican a ambos (el original mezclaba todo).

================================================================================
Filosofía
================================================================================

El dataset cubre publication_date entre oct-2021 y jun-2026, con la holdout de
Kaggle entera en 2026 (ver entregas/entrega_3/error_analysis_v4.md: ~85 % del
RMSE viene de 2 segmentos cuyo principal denominador común es la separación
temporal). Argentina pasó por años de fuerte inflación + variación del dólar
paralelo en ese período → las variables monetarias (price y derivadas) NO son
directamente comparables entre 2022 y 2026 sin algún tipo de normalización.

Este módulo NO decide cuál método aplicar — eso es decisión empírica que se
hace en EF comparando RMSE de cada variante en holdout temporal. Lo que sí
provee es:

  1. Una biblioteca de 7 métodos de corrección, todos parametrizados por
     columna de período.
  2. Aplicación segura sin leakage: estadísticas se ajustan SOLO en train,
     transformación se aplica a train y test.
  3. CLI para correr cada método y persistir un dataset corregido por método.

================================================================================
Métodos disponibles
================================================================================

| metodo            | qué hace                                          | requiere |
|-------------------|---------------------------------------------------|----------|
| rank_simple       | Rank percentil [0, 1] por período                 | nada     |
| rank_cero_fijo    | Rank que mantiene 0 → 0 (positivos / negativos    | nada     |
|                   | rankeados por separado)                           |          |
| estandarizar      | Z-score por período (mu/sigma del train)          | nada     |
| deflacion         | Multiplica por IPC del período                    | indices  |
| dolar_oficial     | Divide por FX oficial del período                 | indices  |
| dolar_blue        | Divide por FX paralelo (blue/MEP) del período     | indices  |
| uva               | Multiplica por factor UVA del período             | indices  |

Métodos `rank_*` y `estandarizar` corren sin datos externos.

Métodos macro (deflacion, dolar_*, uva) requieren un YAML con la serie por
yyyymm (ver entregas/entrega_f/README.md sección "indices_macro_arg.yml" para
el formato esperado y las fuentes públicas de cada serie).

================================================================================
Uso
================================================================================

CLI:

    python entregas/entrega_f/drift_correct.py --metodo rank_simple --cols price
    python entregas/entrega_f/drift_correct.py --metodo dolar_blue --cols price \\
        --indices entregas/entrega_f/indices_macro_arg.yml

Programático:

    from entregas.entrega_f.drift_correct import (
        load_train_test_filtered, apply_drift_correction,
    )

    train, test = load_train_test_filtered()
    train_c, test_c = apply_drift_correction(
        train, test, cols=["price"], metodo="rank_simple",
        period_col="pub_yyyymm",
    )
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Carga + filtros E1 (autocontenido — duplicado intencional respecto a
# entregas/entrega_3/eda_features_v4.py para que entrega_f no dependa de E3).
# ---------------------------------------------------------------------------

CABA_L1 = {"Capital Federal", "Ciudad Autónoma de Buenos Aires"}
PROP_OK = {"departamento", "departamentos", "casa", "casas", "ph", "cochera"}

_MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def _pub_yyyymm(serie: pd.Series) -> pd.Series:
    """Parsea publication_date a entero yyyymm (formato Spanish robusto).

    Reconoce dos formatos: "23 de enero de 2026" y "2026-01-23" / "2026/01".
    Devuelve NaN para los que no matchean.
    """
    out = pd.Series(np.nan, index=serie.index, dtype=float)
    for i, raw in serie.items():
        if pd.isna(raw):
            continue
        s = str(raw).lower()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        m = re.search(r"(\d+)\s+de\s+([a-z]+)\s+(?:de\s+)?(\d{4})", s)
        if m:
            month = _MESES_ES.get(m.group(2))
            if month:
                out.at[i] = int(m.group(3)) * 100 + month
                continue
        m = re.search(r"(\d{4})[-/](\d{1,2})", s)
        if m:
            out.at[i] = int(m.group(1)) * 100 + int(m.group(2))
    return out


def load_train_test_filtered() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga train + test, filtra train con criterios de E1, agrega pub_yyyymm.

    Devuelve dos DataFrames CON LAS COLUMNAS CRUDAS del dataset original
    (sin parsing de features, sin imputación). El parsing avanzado se hace
    aguas abajo en EF, según se decida.
    """
    print("[load] leyendo sqlite + csv...", flush=True)
    t0 = time.time()
    eng = sqlite3.connect(DATA / "entrenamiento.db")
    df_ent = pd.read_sql("SELECT * FROM entrenamiento", eng, index_col="id")
    eng.close()
    df_ap = pd.read_csv(DATA / "a_predecir.csv", index_col="id")
    print(f"[load] train: {df_ent.shape}  test: {df_ap.shape}  ({time.time()-t0:.1f}s)")

    caba_loc2 = set(df_ap["location_2"].dropna().unique())
    caba_loc3 = set(df_ap["location_3"].dropna().unique())
    mask_caba = (
        df_ent["location_1"].isin(CABA_L1)
        | (
            df_ent["location_1"].eq("Buenos Aires")
            & (df_ent["location_2"].isin(caba_loc2)
               | df_ent["location_3"].isin(caba_loc3))
        )
    )
    mask = (
        df_ent["operation_type"].eq("venta")
        & df_ent["currency_type"].eq("dolares")
        & mask_caba
        & df_ent["property_type"].isin(PROP_OK)
        & df_ent["price"].notna()
        & df_ent["price"].between(5_000, 3_000_000)
    )
    df_ent = df_ent.loc[mask].copy()
    df_ent["pub_yyyymm"] = _pub_yyyymm(df_ent["publication_date"]).astype("Int64")
    df_ap["pub_yyyymm"] = _pub_yyyymm(df_ap["publication_date"]).astype("Int64")
    print(f"[load] train post-filtro: {df_ent.shape}")
    print(f"[load] cobertura pub_yyyymm: train {df_ent['pub_yyyymm'].notna().mean()*100:.1f}%  "
          f"test {df_ap['pub_yyyymm'].notna().mean()*100:.1f}%")
    return df_ent, df_ap


# ---------------------------------------------------------------------------
# Métodos de corrección (port directo del R, adaptado a pandas)
# ---------------------------------------------------------------------------
# Convención: cada método recibe (df_train, df_test, cols, period_col, **kwargs)
# y devuelve (df_train_out, df_test_out) con las cols originales reemplazadas
# por su versión corregida (mismo nombre — pandas lo maneja más natural sin
# renombrar a `<col>_rank` / `<col>_normal` como hacía el R).
# ---------------------------------------------------------------------------


def drift_rank_simple(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    cols: list[str], period_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank percentil [0, 1] por período. Robusto a cualquier shift monetario.

    Sin leakage: el rank de cada fila depende SOLO de las otras filas del
    MISMO dataset y MISMO período. Train y test no se mezclan.

    Trade-off: pierde la unidad monetaria → bueno para drift, malo si el
    modelo necesita la magnitud absoluta para algún corte (ej. "propiedades
    > 500 K USD"). Más seguro para RF / GBM / kNN; más cuestionable para
    redes que asumen distribución continua acotada.
    """
    out_train, out_test = df_train.copy(), df_test.copy()
    for c in cols:
        out_train[c] = out_train.groupby(period_col)[c].rank(method="min", pct=True)
        out_test[c] = out_test.groupby(period_col)[c].rank(method="min", pct=True)
    return out_train, out_test


def drift_rank_cero_fijo(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    cols: list[str], period_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank que mantiene 0 → 0; positivos y negativos rankeados por separado.

    Útil para variables monetarias con 0 semánticamente distinto (ej. saldo
    en cuenta = 0 → cuenta inactiva, distinto de un valor chico positivo).
    Para `price` NO aplica (price > 0 siempre); se incluye por completitud
    del port y para futuras features derivadas con signo (ej. delta de precio
    entre listings, balance financiero, etc.).
    """
    def _rank_per_group(serie: pd.Series, period: pd.Series) -> pd.Series:
        out = pd.Series(0.0, index=serie.index)
        df = pd.DataFrame({"v": serie, "p": period})
        for _, g in df.groupby("p"):
            v = g["v"]
            n = len(v)
            if n == 0:
                continue
            pos = v[v > 0]
            neg = v[v < 0]
            if len(pos):
                out.loc[pos.index] = pos.rank(method="min") / n
            if len(neg):
                out.loc[neg.index] = -(-neg).rank(method="min") / n
        return out

    out_train, out_test = df_train.copy(), df_test.copy()
    for c in cols:
        out_train[c] = _rank_per_group(out_train[c], out_train[period_col])
        out_test[c] = _rank_per_group(out_test[c], out_test[period_col])
    return out_train, out_test


def drift_estandarizar(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    cols: list[str], period_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score por período. Estadísticas (mu, sigma) calculadas SOLO en train,
    transformación aplicada a train y test (sin leakage).

    Diferencia con el R original: el original calculaba mean / sd dentro del
    propio dataset (mezclando train+test si estuvieran juntos). Acá lo hacemos
    fit-on-train + transform-on-both, consistente con la regla de
    "Procesar train y test en simultáneo, NUNCA mergearlos antes de procesar"
    del CONTEXT.md.

    Si un período del test no aparece en train, ese período se imputa con
    NaN (válido para Pandas, hay que decidir downstream qué hacer).
    """
    out_train, out_test = df_train.copy(), df_test.copy()
    for c in cols:
        stats = (
            df_train.groupby(period_col)[c]
            .agg(["mean", "std"])
            .rename(columns={"mean": "mu", "std": "sigma"})
        )
        for df_out in (out_train, out_test):
            mu = df_out[period_col].map(stats["mu"])
            sigma = df_out[period_col].map(stats["sigma"]).replace(0, np.nan)
            df_out[c] = (df_out[c] - mu) / sigma
    return out_train, out_test


def _aplicar_indice_macro(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    cols: list[str], period_col: str,
    indices: dict[int, float], op: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Helper común para métodos macro. op ∈ {'mul', 'div'}."""
    if op not in {"mul", "div"}:
        raise ValueError(f"op debe ser 'mul' o 'div', recibí {op!r}")
    out_train, out_test = df_train.copy(), df_test.copy()
    for df_out in (out_train, out_test):
        idx = df_out[period_col].map(indices)
        if idx.isna().any():
            faltantes = sorted(set(df_out.loc[idx.isna(), period_col].dropna()))
            raise ValueError(
                f"faltan índices para los períodos: {faltantes[:10]}"
                f"{'...' if len(faltantes) > 10 else ''}. "
                f"Completar la serie en el YAML antes de correr este método."
            )
        for c in cols:
            df_out[c] = df_out[c] * idx if op == "mul" else df_out[c] / idx
    return out_train, out_test


def drift_deflacion(df_train, df_test, cols, period_col, indices_ipc):
    """Multiplica por IPC del período → escala en pesos constantes.

    Asume que `indices_ipc[yyyymm]` está normalizado a 1.0 para el mes de
    referencia (ej. dic-2024 = 1.0 → meses anteriores > 1.0, posteriores < 1.0,
    siguiendo la convención del R original).
    """
    return _aplicar_indice_macro(df_train, df_test, cols, period_col, indices_ipc, "mul")


def drift_dolar_oficial(df_train, df_test, cols, period_col, indices_dolar):
    """Divide por dólar oficial → escala en USD oficial.

    Para nuestro dataset, `price` ya está en USD según el filtro
    `currency_type = 'dolares'` → este método sólo tiene sentido si después
    de E4 incorporamos features monetarias en pesos (ej. expensas, valor del
    metro cuadrado por barrio en pesos, etc.).
    """
    return _aplicar_indice_macro(df_train, df_test, cols, period_col, indices_dolar, "div")


def drift_dolar_blue(df_train, df_test, cols, period_col, indices_dolar):
    """Divide por dólar paralelo (blue / MEP) del período → escala en USD blue.

    Mismo razonamiento que dolar_oficial. Útil además para CONVERTIR el price
    en USD oficiales (que es la unidad del dataset) a USD-equivalente-paralelo,
    si el inversor real está pensando en blue (típico en Argentina).
    """
    return _aplicar_indice_macro(df_train, df_test, cols, period_col, indices_dolar, "div")


def drift_uva(df_train, df_test, cols, period_col, indices_uva):
    """Multiplica por UVA → escala constante (cesta CER del BCRA).

    Útil para features expresadas en pesos donde queremos abstraer la inflación
    "real" (con CER) en lugar de la del IPC general.
    """
    return _aplicar_indice_macro(df_train, df_test, cols, period_col, indices_uva, "mul")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

METODOS = {
    "rank_simple":    drift_rank_simple,
    "rank_cero_fijo": drift_rank_cero_fijo,
    "estandarizar":   drift_estandarizar,
    "deflacion":      drift_deflacion,
    "dolar_oficial":  drift_dolar_oficial,
    "dolar_blue":     drift_dolar_blue,
    "uva":            drift_uva,
}

REQUIERE_INDICES = {"deflacion", "dolar_oficial", "dolar_blue", "uva"}


def apply_drift_correction(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    cols: list[str], metodo: str, period_col: str = "pub_yyyymm",
    indices: dict[int, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Punto de entrada programático. Despacha al método correspondiente."""
    if metodo not in METODOS:
        raise ValueError(
            f"método desconocido {metodo!r}. Disponibles: {sorted(METODOS)}"
        )
    fn = METODOS[metodo]
    if metodo in REQUIERE_INDICES:
        if indices is None:
            raise ValueError(f"método {metodo!r} requiere parámetro `indices`")
        return fn(df_train, df_test, cols, period_col, indices)
    return fn(df_train, df_test, cols, period_col)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_indices_yaml(path: Path, key: str) -> dict[int, float]:
    """Carga indices_macro_arg.yml y devuelve {yyyymm: valor} para `key`."""
    try:
        import yaml
    except ImportError:
        sys.exit("[drift_correct] falta paquete `pyyaml` (pip install pyyaml)")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    serie = raw.get(key)
    if serie is None:
        raise KeyError(
            f"clave {key!r} no encontrada en {path}. "
            f"Claves disponibles: {sorted(raw)}"
        )
    return {int(yyyymm): float(v) for yyyymm, v in serie.items()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aplica un método de corrección de data drifting "
                    "y persiste el dataset corregido."
    )
    p.add_argument("--metodo", required=True, choices=sorted(METODOS))
    p.add_argument(
        "--cols", nargs="+", default=["price"],
        help="columnas monetarias a corregir (default: price)",
    )
    p.add_argument(
        "--indices", type=Path, default=None,
        help="YAML con índices macro (requerido para deflacion/dolar_*/uva)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="prefijo de salida (default: dataset_corregido_<metodo>)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    print(f"[drift_correct] método: {args.metodo}  cols: {args.cols}")

    df_train, df_test = load_train_test_filtered()

    indices = None
    if args.metodo in REQUIERE_INDICES:
        if args.indices is None:
            sys.exit(
                f"[drift_correct] método {args.metodo!r} requiere --indices "
                f"(YAML con la serie macro). Ver README de entrega_f."
            )
        key_map = {
            "deflacion":     "ipc",
            "dolar_oficial": "dolar_oficial",
            "dolar_blue":    "dolar_blue",
            "uva":           "uva",
        }
        indices = _load_indices_yaml(args.indices, key_map[args.metodo])
        print(f"[drift_correct] cargados {len(indices)} índices de {args.indices}")

    train_c, test_c = apply_drift_correction(
        df_train, df_test, cols=args.cols, metodo=args.metodo, indices=indices,
    )

    prefix = args.out or (OUT_DIR / f"dataset_corregido_{args.metodo}")
    train_path = Path(f"{prefix}.train.parquet")
    test_path = Path(f"{prefix}.test.parquet")
    train_c.to_parquet(train_path)
    test_c.to_parquet(test_path)
    print(f"[drift_correct] grabado:\n  {train_path}\n  {test_path}")

    md = [f"# Reporte drift_correct — método `{args.metodo}`\n"]
    md.append(f"**Columnas transformadas**: {', '.join(args.cols)}")
    md.append(
        f"**Períodos en train**: {df_train['pub_yyyymm'].nunique()} "
        f"(min {df_train['pub_yyyymm'].min()}, max {df_train['pub_yyyymm'].max()})"
    )
    md.append(
        f"**Períodos en test**: {df_test['pub_yyyymm'].nunique()} "
        f"(min {df_test['pub_yyyymm'].min()}, max {df_test['pub_yyyymm'].max()})\n"
    )
    for c in args.cols:
        md.append(f"\n## Mediana de `{c}` por período (post-corrección)")
        med_train = train_c.groupby("pub_yyyymm")[c].median().rename("train_mediana")
        med_test = test_c.groupby("pub_yyyymm")[c].median().rename("test_mediana")
        tabla = pd.concat([med_train, med_test], axis=1).round(4)
        try:
            md.append(tabla.to_markdown())
        except ImportError:
            # Fallback si tabulate no está instalado
            md.append("```\n" + tabla.to_string() + "\n```")

    md_path = Path(f"{prefix}.report.md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
