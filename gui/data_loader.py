"""Carga datasets y los registra como views en DuckDB.

Self-contained: no importa de entregas/ para que gui/ pueda moverse o
distribuirse sin arrastrar el resto del repo. Replica los filtros E1
del proyecto.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"

CABA_L1 = {"Capital Federal", "Ciudad Autónoma de Buenos Aires"}
PROP_OK = {"departamento", "departamentos", "casa", "casas", "ph", "cochera"}

_MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def _pub_yyyymm(serie: pd.Series) -> pd.Series:
    """Parsea publication_date a entero yyyymm (Spanish robusto)."""
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


def load_train_raw() -> pd.DataFrame:
    """Carga sqlite completo SIN filtros (lazy, ~1.3M filas)."""
    db = DATA_DIR / "entrenamiento.db"
    if not db.exists():
        raise FileNotFoundError(f"No encontré {db}")
    with sqlite3.connect(db) as eng:
        df = pd.read_sql("SELECT * FROM entrenamiento", eng, index_col="id")
    return df


def load_test() -> pd.DataFrame:
    """Carga a_predecir.csv con pub_yyyymm."""
    csv = DATA_DIR / "a_predecir.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No encontré {csv}")
    df = pd.read_csv(csv, index_col="id")
    df["pub_yyyymm"] = _pub_yyyymm(df["publication_date"]).astype("Int64")
    return df


def load_train_filtered(test_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Aplica filtros E1: CABA + venta + USD + tipo de propiedad + precio razonable."""
    raw = load_train_raw()
    if test_df is None:
        test_df = load_test()

    caba_loc2 = set(test_df["location_2"].dropna().unique())
    caba_loc3 = set(test_df["location_3"].dropna().unique())
    mask_caba = (
        raw["location_1"].isin(CABA_L1)
        | (
            raw["location_1"].eq("Buenos Aires")
            & (raw["location_2"].isin(caba_loc2)
               | raw["location_3"].isin(caba_loc3))
        )
    )
    mask = (
        raw["operation_type"].eq("venta")
        & raw["currency_type"].eq("dolares")
        & mask_caba
        & raw["property_type"].isin(PROP_OK)
        & raw["price"].notna()
        & raw["price"].between(5_000, 3_000_000)
    )
    df = raw.loc[mask].copy()
    df["pub_yyyymm"] = _pub_yyyymm(df["publication_date"]).astype("Int64")
    return df


class DataRegistry:
    """Singleton que carga datasets una vez y los registra en DuckDB.

    Patrón:
        reg = DataRegistry.get()
        df = reg.duck.execute("SELECT count(*) FROM train_filtered").fetchdf()

    train_raw es lazy (no se carga al startup); el primer query lo dispara.
    """

    _instance: "DataRegistry | None" = None

    def __init__(self):
        self.duck = duckdb.connect(":memory:")
        self._test: Optional[pd.DataFrame] = None
        self._train_filtered: Optional[pd.DataFrame] = None
        self._train_raw: Optional[pd.DataFrame] = None
        self._eager_load()

    @classmethod
    def get(cls) -> "DataRegistry":
        if cls._instance is None:
            cls._instance = DataRegistry()
        return cls._instance

    def _eager_load(self):
        """Carga test + train_filtered al startup (los más usados)."""
        self._test = load_test()
        self._train_filtered = load_train_filtered(self._test)
        self.duck.register("test", self._test.reset_index())
        self.duck.register("train_filtered", self._train_filtered.reset_index())

    def _ensure_train_raw(self):
        """Lazy load del raw cuando algún query lo necesita."""
        if self._train_raw is None:
            self._train_raw = load_train_raw()
            self.duck.register("train_raw", self._train_raw.reset_index())

    def query(self, sql: str) -> pd.DataFrame:
        """Ejecuta SQL. Si menciona train_raw, lo carga lazy primero."""
        sql_upper = sql.upper()
        if "TRAIN_RAW" in sql_upper:
            self._ensure_train_raw()
        return self.duck.execute(sql).fetchdf()

    def list_tables(self) -> list[str]:
        return ["train_filtered", "test", "train_raw (lazy)"]

    def schema(self, table: str) -> pd.DataFrame:
        if table == "train_raw":
            self._ensure_train_raw()
        try:
            return self.duck.execute(f"DESCRIBE {table}").fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame()

    def table(self, name: str) -> pd.DataFrame:
        if name == "train_filtered":
            return self._train_filtered
        if name == "test":
            return self._test
        if name == "train_raw":
            self._ensure_train_raw()
            return self._train_raw
        raise ValueError(f"Tabla desconocida: {name}")
