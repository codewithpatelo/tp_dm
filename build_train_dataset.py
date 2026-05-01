"""
Genera datasets/train_clean.parquet con el pipeline del colab campeón (entrega_2/v4).

Pipeline:
  1. Filtrado CABA + venta + dolares + property_type + rango de precio.
  2. Parseo de la columna `features` → m2, n_dormitorios, n_banos, flags f_*.
  3. Atributos extra: len_descripcion, n_features, barrio.
  4. Winsorización a NaN fuera de rangos razonables (m2, dormitorios, baños).
  5. IsolationForest sobre (price, m2, n_dormitorios, n_banos) — contamination=0.01.
  6. Indicador de ausencia m2_was_na.
  7. Imputación: mediana global para numéricas, moda para categóricas (fit sólo en train).
  8. Factorización de barrio y property_type para compatibilidad con RandomForest.
  9. Salida: datasets/train_clean.parquet (y opcionalmente train_clean.csv con --csv).
"""

import argparse
import re
import sqlite3
import unicodedata

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DIR = "datasets"

CABA_L1 = {"Capital Federal", "Ciudad Autónoma de Buenos Aires"}
PROP_OK  = {"departamento", "departamentos", "casa", "casas", "ph", "cochera"}

AMENITIES = [
    "balcon", "garage", "cochera", "pileta", "parrilla",
    "aire acondicionado", "calefaccion", "gas natural",
    "internet", "seguridad", "alarma", "gimnasio", "jardin",
    "cuarto de servicio", "cocina equipada", "bodega",
]

NUM_MED  = ["m2", "lat", "lon", "n_dormitorios", "n_banos", "len_descripcion", "n_features"]
CAT_MODE = ["property_type", "barrio"]


def parse_features(serie: pd.Series) -> pd.DataFrame:
    s = serie.fillna("").astype(str).str.lower()
    for src, dst in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n")]:
        s = s.str.replace(src, dst, regex=False)

    out = pd.DataFrame(index=serie.index)
    out["n_dormitorios"] = s.str.extract(r"(\d+)\s*dormitor", expand=False).astype(float)
    out["n_banos"]       = s.str.extract(r"(\d+)\s*bano",     expand=False).astype(float)
    out["m2"]            = s.str.extract(r"(\d+)\s*m[²2]",    expand=False).astype(float)
    for a in AMENITIES:
        out[f"f_{a.replace(' ', '_')}"] = s.str.contains(a, regex=False).astype(int)
    return out


def extra_attrs(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["len_descripcion"] = df["description"].fillna("").str.len()
    out["n_features"]      = df["features"].fillna("").str.count(";")
    out["barrio"]          = df["location_3"].fillna("desconocido").astype(str)
    return out


def main(args):
    print("Cargando datos…")
    engine   = sqlite3.connect(f"{DIR}/entrenamiento.db")
    df_train = pd.read_sql("SELECT * FROM entrenamiento", engine, index_col="id")
    df_test  = pd.read_csv(f"{DIR}/a_predecir.csv", index_col="id")
    print(f"  raw train: {df_train.shape}  |  test: {df_test.shape}")

    # ── 1. Filtrado ──────────────────────────────────────────────────────────
    caba_loc2 = set(df_test["location_2"].dropna().unique())
    caba_loc3 = set(df_test["location_3"].dropna().unique())

    mask_caba = (
        df_train["location_1"].isin(CABA_L1)
        | (
            df_train["location_1"].eq("Buenos Aires")
            & (df_train["location_2"].isin(caba_loc2)
               | df_train["location_3"].isin(caba_loc3))
        )
    )
    mask = (
        df_train["operation_type"].eq("venta")
        & df_train["currency_type"].eq("dolares")
        & mask_caba
        & df_train["property_type"].isin(PROP_OK)
        & df_train["price"].notna()
        & df_train["price"].between(5_000, 3_000_000)
    )
    df_train = df_train.loc[mask].copy()

    for df_ in (df_train, df_test):
        df_["property_type"] = df_["property_type"].replace(
            {"departamentos": "departamento", "casas": "casa"}
        )
    print(f"  tras filtros: {df_train.shape}")

    # ── 2 & 3. Parseo de features + atributos extra ──────────────────────────
    for df_ in (df_train, df_test):
        parsed = pd.concat([parse_features(df_["features"]), extra_attrs(df_)], axis=1)
        for col in parsed.columns:
            df_[col] = parsed[col]

    # ── 4. Winsorización a NaN ───────────────────────────────────────────────
    for df_ in (df_train, df_test):
        df_.loc[~df_["m2"].between(10, 1500),    "m2"]            = np.nan
        df_.loc[~df_["n_dormitorios"].between(0, 15), "n_dormitorios"] = np.nan
        df_.loc[~df_["n_banos"].between(0, 15),       "n_banos"]       = np.nan

    # ── 5. IsolationForest (sólo train) ─────────────────────────────────────
    iso_cols     = ["price", "m2", "n_dormitorios", "n_banos"]
    mask_complete = df_train[iso_cols].notna().all(axis=1)
    X_iso         = df_train.loc[mask_complete, iso_cols].astype(float)

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_iso)
    iso     = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    y_iso   = iso.fit_predict(X_sc)
    outlier_ids = X_iso.index[y_iso == -1]

    df_train = df_train.drop(index=outlier_ids)
    print(f"  outliers IF eliminados: {len(outlier_ids)}  |  train final: {df_train.shape}")

    # ── 6. Indicador de ausencia ─────────────────────────────────────────────
    for df_ in (df_train, df_test):
        df_["m2_was_na"] = df_["m2"].isna().astype(int)

    # ── 7. Imputación (fit sólo en train) ───────────────────────────────────
    imp_med = SimpleImputer(strategy="median")
    imp_med.fit(df_train[NUM_MED])
    df_train[NUM_MED] = imp_med.transform(df_train[NUM_MED])
    df_test[NUM_MED]  = imp_med.transform(df_test[NUM_MED])

    flag_cols = [c for c in df_train.columns if c.startswith("f_")]
    for df_ in (df_train, df_test):
        df_[flag_cols] = df_[flag_cols].fillna(0).astype(int)

    imp_mod = SimpleImputer(strategy="most_frequent")
    imp_mod.fit(df_train[CAT_MODE])
    df_train[CAT_MODE] = imp_mod.transform(df_train[CAT_MODE])
    df_test[CAT_MODE]  = imp_mod.transform(df_test[CAT_MODE])

    # ── 8. Factorización de categóricas ─────────────────────────────────────
    for col in ["barrio", "property_type"]:
        codes, uniques = pd.factorize(df_train[col].astype(str))
        df_train[f"{col}_id"] = codes
        mapping = {v: i for i, v in enumerate(uniques)}
        df_test[f"{col}_id"]  = df_test[col].astype(str).map(mapping).fillna(-1).astype(int)

    # ── 9. Salida ────────────────────────────────────────────────────────────
    out_train = f"{DIR}/train_clean.parquet"
    out_test  = f"{DIR}/test_clean.parquet"

    df_train.to_parquet(out_train)
    df_test.to_parquet(out_test)
    print(f"\n  guardado: {out_train}  ({df_train.shape})")
    print(f"  guardado: {out_test}   ({df_test.shape})")

    if args.csv:
        df_train.to_csv(out_train.replace(".parquet", ".csv"))
        df_test.to_csv(out_test.replace(".parquet", ".csv"))
        print("  CSVs también guardados.")

    print("\nColumnas del train limpio:")
    num_cols = df_train.select_dtypes("number").columns.tolist()
    print(f"  numéricas ({len(num_cols)}): {num_cols}")
    print(f"\nFaltantes residuales (train):")
    missing = df_train.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  ninguno")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="también guardar CSVs")
    main(parser.parse_args())
