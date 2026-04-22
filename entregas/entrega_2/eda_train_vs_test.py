"""
EDA comparativo TRAIN vs TEST para la Entrega 2.
Su único objetivo es justificar (o desafiar) los filtros que mantenemos
desde la Entrega 1, en particular:

    * ¿el filtro `location_1 ∈ {Capital Federal, Ciudad Autónoma de Buenos Aires}`
      cubre todo el universo de `a_predecir`?
    * ¿hay filas en train con `location_1 == "Buenos Aires"` que en realidad
      son CABA y nos estamos perdiendo?
    * ¿el rango de price [5K, 3M] cubre el rango razonable de las predicciones?
    * ¿qué tipos de propiedad / barrios aparecen en test?
    * ¿hay diferencias notorias en m², dormitorios, baños entre train y test?
"""
import io
import sqlite3
import sys
import textwrap

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
DIR = "datasets"

# ------------------------------------------------------------------ #
# 1. Carga
# ------------------------------------------------------------------ #
con = sqlite3.connect(f"{DIR}/entrenamiento.db")
train = pd.read_sql("SELECT * FROM entrenamiento", con)
test = pd.read_csv(f"{DIR}/a_predecir.csv")

print(f"train: {train.shape}  |  test: {test.shape}")
print(f"train con price: {train['price'].notna().sum()}")

# ------------------------------------------------------------------ #
# 2. Operación, moneda, tipos de propiedad en test (ya sabemos pero
#    lo dejamos chequeado en duro)
# ------------------------------------------------------------------ #
print("\n--- TEST: composición ---")
for col in ("operation_type", "currency_type", "property_type"):
    print(f"\n{col}:")
    print(test[col].value_counts(dropna=False))

# ------------------------------------------------------------------ #
# 3. Geografía: ¿qué cubrir?
# ------------------------------------------------------------------ #
print("\n--- TEST: location_1 ---")
print(test["location_1"].value_counts(dropna=False))
print("\n--- TEST: location_2 (top 20) ---")
print(test["location_2"].value_counts(dropna=False).head(20))
print("\n--- TEST: location_3 (top 20) ---")
print(test["location_3"].value_counts(dropna=False).head(20))

# El sospechoso: train con location_1 == "Buenos Aires" — ¿son CABA o Provincia?
print("\n--- TRAIN: location_1 == 'Buenos Aires' ¿qué location_2 tienen? ---")
ba = train.loc[train["location_1"] == "Buenos Aires"]
print(f"  total: {len(ba)}")
print(ba["location_2"].value_counts(dropna=False).head(15))

# Cruce con barrios CABA conocidos a partir de test
caba_barrios = set(test["location_3"].dropna().unique())
caba_locations2 = set(test["location_2"].dropna().unique())
print(f"\n  barrios CABA distintos en test: {len(caba_barrios)}")
print(f"  ejemplos: {list(caba_barrios)[:10]}")

# ¿Cuántas filas de location_1='Buenos Aires' tienen un barrio CABA?
ba_caba = ba.loc[
    ba["location_3"].isin(caba_barrios)
    | ba["location_2"].isin(caba_locations2)
]
print(f"  filas con location_1='Buenos Aires' que parecen CABA "
      f"(por barrio o location_2): {len(ba_caba)}")

# ------------------------------------------------------------------ #
# 4. Filtro actual vs universo objetivo (test)
# ------------------------------------------------------------------ #
CABA_L1_ACTUAL = {"Capital Federal", "Ciudad Autónoma de Buenos Aires"}
CABA_L1_AMPLIADO = CABA_L1_ACTUAL | {"Buenos Aires"}  # con la sospecha

PROP_OK = {"departamento", "departamentos", "casa", "casas", "ph", "cochera"}

mask_actual = (
    train["operation_type"].eq("venta")
    & train["currency_type"].eq("dolares")
    & train["location_1"].isin(CABA_L1_ACTUAL)
    & train["property_type"].isin(PROP_OK)
    & train["price"].notna()
    & train["price"].between(5_000, 3_000_000)
)

mask_ampliado_caba = (
    train["operation_type"].eq("venta")
    & train["currency_type"].eq("dolares")
    & (
        train["location_1"].isin(CABA_L1_ACTUAL)
        | (
            train["location_1"].eq("Buenos Aires")
            & (train["location_2"].isin(caba_locations2)
               | train["location_3"].isin(caba_barrios))
        )
    )
    & train["property_type"].isin(PROP_OK)
    & train["price"].notna()
    & train["price"].between(5_000, 3_000_000)
)

print("\n--- COMPARACIÓN DE FILTROS ---")
print(f"Filtro actual (location_1 estricto):       {mask_actual.sum():>7} filas")
print(f"Filtro ampliado (también 'Buenos Aires'):  {mask_ampliado_caba.sum():>7} filas")
print(f"Δ filas extra capturadas:                  "
      f"{mask_ampliado_caba.sum() - mask_actual.sum():>7}")

# ------------------------------------------------------------------ #
# 5. Rango de precios: ¿el corte [5K, 3M] tiene sentido?
# ------------------------------------------------------------------ #
print("\n--- price (USD) post-filtro actual ---")
print(train.loc[mask_actual, "price"].describe(
    percentiles=[.001, .01, .05, .5, .95, .99, .999]
))

# ¿Cuántas filas perdemos por el corte de precio?
mask_pre_price = (
    train["operation_type"].eq("venta")
    & train["currency_type"].eq("dolares")
    & train["location_1"].isin(CABA_L1_ACTUAL)
    & train["property_type"].isin(PROP_OK)
    & train["price"].notna()
)
sin_corte = train.loc[mask_pre_price, "price"]
print(f"\n  filas sin corte: {len(sin_corte)}")
print(f"  filas con corte [5K, 3M]: {mask_actual.sum()}  "
      f"({mask_actual.sum()/len(sin_corte)*100:.1f}%)")
print(f"  perdidas por < 5K: {(sin_corte < 5_000).sum()}")
print(f"  perdidas por > 3M: {(sin_corte > 3_000_000).sum()}")

# ------------------------------------------------------------------ #
# 6. Comparación de variables parseables (m2, dorm, banos) train vs test
# ------------------------------------------------------------------ #
def parse_num(s, pat):
    return s.fillna("").astype(str).str.lower().str.extract(pat, expand=False).astype(float)

train_filt = train.loc[mask_actual]
for label, df in [("TRAIN (filtrado)", train_filt), ("TEST", test)]:
    m2 = parse_num(df["features"], r"(\d+)\s*m")
    dorm = parse_num(df["features"], r"(\d+)\s*dormitor")
    banos = parse_num(df["features"], r"(\d+)\s*ba(?:ñ|n)o")
    print(f"\n--- {label}: m2 (validos en [10, 1500]) ---")
    print(m2.where(m2.between(10, 1500)).describe(percentiles=[.05, .5, .95]))
    print(f"--- {label}: dormitorios (validos [0, 15]) ---")
    print(dorm.where(dorm.between(0, 15)).describe(percentiles=[.05, .5, .95]))
    print(f"--- {label}: baños (validos [0, 15]) ---")
    print(banos.where(banos.between(0, 15)).describe(percentiles=[.05, .5, .95]))

# ------------------------------------------------------------------ #
# 7. % de filas de TEST que caerían fuera del filtro (algunas categorías)
# ------------------------------------------------------------------ #
print("\n--- TEST: cobertura del filtro actual ---")
print(f"  property_type cubierto:  "
      f"{test['property_type'].isin(PROP_OK).mean()*100:.1f}%")
print(f"  operation_type='venta':  "
      f"{test['operation_type'].eq('venta').mean()*100:.1f}%")
print(f"  currency='dolares':      "
      f"{test['currency_type'].eq('dolares').mean()*100:.1f}%")
print(f"  location_1 ∈ CABA:       "
      f"{test['location_1'].isin(CABA_L1_ACTUAL).mean()*100:.1f}%")
