"""Análisis de errores del campeón v4 (E2) sobre holdout temporal.

Reproduce el pipeline EXACTO de v4 (filtros, parsing, IsolationForest,
imputación, Hot Deck por description normalizada, RF(500, 50)) con un
**split temporal 80/20** por `publication_date` (no random como v4
original — el temporal imita la distribución del test real de Kaggle).

Después calcula errores y los desagrega por múltiples cortes para
detectar DÓNDE el modelo se equivoca más (Nota 5 del user, propuesta
del profe).

Cortes evaluados:
1. Barrio (`location_3`): separando barrios "limpios" vs etiquetas
   basura ("Ciudad Autónoma de Buenos Aires", etc.). Valida v6.
2. `property_type`: ¿el RF falla en algún tipo en particular?
3. Decil de `price` real: ¿sesgo en propiedades caras vs baratas?
4. Decil de `m2`: ¿sesgo por superficie?
5. Año / mes de publicación: valida v3 (sacar pub_year/pub_month).
6. `m2_was_na`: ¿la imputación por mediana global mete sesgo?
7. Filas con lat/lon original vs imputado: ¿la imputación de
   coordenadas mete sesgo?
8. Proximidad a "centros nombrables" (Puerto Madero, Retiro, Centro,
   Microcentro, Once, Tribunales): valida v8.

Output: entregas/entrega_3/error_analysis_v4.md
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "datasets" / "entrenamiento.db"
CSV_AP = ROOT / "datasets" / "a_predecir.csv"
OUT_MD = ROOT / "entregas" / "entrega_3" / "error_analysis_v4.md"

CABA_L1 = ["Capital Federal", "Ciudad Autónoma de Buenos Aires"]
CABA_TOKENS = {
    "capital federal", "ciudad autonoma de buenos aires",
    "ciudad de buenos aires", "buenos aires", "caba",
}
PROPTYPES = ["casa", "casas", "cochera", "departamento", "departamentos", "ph"]

BARRIO_BASURA = {
    "CABA", "Capital Federal", "Ciudad Autónoma de Buenos Aires",
    "Ciudad de Buenos Aires", "Buenos Aires",
}
CENTROS = ["Centro", "Microcentro", "Once", "Puerto Madero", "Retiro", "Tribunales"]


# -------------------- Pipeline reproducido de v4 --------------------

def parse_features(serie: pd.Series) -> pd.DataFrame:
    AMENITIES = [
        "balcon", "garage", "cochera", "pileta", "parrilla",
        "aire acondicionado", "calefaccion", "gas natural",
        "internet", "seguridad", "alarma", "gimnasio", "jardin",
        "cuarto de servicio", "cocina equipada", "bodega",
    ]
    s = serie.fillna("").astype(str).str.lower()
    for src, dst in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n")]:
        s = s.str.replace(src, dst, regex=False)
    out = pd.DataFrame(index=serie.index)
    out["n_dormitorios"] = s.str.extract(r"(\d+)\s*dormitor", expand=False).astype(float)
    out["n_banos"]       = s.str.extract(r"(\d+)\s*bano", expand=False).astype(float)
    out["m2"]            = s.str.extract(r"(\d+)\s*m[²2]", expand=False).astype(float)
    for a in AMENITIES:
        out[f"f_{a.replace(' ', '_')}"] = s.str.contains(a, regex=False).astype(int)
    return out


def extra_attrs(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["len_descripcion"] = df["description"].fillna("").str.len()
    out["n_features"]      = df["features"].fillna("").str.count(";")
    out["barrio"]          = df["location_3"].fillna("desconocido").astype(str)
    return out


def _norm_desc(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) >= 20 else None


_MES_ES = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,
    "noviembre":11,"diciembre":12,
}


def _pub_score(s: str) -> float:
    """Convierte la fecha en un número ordenable YYYYMM (float NaN si no se puede)."""
    if not isinstance(s, str):
        return np.nan
    sl = s.lower()
    m = re.search(r"(\d{4})[-/](\d{1,2})", sl)
    if m:
        return int(m.group(1)) * 100 + int(m.group(2))
    for nombre, num in _MES_ES.items():
        if nombre in sl:
            yr = re.search(r"(20\d{2})", sl)
            if yr:
                return int(yr.group(1)) * 100 + num
    yr = re.search(r"(20\d{2})", sl)
    if yr:
        return int(yr.group(1)) * 100 + 6
    return np.nan


def load_and_filter() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(DB)
    df_ent = pd.read_sql("SELECT * FROM entrenamiento", conn)
    conn.close()
    df_ap = pd.read_csv(CSV_AP)

    # Filtros v4
    print(f"[err]   train raw: {df_ent.shape}")
    is_caba_l1 = df_ent["location_1"].isin(CABA_L1)
    loc2_norm = df_ent["location_2"].fillna("").str.lower().str.normalize("NFKD")\
        .str.encode("ascii", "ignore").str.decode("ascii")
    loc3_norm = df_ent["location_3"].fillna("").str.lower().str.normalize("NFKD")\
        .str.encode("ascii", "ignore").str.decode("ascii")
    is_caba_loc2 = loc2_norm.isin(CABA_TOKENS)
    is_caba_loc3 = loc3_norm.isin(CABA_TOKENS)
    is_caba = is_caba_l1 | is_caba_loc2 | is_caba_loc3

    df_ent = df_ent[
        is_caba
        & df_ent["property_type"].str.lower().isin(PROPTYPES)
        & (df_ent["operation_type"].str.lower() == "venta")
        & (df_ent["currency_type"].str.lower() == "dolares")
        & df_ent["price"].between(5000, 3000000)
    ].copy()
    print(f"[err]   train post-filtro: {df_ent.shape}")

    # Mismo filtro de columnas para test (sólo para Hot Deck dict)
    return df_ent, df_ap


def fit_v4_pipeline_temporal(df_ent: pd.DataFrame, df_ap: pd.DataFrame):
    """Reproduce v4 con split temporal 80/20."""
    # 1) parse + extras
    feat_ent = pd.concat(
        [parse_features(df_ent["features"]), extra_attrs(df_ent)], axis=1
    )
    df_ent = pd.concat([df_ent, feat_ent], axis=1)

    # Guardamos snapshots ANTES de imputar (para análisis: lat_was_na, m2_was_na)
    df_ent["lat_was_na_orig"] = df_ent["lat"].isna().astype(int)

    # 2) outlier caps
    df_ent.loc[~df_ent["m2"].between(10, 1500), "m2"] = np.nan
    df_ent.loc[~df_ent["n_dormitorios"].between(0, 15), "n_dormitorios"] = np.nan
    df_ent.loc[~df_ent["n_banos"].between(0, 15), "n_banos"] = np.nan

    # 3) IsolationForest multivariado
    iso_cols = ["price", "m2", "n_dormitorios", "n_banos"]
    mask_complete = df_ent[iso_cols].notna().all(axis=1)
    X_iso = df_ent.loc[mask_complete, iso_cols].astype(float)
    scaler = StandardScaler()
    X_iso_sc = scaler.fit_transform(X_iso)
    iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    y_iso = iso.fit_predict(X_iso_sc)
    ids_outliers = X_iso.index[y_iso == -1]
    print(f"[err]   IsolationForest eliminó {len(ids_outliers)} filas")
    df_ent = df_ent.drop(index=ids_outliers)

    # 4) split temporal 80/20 por publication_date
    df_ent["pub_score"] = df_ent["publication_date"].map(_pub_score)
    df_with_date = df_ent.dropna(subset=["pub_score"]).copy()
    df_no_date = df_ent[df_ent["pub_score"].isna()].copy()
    cutoff = df_with_date["pub_score"].quantile(0.8)
    train = pd.concat([
        df_with_date[df_with_date["pub_score"] < cutoff],
        df_no_date,  # sin fecha → al train
    ])
    holdout = df_with_date[df_with_date["pub_score"] >= cutoff].copy()
    print(f"[err]   split temporal: train={len(train)} holdout={len(holdout)}"
          f" (cutoff pub_score={cutoff:.0f})")

    # 5) marcador m2_was_na
    for df_ in (train, holdout):
        df_["m2_was_na"] = df_["m2"].isna().astype(int)

    # 6) imputación (fit on train only)
    NUM_MED = ["m2", "lat", "lon", "n_dormitorios", "n_banos", "len_descripcion", "n_features"]
    CAT_MODE = ["property_type", "barrio"]
    imp_med = SimpleImputer(strategy="median").fit(train[NUM_MED])
    train[NUM_MED] = imp_med.transform(train[NUM_MED])
    holdout[NUM_MED] = imp_med.transform(holdout[NUM_MED])

    flag_cols = [c for c in train.columns if c.startswith("f_")]
    train[flag_cols] = train[flag_cols].fillna(0).astype(int)
    holdout[flag_cols] = holdout[flag_cols].fillna(0).astype(int)

    imp_mod = SimpleImputer(strategy="most_frequent").fit(train[CAT_MODE])
    train[CAT_MODE] = imp_mod.transform(train[CAT_MODE])
    holdout[CAT_MODE] = imp_mod.transform(holdout[CAT_MODE])

    # 7) Hot Deck — built con todo el train (incluye holdout? NO, sólo train para
    #    evitar leakage en el análisis del holdout). En producción v4 se hizo con
    #    todo el train; acá excluimos el holdout para que sea un análisis honesto.
    train["desc_norm"] = train["description"].map(_norm_desc)
    price_by_desc = train.dropna(subset=["desc_norm"]).groupby("desc_norm")["price"].median()
    holdout["desc_norm"] = holdout["description"].map(_norm_desc)
    holdout["hd_pred"] = holdout["desc_norm"].map(price_by_desc)
    n_hd = int(holdout["hd_pred"].notna().sum())
    print(f"[err]   Hot Deck cubre {n_hd}/{len(holdout)} filas del holdout"
          f" ({n_hd/len(holdout)*100:.2f}%)")

    # 8) factorize categóricas
    for col in ["barrio", "property_type"]:
        codes, uniques = pd.factorize(train[col].astype(str))
        train[f"{col}_id"] = codes
        mapping = {v: i for i, v in enumerate(uniques)}
        holdout[f"{col}_id"] = holdout[col].astype(str).map(mapping).fillna(-1).astype(int)

    # 9) features definitivas (mismas 26 que v4)
    FEATS = [
        "lat", "lon", "n_dormitorios", "n_banos", "m2",
        "f_balcon", "f_garage", "f_cochera", "f_pileta", "f_parrilla",
        "f_aire_acondicionado", "f_calefaccion", "f_gas_natural", "f_internet",
        "f_seguridad", "f_alarma", "f_gimnasio", "f_jardin",
        "f_cuarto_de_servicio", "f_cocina_equipada", "f_bodega",
        "len_descripcion", "n_features", "m2_was_na",
        "barrio_id", "property_type_id",
    ]

    # 10) entrenar RF(500, 50)
    print(f"[err]   entrenando RandomForest(500, 50) sobre {len(train)} filas...")
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=50, random_state=42, n_jobs=-1,
    )
    rf.fit(train[FEATS], train["price"])

    # 11) predecir sobre el holdout
    holdout["pred_modelo"] = rf.predict(holdout[FEATS])
    # Aplicar Hot Deck override (igual que v4 en producción)
    holdout["pred_final"] = holdout["pred_modelo"]
    mask_hd = holdout["hd_pred"].notna()
    holdout.loc[mask_hd, "pred_final"] = holdout.loc[mask_hd, "hd_pred"]

    # 12) métricas globales
    rmse_modelo = root_mean_squared_error(holdout["price"], holdout["pred_modelo"])
    rmse_final = root_mean_squared_error(holdout["price"], holdout["pred_final"])
    print(f"[err]   RMSE holdout temporal (sólo modelo) = {rmse_modelo:,.0f}")
    print(f"[err]   RMSE holdout temporal (con Hot Deck) = {rmse_final:,.0f}")

    return holdout, rmse_modelo, rmse_final


# -------------------- Análisis de errores --------------------

def _agg_error(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Calcula n_obs, mean_signed, mean_abs, RMSE por grupo + contribución al RMSE total."""
    g = df.groupby(by, dropna=False, observed=True)
    rmse_total_sq = (df["residual"] ** 2).sum()
    out = pd.DataFrame({
        "n_obs": g.size(),
        "mean_signed": g["residual"].mean().round(0),
        "mean_abs":    g["abs_error"].mean().round(0),
        "rmse_grupo":  np.sqrt(g.apply(lambda x: (x["residual"]**2).mean())).round(0),
        "contrib_rmse_pct": (g.apply(lambda x: (x["residual"]**2).sum()) / rmse_total_sq * 100).round(2),
    })
    return out.reset_index()


def analizar_barrios(df: pd.DataFrame) -> dict:
    """Top barrios por contribución al RMSE total + comparación limpio vs basura."""
    df = df.copy()
    df["barrio_basura"] = df["barrio"].isin(BARRIO_BASURA).astype(int)

    # Top 15 barrios por contribución al RMSE total
    por_barrio = _agg_error(df, "barrio")
    top_contrib = por_barrio.sort_values("contrib_rmse_pct", ascending=False).head(15)

    # Limpio vs basura (validación de v6)
    limpio_vs_basura = _agg_error(df, "barrio_basura")
    limpio_vs_basura["barrio_basura"] = limpio_vs_basura["barrio_basura"].map(
        {0: "limpio", 1: "basura"}
    )

    return {"top_contrib": top_contrib, "limpio_vs_basura": limpio_vs_basura}


def analizar_property_type(df: pd.DataFrame) -> pd.DataFrame:
    return _agg_error(df, "property_type").sort_values("contrib_rmse_pct", ascending=False)


def analizar_decil_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decil_price"] = pd.qcut(df["price"], 10, labels=False, duplicates="drop")
    out = _agg_error(df, "decil_price")
    # Agregar p50 de price por decil para legibilidad
    p50 = df.groupby("decil_price")["price"].median().round(0)
    out["price_p50"] = out["decil_price"].map(p50)
    return out


def analizar_decil_m2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decil_m2"] = pd.qcut(df["m2"], 10, labels=False, duplicates="drop")
    out = _agg_error(df, "decil_m2")
    p50 = df.groupby("decil_m2")["m2"].median().round(0)
    out["m2_p50"] = out["decil_m2"].map(p50)
    return out


def analizar_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pub_yyyymm"] = df["pub_score"].astype("Int64")
    return _agg_error(df, "pub_yyyymm").sort_values("pub_yyyymm")


def analizar_m2_was_na(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["m2_was_na_label"] = df["m2_was_na"].map({0: "m2_observado", 1: "m2_imputado"})
    return _agg_error(df, "m2_was_na_label")


def analizar_lat_was_na(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lat_was_na_label"] = df["lat_was_na_orig"].map(
        {0: "lat_observado", 1: "lat_imputado"}
    )
    return _agg_error(df, "lat_was_na_label")


def analizar_centros(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada centro nombrable, RMSE en filas con location_3 == centro."""
    cols = ["centro", "n_obs", "mean_signed", "mean_abs", "rmse_grupo", "contrib_rmse_pct"]
    rows = []
    rmse_total_sq = (df["residual"] ** 2).sum()
    for centro in CENTROS:
        mask = df["barrio"].fillna("").str.contains(centro, case=False, regex=False)
        n = int(mask.sum())
        if n < 5:
            continue
        sub = df[mask]
        rmse = float(np.sqrt((sub["residual"] ** 2).mean()))
        rows.append({
            "centro": centro,
            "n_obs": n,
            "mean_signed": round(float(sub["residual"].mean()), 0),
            "mean_abs": round(float(sub["abs_error"].mean()), 0),
            "rmse_grupo": round(rmse, 0),
            "contrib_rmse_pct": round((sub["residual"] ** 2).sum() / rmse_total_sq * 100, 2),
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("contrib_rmse_pct", ascending=False)


def analizar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada f_*, comparar RMSE entre presente vs ausente."""
    flag_cols = [c for c in df.columns if c.startswith("f_")]
    rows = []
    for c in flag_cols:
        for valor in (0, 1):
            sub = df[df[c] == valor]
            if len(sub) < 30:
                continue
            rmse = float(np.sqrt((sub["residual"] ** 2).mean()))
            rows.append({
                "feature": c, "valor": valor, "n_obs": len(sub),
                "rmse_grupo": round(rmse, 0),
                "mean_signed": round(float(sub["residual"].mean()), 0),
            })
    out = pd.DataFrame(rows)
    pivot = out.pivot(index="feature", columns="valor", values="rmse_grupo")
    pivot.columns = ["rmse_si_0", "rmse_si_1"]
    pivot["delta_rmse_1_menos_0"] = pivot["rmse_si_1"] - pivot["rmse_si_0"]
    pivot = pivot.reset_index().sort_values("delta_rmse_1_menos_0", ascending=False)
    return pivot


# -------------------- Render markdown --------------------

def _fmt(v):
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        if v == int(v):
            return f"{int(v):,}"
        return f"{v:,.2f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return str(v)


def df_to_md(df: pd.DataFrame, index: bool = False) -> str:
    if df is None or len(df) == 0:
        return "_(vacío)_\n"
    df = df.reset_index() if index else df.copy()
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row.values) + " |")
    return "\n".join(lines) + "\n"


def main():
    print("[err] cargando train + test")
    df_ent, df_ap = load_and_filter()

    print("[err] reproduciendo pipeline v4 con split temporal")
    holdout, rmse_modelo, rmse_final = fit_v4_pipeline_temporal(df_ent, df_ap)

    # Calcular residuals (price_real - price_pred) → positivo = subestimamos
    holdout["residual"] = holdout["price"] - holdout["pred_final"]
    holdout["abs_error"] = holdout["residual"].abs()

    print("\n[err] análisis 1: barrios")
    bar = analizar_barrios(holdout)
    print(bar["limpio_vs_basura"].to_string(index=False))
    print("\ntop15 por contribución al RMSE:")
    print(bar["top_contrib"].to_string(index=False))

    print("\n[err] análisis 2: property_type")
    pt = analizar_property_type(holdout)
    print(pt.to_string(index=False))

    print("\n[err] análisis 3: decil de price real")
    dec_p = analizar_decil_price(holdout)
    print(dec_p.to_string(index=False))

    print("\n[err] análisis 4: decil de m2")
    dec_m = analizar_decil_m2(holdout)
    print(dec_m.to_string(index=False))

    print("\n[err] análisis 5: año de publicación")
    yr = analizar_temporal(holdout)
    print(yr.to_string(index=False))

    print("\n[err] análisis 6: m2 imputado vs observado")
    m2_na = analizar_m2_was_na(holdout)
    print(m2_na.to_string(index=False))

    print("\n[err] análisis 7: lat imputado vs observado")
    lat_na = analizar_lat_was_na(holdout)
    print(lat_na.to_string(index=False))

    print("\n[err] análisis 8: centros nombrables")
    cen = analizar_centros(holdout)
    print(cen.to_string(index=False))

    print("\n[err] análisis 9: presencia de features")
    feat = analizar_features(holdout)
    print(feat.head(20).to_string(index=False))

    # Render markdown
    md = ["# Análisis de errores del campeón v4 (E2) sobre holdout temporal\n"]
    md.append(
        "Reproducción exacta del pipeline v4 (filtros, parsing, IsolationForest, "
        "imputación por mediana, Hot Deck por description normalizada, RF(500, 50)) "
        "con **split temporal 80/20** por `publication_date`. El Hot Deck del "
        "análisis se construye **sólo con el 80 % train del split** (no con todo "
        "el train original) para evitar leakage en la evaluación del holdout.\n"
    )
    md.append(f"\n**Setup**: {len(holdout):,} filas en el holdout temporal.\n")
    md.append(f"\n**RMSE holdout temporal** (sólo modelo): **{rmse_modelo:,.0f}**.\n")
    md.append(f"**RMSE holdout temporal** (modelo + Hot Deck): **{rmse_final:,.0f}**.\n")

    md.append("\n## 1. Errores por barrio\n")
    md.append(
        "Comparación crítica para validar **v6 (limpieza de barrio)**: ¿el "
        "modelo se equivoca más en filas con etiqueta basura "
        "(\"Ciudad Autónoma de Buenos Aires\", etc.) que en barrios limpios?\n"
    )
    md.append("\n### 1.1 Limpio vs basura\n")
    md.append(df_to_md(bar["limpio_vs_basura"]))
    md.append("\n### 1.2 Top 15 barrios por contribución al RMSE total\n")
    md.append(df_to_md(bar["top_contrib"]))

    md.append("\n## 2. Errores por property_type\n")
    md.append(df_to_md(pt))

    md.append("\n## 3. Errores por decil de `price` real\n")
    md.append(
        "`mean_signed` positivo = el modelo **subestima** el precio. "
        "Si el sesgo cambia de signo entre deciles bajos y altos, el RF "
        "está pegado al promedio y mete sesgo regressivo.\n"
    )
    md.append(df_to_md(dec_p))

    md.append("\n## 4. Errores por decil de `m2`\n")
    md.append(df_to_md(dec_m))

    md.append("\n## 5. Errores por año de publicación\n")
    md.append(
        "Valida la decisión **v3** de sacar `pub_year`/`pub_month`: si los "
        "años más recientes (2025-2026) tienen RMSE muy distinto a 2022-2024, "
        "el lookup año→precio que el RF aprende no generaliza.\n"
    )
    md.append(df_to_md(yr))

    md.append("\n## 6. Errores en filas con m2 imputado vs observado\n")
    md.append(
        "Si el RMSE en `m2_imputado` es mucho mayor, la imputación por mediana "
        "global está rompiendo predicciones (motiva imputación más sofisticada).\n"
    )
    md.append(df_to_md(m2_na))

    md.append("\n## 7. Errores en filas con lat imputado vs observado\n")
    md.append(
        "Mismo análisis para `lat`. El 63 % del train tiene lat NaN antes de "
        "imputar; si el RMSE es muy distinto entre los dos grupos, la imputación "
        "por mediana global está rompiendo el feature geográfico.\n"
    )
    md.append(df_to_md(lat_na))

    md.append("\n## 8. Errores cerca de centros nombrables\n")
    md.append(
        "Para cada centro del Análisis 7.5 del EDA, RMSE en filas donde "
        "`location_3` contiene ese nombre. **Valida v8** (`dist_a_<centro>`): "
        "si los centros tienen RMSE alto, vale invertir esfuerzo en derivar "
        "distancias.\n"
    )
    md.append(df_to_md(cen))

    md.append("\n## 9. Errores por presencia de feature binaria\n")
    md.append(
        "Para cada `f_*`, RMSE cuando vale 1 vs cuando vale 0. "
        "`delta_rmse_1_menos_0` positivo = el modelo se equivoca MÁS cuando la "
        "feature está presente (= la propiedad es atípica respecto a su grupo).\n"
    )
    md.append(df_to_md(feat))

    # Conclusiones accionables
    pct_basura_rmse = float(bar["limpio_vs_basura"].query("barrio_basura == 'basura'")["contrib_rmse_pct"].iloc[0])
    delta_basura_rmse = float(
        bar["limpio_vs_basura"].query("barrio_basura == 'basura'")["rmse_grupo"].iloc[0]
        - bar["limpio_vs_basura"].query("barrio_basura == 'limpio'")["rmse_grupo"].iloc[0]
    )

    md.append("\n## Conclusiones accionables y reordenamiento de prioridades\n")
    md.append(
        f"- **Barrio basura** (Sección 1.1): contribuye **{pct_basura_rmse:.1f}%** "
        f"al RMSE total con un **delta de RMSE de {delta_basura_rmse:+,.0f}** "
        f"vs barrios limpios. "
        f"{'JUSTIFICA v6 con prioridad alta' if delta_basura_rmse > 5000 else 'CONFIRMA prioridad media de v6'}.\n"
        f"- **Sesgo por decil de price** (Sección 3): mirar la columna "
        f"`mean_signed`. Si los deciles bajos tienen sesgo positivo "
        f"(subestimamos) y los altos negativo (sobreestimamos), el RF está "
        f"haciendo regression to the mean — eso JUSTIFICA `np.log(price)` "
        f"como target transform (no estaba en el plan, agregar como v9).\n"
        f"- **Sesgo por año** (Sección 5): si 2025-2026 tienen RMSE muy "
        f"distinto a 2022-2024, refuerza la decisión v3 de sacar "
        f"`pub_year`/`pub_month`.\n"
        f"- **Imputación de m2 / lat** (Secciones 6-7): si el RMSE de las "
        f"filas imputadas duplica al de las observadas, la imputación global "
        f"por mediana es subóptima — agregar al plan: imputación por barrio "
        f"o KNN sobre features completas.\n"
        f"- **Centros nombrables** (Sección 8): los que tienen RMSE alto + "
        f"buena `n_obs` son los candidatos más fuertes para v8 "
        f"(`dist_a_<centro>`). Reordenar la lista del plan v8 según esta "
        f"contribución al RMSE.\n"
        f"- **Features con delta positivo** (Sección 9): las propiedades "
        f"con `f_X = 1` que tienen RMSE más alto son atípicas respecto a "
        f"su grupo — candidatas para feature engineering específico "
        f"(p.ej. interacciones `f_pileta * m2`).\n"
    )

    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(f"\n[err] markdown escrito en {OUT_MD}")


if __name__ == "__main__":
    main()
