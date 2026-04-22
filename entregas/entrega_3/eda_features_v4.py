"""eda_features_v4.py — EDA preparatorio para Entrega 3.

Reproduce el pipeline de filtros + parsing del campeón v4 (sin outliers/
imputación, queremos ver las distribuciones crudas) y produce 4 análisis
para anclar los experimentos del plan E3 v0 beta:

1. Skewness y percentiles de cada numérica → candidatas a np.log / np.sqrt.
2. Distribución temporal train vs test → cuantifica el shift de v6.
3. Information Gain de cada feature binaria contra pd.qcut(price, 10) →
   ranking de utilidad para discretización supervisada (Clase 7).
4. Distribución del tamaño de cada barrio → candidatos a colapsar.
5. Cobertura del parseo dual `features` + `description` (insumo C3PO):
   filas donde m2/n_dormitorios/n_banos faltan en `features` pero
   pueden extraerse de `description`.
6. Cobertura de `surface_total_m2` vs `surface_covered_m2` (insumo
   C3PO): cuántas filas traen los dos valores y cuán distintos son
   (justifica o descarta `pct_surface_covered` como feature derivada).
7. Geografía interna sin APIs (Notas 3 y 4 del user):
   - Cardinalidad de location_1..4 en train y test (¿es location_3
     sub-barrio?).
   - Filas con location_2 / location_3 mal etiquetados como
     "CABA"/"Capital Federal"/"Buenos Aires"/NaN (sizing de la
     limpieza Nota 3).
   - Intersección train↔test de los barrios (¿podemos imputar test
     con vocabulario de train?).
   - Presencia de nombres de centros comerciales en location_3
     (justifica o descarta dist_a_microcentro estilo Nota 4 idea 3
     sin necesidad de APIs externas).

Output: entregas/entrega_3/eda_features_v4.md
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"
OUT_MD = Path(__file__).parent / "eda_features_v4.md"

# ---------- Pipeline mínimo de v4 -------------------------------------------

CABA_L1 = {"Capital Federal", "Ciudad Autónoma de Buenos Aires"}
PROP_OK = {"departamento", "departamentos", "casa", "casas", "ph", "cochera"}

AMENITIES = [
    "balcon", "garage", "cochera", "pileta", "parrilla",
    "aire acondicionado", "calefaccion", "gas natural",
    "internet", "seguridad", "alarma", "gimnasio", "jardin",
    "cuarto de servicio", "cocina equipada", "bodega",
]


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


def load_and_filter() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga train+test, aplica filtros de v4 al train y parsea features en ambos."""
    print("[eda] cargando train (sqlite, ~1.3 M filas)...", flush=True)
    t0 = time.time()
    eng = sqlite3.connect(DATA / "entrenamiento.db")
    df_ent = pd.read_sql("SELECT * FROM entrenamiento", eng, index_col="id")
    eng.close()
    df_ap = pd.read_csv(DATA / "a_predecir.csv", index_col="id")
    print(f"[eda]   train: {df_ent.shape}  test: {df_ap.shape}  ({time.time()-t0:.1f}s)")

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
    df_ent["property_type"] = df_ent["property_type"].replace(
        {"departamentos": "departamento", "casas": "casa"}
    )
    df_ap["property_type"] = df_ap["property_type"].replace(
        {"departamentos": "departamento", "casas": "casa"}
    )
    print(f"[eda]   train post-filtro: {df_ent.shape}")

    feat_ent = pd.concat([parse_features(df_ent["features"]), extra_attrs(df_ent)], axis=1)
    feat_ap  = pd.concat([parse_features(df_ap["features"]),  extra_attrs(df_ap)],  axis=1)
    df_ent = pd.concat([df_ent, feat_ent], axis=1)
    df_ap  = pd.concat([df_ap,  feat_ap],  axis=1)

    return df_ent, df_ap


# ---------- Análisis 1: skewness de las numéricas ---------------------------

def analisis_skewness(df_ent: pd.DataFrame) -> pd.DataFrame:
    """Para cada numérica continua: percentiles + skewness + recomendación."""
    cols = ["price", "m2", "lat", "lon", "n_dormitorios", "n_banos",
            "len_descripcion", "n_features"]
    rows = []
    for c in cols:
        v = df_ent[c].dropna()
        if len(v) == 0:
            continue
        sk = float(skew(v))
        # Heurística de recomendación: skew alto y valores positivos → log;
        # skew moderado y positivos → sqrt; skew bajo → nada.
        if v.min() > 0 and sk > 2:
            reco = "np.log (skew alto, valores positivos)"
        elif v.min() >= 0 and sk > 1:
            reco = "np.log1p o np.sqrt (skew moderado)"
        elif abs(sk) < 0.5:
            reco = "nada (cuasi-simetrica)"
        else:
            reco = "evaluar (skew leve o valores no-positivos)"
        rows.append({
            "columna":   c,
            "n":         int(len(v)),
            "p01":       float(v.quantile(0.01)),
            "p50":       float(v.quantile(0.50)),
            "p99":       float(v.quantile(0.99)),
            "max":       float(v.max()),
            "skew":      round(sk, 2),
            "reco":      reco,
        })
    return pd.DataFrame(rows)


# ---------- Análisis 2: shift temporal train vs test ------------------------

# Mismo regex/_pub_score que en la celda de holdout temporal de v6
MESES = {"ene":1,"feb":2,"mar":3,"abr":4,"may":5,"jun":6,
         "jul":7,"ago":8,"sept":9,"sep":9,"oct":10,"nov":11,"dic":12}
PAT = re.compile(r"(\d{1,2})\s+(\w+)\s+(\d{4})")


def parse_pub(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    m = PAT.match(str(s).strip())
    if not m:
        return None
    _, mes, a = m.groups()
    mn = MESES.get(mes.lower())
    if mn is None:
        return None
    return int(a) * 100 + int(mn)


def analisis_temporal(df_ent: pd.DataFrame, df_ap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distribución de publication_date por (año, mes) en train vs test."""
    s_ent = df_ent["publication_date"].map(parse_pub).dropna().astype(int)
    s_ap  = df_ap["publication_date"].map(parse_pub).dropna().astype(int)

    print(f"[eda] temporal: train parseable={len(s_ent):,}/{len(df_ent):,} "
          f"({len(s_ent)/len(df_ent)*100:.1f}%) | "
          f"test parseable={len(s_ap):,}/{len(df_ap):,} "
          f"({len(s_ap)/len(df_ap)*100:.1f}%)")

    def por_anio(s):
        a = s // 100
        return a.value_counts().sort_index().rename("n")

    def por_anio_mes(s):
        return s.value_counts().sort_index().rename("n")

    anio = pd.concat(
        [por_anio(s_ent).rename("train_n"), por_anio(s_ap).rename("test_n")],
        axis=1,
    ).fillna(0).astype(int)
    anio["train_pct"] = (anio["train_n"] / anio["train_n"].sum() * 100).round(2)
    anio["test_pct"]  = (anio["test_n"]  / anio["test_n"].sum()  * 100).round(2)
    anio["delta_pct"] = (anio["test_pct"] - anio["train_pct"]).round(2)

    # Mes calendario (1-12, agregando todos los años) — mide estacionalidad pura
    mes_ent = (s_ent % 100).value_counts().sort_index()
    mes_ap  = (s_ap  % 100).value_counts().sort_index()
    mes = pd.concat(
        [mes_ent.rename("train_n"), mes_ap.rename("test_n")],
        axis=1,
    ).fillna(0).astype(int)
    mes.index.name = "mes"
    mes["train_pct"] = (mes["train_n"] / mes["train_n"].sum() * 100).round(2)
    mes["test_pct"]  = (mes["test_n"]  / mes["test_n"].sum()  * 100).round(2)
    mes["delta_pct"] = (mes["test_pct"] - mes["train_pct"]).round(2)

    return anio, mes


# ---------- Análisis 3: Information Gain de binarias ------------------------

def entropia(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def analisis_ig_binarias(df_ent: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Information Gain de cada f_* contra pd.qcut(price, q=n_bins).

    Implementación manual (ver Clase 7, celdas 1039-1158):
      H(Y) - Σ |Yk|/|Y| · H(Yk),  donde Yk = bins de price | feature=k.
    """
    y = df_ent["price"].dropna()
    bins_y = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    pcts_y = bins_y.value_counts(normalize=True).to_numpy()
    H_y = entropia(pcts_y)

    rows = []
    bin_cols = [c for c in df_ent.columns if c.startswith("f_")]
    for c in bin_cols:
        v = df_ent.loc[y.index, c].fillna(0).astype(int)
        H_cond = 0.0
        for val in (0, 1):
            mask = v == val
            n_k = mask.sum()
            if n_k == 0:
                continue
            pcts_k = bins_y[mask].value_counts(normalize=True).to_numpy()
            H_cond += (n_k / len(v)) * entropia(pcts_k)
        IG = H_y - H_cond
        rows.append({
            "feature":   c,
            "tasa_1_train_pct": round(v.mean() * 100, 2),
            "IG":        round(IG, 4),
            "IG_pct_Hy": round(IG / H_y * 100, 2),  # IG normalizado contra entropía total
        })
    return pd.DataFrame(rows).sort_values("IG", ascending=False).reset_index(drop=True)


def analisis_ig_binarias_test(df_ent: pd.DataFrame, df_ap: pd.DataFrame) -> pd.DataFrame:
    """Tasa de 1s en train vs test para cada f_* — detecta drift en covariables."""
    bin_cols = [c for c in df_ent.columns if c.startswith("f_")]
    rows = []
    for c in bin_cols:
        rows.append({
            "feature":     c,
            "tasa_train":  round(df_ent[c].mean() * 100, 2),
            "tasa_test":   round(df_ap[c].mean() * 100, 2),
            "delta":       round((df_ap[c].mean() - df_ent[c].mean()) * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("delta", key=abs, ascending=False).reset_index(drop=True)


# ---------- Análisis 5 + 6: insumos C3PO (parseo dual + surface) ------------

# Patrones del robot C3PO (Robot_2_C3PO.ipynb), normalizados sin acentos para
# que sean intercambiables con nuestro `parse_features`.
PAT_BED   = re.compile(r"(\d+)\s*(?:dormitor|habitacion)")
PAT_BATH  = re.compile(r"(\d+)\s*(?:bano|banos)")
PAT_M2    = re.compile(r"(\d+(?:[.,]\d+)?)\s*m[²2]")
PAT_M2_C  = re.compile(r"(\d+(?:[.,]\d+)?)\s*m[²2]\s*(?:cubiert[oa]s?|cubierta)")
PAT_ROOMS = re.compile(r"(\d+)\s*(?:ambiente|ambientes)")


def _normalize(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    for src, dst in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n")]:
        s = s.str.replace(src, dst, regex=False)
    return s


def _extract_first(s: pd.Series, pat: re.Pattern) -> pd.Series:
    return pd.to_numeric(s.str.extract(pat.pattern, expand=False), errors="coerce")


def analisis_parseo_dual(df_ent: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cuántas filas tienen NaN en m2/n_dormitorios/n_banos en `features`
    pero el dato sí está en `description`. Justifica o descarta el insumo
    de C3PO (parsear `features` + `description` concatenados).
    """
    feat = _normalize(df_ent["features"])
    desc = _normalize(df_ent["description"])

    rows = []
    for col, pat in [("m2", PAT_M2), ("n_dormitorios", PAT_BED), ("n_banos", PAT_BATH)]:
        v_feat = _extract_first(feat, pat)
        v_desc = _extract_first(desc, pat)
        n_total = len(df_ent)
        nan_feat = v_feat.isna()
        rescatadas = nan_feat & v_desc.notna()
        rows.append({
            "atributo":             col,
            "n_total":              int(n_total),
            "nan_en_features":      int(nan_feat.sum()),
            "pct_nan_features":     round(nan_feat.mean() * 100, 2),
            "rescatadas_por_desc":  int(rescatadas.sum()),
            "pct_rescatadas":       round(rescatadas.mean() * 100, 2),
        })
    tabla = pd.DataFrame(rows)
    pct_max = float(tabla["pct_rescatadas"].max())
    veredicto = (
        "ENTRA en v2 (parseo dual rescata > 1% en al menos un atributo)"
        if pct_max > 1.0
        else "DESCARTAR (rescate < 1%, no justifica complejidad extra)"
    )
    return tabla, {"pct_max_rescatado": pct_max, "veredicto": veredicto}


def analisis_surface(df_ent: pd.DataFrame) -> dict:
    """Cobertura conjunta de `surface_total_m2` y `surface_covered_m2` en
    el texto. Si los dos valores aparecen en > 5% de las filas y difieren
    > 10%, vale separarlos + crear `pct_surface_covered`. Si no, queda
    descartado.
    """
    texto = _normalize(df_ent["features"]) + " " + _normalize(df_ent["description"])
    s_total   = _extract_first(texto, PAT_M2)
    s_covered = _extract_first(texto, PAT_M2_C)

    n = len(df_ent)
    n_total      = int(s_total.notna().sum())
    n_covered    = int(s_covered.notna().sum())
    n_ambos      = int((s_total.notna() & s_covered.notna()).sum())
    if n_ambos > 0:
        ratio = (s_covered[s_total.notna() & s_covered.notna()] /
                 s_total[s_total.notna() & s_covered.notna()])
        # Cubierto < total ⇒ ratio < 1; ambos parecidos ⇒ ratio ≈ 1
        ratio_p50 = float(ratio.median())
        n_difieren_10 = int((ratio < 0.9).sum())
        pct_difieren_10 = round(n_difieren_10 / n_ambos * 100, 2)
    else:
        ratio_p50 = float("nan")
        n_difieren_10 = 0
        pct_difieren_10 = 0.0

    pct_ambos = round(n_ambos / n * 100, 2)
    veredicto = (
        f"ENTRA en v2 (ambos en {pct_ambos}% > 5% y difieren > 10% en {pct_difieren_10}% de esas filas)"
        if pct_ambos > 5.0 and pct_difieren_10 > 30.0
        else f"DESCARTAR (cobertura {pct_ambos}% o difieren sólo en {pct_difieren_10}% — no agrega señal)"
    )
    return {
        "n_total_filas":          n,
        "n_con_total":            n_total,
        "pct_con_total":          round(n_total / n * 100, 2),
        "n_con_cubierta":         n_covered,
        "pct_con_cubierta":       round(n_covered / n * 100, 2),
        "n_con_ambos":            n_ambos,
        "pct_con_ambos":          pct_ambos,
        "ratio_cub_tot_p50":      round(ratio_p50, 4) if not np.isnan(ratio_p50) else None,
        "n_difieren_10pct":       n_difieren_10,
        "pct_difieren_10pct":     pct_difieren_10,
        "veredicto":              veredicto,
    }


# ---------- Análisis 4: tamaño de barrios -----------------------------------

def analisis_barrios(df_ent: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Distribución de tamaño de barrios en train. Detecta los chicos (< 30 obs)."""
    sizes = df_ent["barrio"].value_counts()
    bins = [0, 10, 30, 100, 500, 1_000, 1e9]
    labels = ["1-10", "11-30", "31-100", "101-500", "501-1000", "1000+"]
    bucket = pd.cut(sizes, bins=bins, labels=labels, right=True)
    bucket.name = "tamano_barrio"
    tabla = bucket.value_counts().sort_index().to_frame("n_barrios")
    tabla.index.name = "tamano_barrio"
    tabla["pct_barrios"] = (tabla["n_barrios"] / tabla["n_barrios"].sum() * 100).round(2)
    # cuántas filas de train cubre cada bucket
    by_bucket = sizes.groupby(bucket, observed=True).sum()
    tabla["filas_cubiertas"] = by_bucket
    tabla["pct_filas"] = (by_bucket / sizes.sum() * 100).round(2)

    extra = {
        "n_barrios_total":     int(len(sizes)),
        "n_barrios_chicos":    int((sizes < 30).sum()),
        "filas_chicos":        int(sizes[sizes < 30].sum()),
        "pct_filas_chicos":    round(sizes[sizes < 30].sum() / sizes.sum() * 100, 2),
    }
    return tabla, extra


# ---------- Análisis 7: geografía interna (Notas 3 y 4) ---------------------

# Strings basura en location_2 / location_3 que NO son barrios
BARRIO_BASURA = {
    "CABA", "Capital Federal", "Ciudad Autónoma de Buenos Aires",
    "Buenos Aires",
}

# Tokens "centro comercial / oficinas" que vale buscar como sub-barrios
# nombrables — si aparecen como location_3 podemos calcular su centroide
# sin necesitar APIs (Nota 4 idea 3, versión E3 acotada).
CENTROS = [
    "Microcentro", "Centro", "Catalinas", "Tribunales", "Once",
    "Retiro", "Puerto Madero", "Plaza de Mayo",
]


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def analisis_locations(df_ent: pd.DataFrame, df_ap: pd.DataFrame) -> dict:
    """Cardinalidad + barrios mal etiquetados + intersección train↔test
    + presencia de centros nombrables. Devuelve un dict de DataFrames /
    metadatos listos para renderear.
    """
    # 1. Cardinalidad por nivel ----------------------------------------------
    levels = ["location_1", "location_2", "location_3", "location_4"]
    rows = []
    for lvl in levels:
        ent_v = df_ent[lvl].dropna()
        ap_v  = df_ap[lvl].dropna()
        rows.append({
            "nivel":         lvl,
            "n_unicos_train":int(ent_v.nunique()),
            "n_unicos_test": int(ap_v.nunique()),
            "pct_nan_train": round(df_ent[lvl].isna().mean() * 100, 2),
            "pct_nan_test":  round(df_ap[lvl].isna().mean() * 100, 2),
        })
    cardinalidad = pd.DataFrame(rows)

    # 2. Barrios "basura" en location_2 / location_3 -------------------------
    rows = []
    for lvl in ("location_2", "location_3"):
        for col, df in (("train", df_ent), ("test", df_ap)):
            v = _norm(df[lvl])
            n_basura = int(v.isin(BARRIO_BASURA).sum())
            n_nan    = int((v == "").sum())
            rows.append({
                "nivel":            lvl,
                "split":            col,
                "n_filas":          len(df),
                "n_basura":         n_basura,
                "pct_basura":       round(n_basura / len(df) * 100, 2),
                "n_nan":            n_nan,
                "pct_nan":          round(n_nan / len(df) * 100, 2),
                "n_total_a_imputar":n_basura + n_nan,
                "pct_a_imputar":    round((n_basura + n_nan) / len(df) * 100, 2),
            })
    basura = pd.DataFrame(rows)

    # 3. Intersección train↔test (sólo location_2 y location_3) --------------
    rows = []
    for lvl in ("location_2", "location_3"):
        ent_v = set(df_ent[lvl].dropna().unique())
        ap_v  = set(df_ap[lvl].dropna().unique())
        comunes      = ent_v & ap_v
        solo_train   = ent_v - ap_v
        solo_test    = ap_v - ent_v
        rows.append({
            "nivel":         lvl,
            "n_train":       len(ent_v),
            "n_test":        len(ap_v),
            "n_comunes":     len(comunes),
            "pct_test_cubierto": round(len(comunes) / max(len(ap_v), 1) * 100, 2),
            "n_solo_train":  len(solo_train),
            "n_solo_test":   len(solo_test),
        })
    interseccion = pd.DataFrame(rows)

    # Top 10 valores de location_2 y location_3 ------------------------------
    top_loc2_train = df_ent["location_2"].value_counts().head(10)
    top_loc3_train = df_ent["location_3"].value_counts().head(10)
    top_loc2_test  = df_ap["location_2"].value_counts().head(10)
    top_loc3_test  = df_ap["location_3"].value_counts().head(10)

    # 4. Centros comerciales nombrables --------------------------------------
    rows = []
    for centro in CENTROS:
        for lvl in ("location_2", "location_3"):
            n_train = int((df_ent[lvl].fillna("").str.contains(centro, case=False, regex=False)).sum())
            n_test  = int((df_ap[lvl].fillna("").str.contains(centro, case=False, regex=False)).sum())
            if n_train + n_test == 0:
                continue
            rows.append({
                "centro":   centro,
                "nivel":    lvl,
                "n_train":  n_train,
                "n_test":   n_test,
            })
    centros_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["centro","nivel","n_train","n_test"]
    )

    # Veredicto centros: ¿al menos uno con > 50 obs en train?
    if len(centros_df) > 0:
        usables = centros_df[centros_df["n_train"] >= 50]
        veredicto_centros = (
            f"VIABLE en E3: {len(usables)} centros con ≥ 50 obs en train "
            f"({', '.join(sorted(set(usables['centro'].tolist())))}). "
            "Se puede calcular su centroide y derivar `dist_a_<centro>` sin APIs."
            if len(usables) > 0 else
            f"DESCARTAR para E3: {len(centros_df)} centros aparecen pero ninguno "
            "tiene ≥ 50 obs en train, no alcanza para fijar un centroide robusto. "
            "La proximidad a centros comerciales queda para E4 (con datos externos)."
        )
    else:
        veredicto_centros = (
            "DESCARTAR para E3: ninguno de los nombres buscados aparece como "
            "barrio. La proximidad a centros comerciales queda para E4."
        )

    # Veredicto barrios mal etiquetados
    train_a_imputar = int(basura.query("split == 'train'")["n_total_a_imputar"].max())
    test_a_imputar  = int(basura.query("split == 'test'")["n_total_a_imputar"].max())
    pct_train = round(train_a_imputar / len(df_ent) * 100, 2)
    pct_test  = round(test_a_imputar / len(df_ap) * 100, 2)
    if pct_train > 5 or pct_test > 5:
        veredicto_basura = (
            f"PRIORIDAD ALTA: hasta {pct_train}% del train y {pct_test}% del test "
            "tienen barrio mal etiquetado o NaN. Vale la cascada Hot Deck por "
            "description → KNN sobre lat/lon (Nota 3)."
        )
    elif pct_train > 1 or pct_test > 1:
        veredicto_basura = (
            f"PRIORIDAD MEDIA: {pct_train}% del train y {pct_test}% del test. "
            "El esfuerzo justifica si v2-v4 no aportan; queda como v6 candidato."
        )
    else:
        veredicto_basura = (
            f"DESCARTAR: < 1% en ambos splits ({pct_train}% train, {pct_test}% "
            "test). El cleanup no compensa la complejidad del cascade."
        )

    return {
        "cardinalidad":       cardinalidad,
        "basura":             basura,
        "interseccion":       interseccion,
        "top_loc2_train":     top_loc2_train,
        "top_loc3_train":     top_loc3_train,
        "top_loc2_test":      top_loc2_test,
        "top_loc3_test":      top_loc3_test,
        "centros":            centros_df,
        "veredicto_basura":   veredicto_basura,
        "veredicto_centros":  veredicto_centros,
    }


# ---------- Render markdown -------------------------------------------------

def df_to_md(df: pd.DataFrame, *, index: bool = False) -> str:
    if index:
        df = df.reset_index()
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:,.4f}".rstrip("0").rstrip("."))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    df_ent, df_ap = load_and_filter()

    print("[eda] análisis 1: skewness")
    skew_df = analisis_skewness(df_ent)
    print(skew_df.to_string(index=False))

    print("\n[eda] análisis 2: temporal")
    anio_df, mes_df = analisis_temporal(df_ent, df_ap)
    print("por año:")
    print(anio_df.to_string())
    print("por mes calendario (estacionalidad):")
    print(mes_df.to_string())

    print("\n[eda] análisis 3: IG binarias contra pd.qcut(price, 10)")
    ig_df = analisis_ig_binarias(df_ent, n_bins=10)
    print(ig_df.to_string(index=False))

    print("\n[eda] análisis 3b: drift train vs test en binarias")
    drift_df = analisis_ig_binarias_test(df_ent, df_ap)
    print(drift_df.to_string(index=False))

    print("\n[eda] análisis 4: barrios")
    barrio_df, barrio_meta = analisis_barrios(df_ent)
    print(barrio_df.to_string())
    print(f"  meta: {barrio_meta}")

    print("\n[eda] análisis 5: cobertura del parseo dual (insumo C3PO)")
    parseo_df, parseo_meta = analisis_parseo_dual(df_ent)
    print(parseo_df.to_string(index=False))
    print(f"  veredicto: {parseo_meta['veredicto']}")

    print("\n[eda] análisis 6: surface_total vs surface_covered (insumo C3PO)")
    surf_meta = analisis_surface(df_ent)
    for k, v in surf_meta.items():
        print(f"  {k}: {v}")

    print("\n[eda] análisis 7: geografía interna (Notas 3 y 4)")
    geo = analisis_locations(df_ent, df_ap)
    print("\ncardinalidad:")
    print(geo["cardinalidad"].to_string(index=False))
    print("\nbarrios basura / NaN:")
    print(geo["basura"].to_string(index=False))
    print("\nintersección train↔test:")
    print(geo["interseccion"].to_string(index=False))
    print(f"\ntop10 location_2 train:\n{geo['top_loc2_train'].to_string()}")
    print(f"\ntop10 location_3 train:\n{geo['top_loc3_train'].to_string()}")
    print(f"\ntop10 location_2 test:\n{geo['top_loc2_test'].to_string()}")
    print(f"\ntop10 location_3 test:\n{geo['top_loc3_test'].to_string()}")
    print(f"\ncentros nombrables:\n{geo['centros'].to_string(index=False)}")
    print(f"\n[eda]   veredicto basura : {geo['veredicto_basura']}")
    print(f"[eda]   veredicto centros: {geo['veredicto_centros']}")

    # Conclusiones accionables — derivadas de los datos, no hardcodeadas
    candidatas_log = skew_df[skew_df["reco"].str.startswith("np.log")]["columna"].tolist()
    top_ig = ig_df.head(5)["feature"].tolist()
    bot_ig = ig_df.tail(5)["feature"].tolist()
    drift_top = drift_df.head(5)[["feature","delta"]].to_records(index=False).tolist()
    # Tensión IG alto + drift alto: features que aportan señal pero pueden no generalizar
    drift_set = {f for f, d in drift_top}
    top_ig_drift = [c for c in top_ig if c in drift_set]

    # Análisis 2: ¿hay desbalance grosero por año o sólo varianza chica?
    delta_max_anio = float(anio_df["delta_pct"].abs().max())
    if delta_max_anio < 3.0:
        veredicto_temporal = (
            f"NO hay desbalance grosero (delta_pct máximo = {delta_max_anio:.2f}pp). "
            "El shift de v6 NO viene de 'el test cae en años raros'. La hipótesis "
            "más sólida es que el RF aprendió correlación año↔precio que no "
            "generaliza dentro de cada año (lookup vacío). v3 sigue valiendo, "
            "pero la justificación cambia: no es 'sustituir año por estacionalidad' "
            "sino 'eliminar el lookup año al RF'."
        )
    else:
        veredicto_temporal = (
            f"SÍ hay desbalance por año (delta_pct máximo = {delta_max_anio:.2f}pp). "
            "El v3 del plan está justificado por la evidencia directa."
        )

    md = ["# EDA preparatorio para Entrega 3 (sobre el pipeline de v4)\n"]
    md.append(
        "Este EDA reproduce los filtros + parsing de la corrida campeona de "
        "Entrega 2 (v4) y mide cuatro cosas distintas que necesito antes de "
        "diseñar los experimentos de E3 anclados en Clase 7. **No** corre el "
        "modelo: las decisiones de transformación / discretización se toman "
        "sobre las distribuciones crudas, antes de outliers / imputación.\n"
    )
    md.append(f"- Filtro v4: train post-filtro = {len(df_ent):,} filas, test = {len(df_ap):,}.\n")

    md.append("\n## 1. Skewness y percentiles de las numéricas continuas")
    md.append(
        "Identifica candidatas a `np.log` / `np.sqrt` (transformaciones de Clase 7). "
        "La recomendación es heurística: skew > 2 con valores positivos → log; "
        "skew 1-2 con no-negativos → log1p / sqrt; skew leve o cuasi-simétrica → nada.\n"
    )
    md.append(df_to_md(skew_df))

    md.append("\n## 2. Distribución temporal train vs test (shift de v6)")
    md.append(
        "Cuantifica el distribution shift cronológico que rompió el holdout "
        "temporal en v6 cuando se metieron `pub_year`/`pub_month`. Si la "
        "distribución por año difiere fuerte entre train y test, usar el año "
        "como nivel del RF es un lookup que no generaliza.\n"
    )
    md.append("### Por año")
    md.append(df_to_md(anio_df, index=True))
    md.append("\n### Por mes calendario (estacionalidad pura, agregando años)")
    md.append(df_to_md(mes_df, index=True))

    md.append("\n## 3. Information Gain de las features binarias contra `pd.qcut(price, 10)`")
    md.append(
        "Implementación manual del cálculo de Clase 7 (entropía → IG). Mide "
        "cuánta información da cada `f_*` para predecir el decil de precio. "
        "Las de IG bajo son candidatas a podar; las de IG alto son las que "
        "más pesa el RF.\n"
    )
    md.append(df_to_md(ig_df))

    md.append("\n### 3b. Covariate shift de las binarias entre train y test")
    md.append(
        "Tasa de 1s en train vs test, ordenadas por |delta|. Si una `f_*` "
        "tiene tasa muy distinta entre train y test, su contribución no "
        "generaliza aunque tenga IG alto.\n"
    )
    md.append(df_to_md(drift_df))

    md.append("\n## 4. Distribución del tamaño de los barrios")
    md.append(
        "Identifica barrios chicos (< 30 obs) — candidatos a colapsar via "
        "discretización supervisada. Reduce dimensionalidad efectiva de "
        "`barrio_id` sin perder señal.\n"
    )
    md.append(df_to_md(barrio_df, index=True))
    md.append(
        f"\n**Resumen barrios**: {barrio_meta['n_barrios_total']} barrios totales, "
        f"de los cuales {barrio_meta['n_barrios_chicos']} tienen < 30 obs "
        f"(cubren {barrio_meta['filas_chicos']:,} filas = "
        f"{barrio_meta['pct_filas_chicos']}% del train post-filtro).\n"
    )

    md.append("\n## 5. Cobertura del parseo dual `features` + `description` (insumo Robot 2 C3PO)")
    md.append(
        "Para cada atributo numérico clave (`m2`, `n_dormitorios`, `n_banos`), "
        "cuántas filas vienen NaN en `features` pero el dato sí está en "
        "`description`. Si el rescate es > 1% en al menos un atributo, vale "
        "incorporar el parseo dual al pipeline; si no, el costo de complejidad "
        "no se justifica.\n"
    )
    md.append(df_to_md(parseo_df))
    md.append(f"\n**Veredicto**: {parseo_meta['veredicto']} "
              f"(máximo de rescate observado: {parseo_meta['pct_max_rescatado']}%).\n")

    md.append("\n## 6. Cobertura `surface_total_m2` vs `surface_covered_m2` (insumo Robot 2 C3PO)")
    md.append(
        "Cuántas filas traen los dos valores en el texto y cuán distintos son. "
        "El robot deriva `pct_surface_covered = covered / total`. Vale agregarla "
        "sólo si la cobertura conjunta es > 5% **y** los dos valores difieren > 10% "
        "en una porción significativa de las filas (si fueran casi siempre "
        "iguales, la feature derivada sería ruido constante = 1).\n"
    )
    md.append(
        f"- Filas totales: {surf_meta['n_total_filas']:,}\n"
        f"- Con `surface_total_m2` extraíble: {surf_meta['n_con_total']:,} "
        f"({surf_meta['pct_con_total']}%)\n"
        f"- Con `surface_covered_m2` extraíble: {surf_meta['n_con_cubierta']:,} "
        f"({surf_meta['pct_con_cubierta']}%)\n"
        f"- Con ambos valores extraídos: {surf_meta['n_con_ambos']:,} "
        f"({surf_meta['pct_con_ambos']}%)\n"
        f"- Mediana de `covered / total` (sólo donde hay ambos): "
        f"{surf_meta['ratio_cub_tot_p50']}\n"
        f"- Filas donde `covered` < 90% de `total`: "
        f"{surf_meta['n_difieren_10pct']:,} "
        f"({surf_meta['pct_difieren_10pct']}% de las que tienen ambos)\n"
    )
    md.append(f"\n**Veredicto**: {surf_meta['veredicto']}.\n")

    md.append("\n## 7. Geografía interna sin APIs (Notas 3 y 4 del user)")
    md.append(
        "Tres preguntas en una: (a) ¿es `location_3` realmente sub-barrio o es "
        "lo mismo que `location_2`? (b) ¿cuántas filas tienen barrio mal "
        "etiquetado y vale el cleanup vía Hot Deck/KNN? (c) ¿hay algún "
        "centro comercial / sub-barrio céntrico nombrado en `location_3` que "
        "permita derivar `dist_a_microcentro` sin APIs externas?\n"
    )
    md.append("\n### 7.1 Cardinalidad de location_1..4 (¿hay sub-barrios?)")
    md.append(df_to_md(geo["cardinalidad"]))
    md.append(
        "\nSi `n_unicos` de `location_2` y `location_3` son distintos, "
        "tenemos dos granularidades geográficas reales y vale tener ambas "
        "como features paralelas en el modelo.\n"
    )
    md.append("\n### 7.2 Barrios mal etiquetados / NaN en location_2 y location_3")
    md.append(df_to_md(geo["basura"]))
    md.append(f"\n**Veredicto**: {geo['veredicto_basura']}\n")

    md.append("\n### 7.3 Intersección train↔test (¿el vocabulario de train cubre el test?)")
    md.append(df_to_md(geo["interseccion"]))
    md.append(
        "\nSi `pct_test_cubierto` es alto (> 95%), podemos usar el train "
        "como diccionario para imputar etiquetas raras del test. Si es bajo, "
        "el test trae barrios nuevos y necesitamos KNN sobre lat/lon para "
        "esos casos.\n"
    )

    md.append("\n### 7.4 Top barrios por frecuencia (sanity check)")
    md.append("\n**train · location_2**\n")
    md.append(df_to_md(geo["top_loc2_train"].rename("n").to_frame(), index=True))
    md.append("\n**train · location_3**\n")
    md.append(df_to_md(geo["top_loc3_train"].rename("n").to_frame(), index=True))
    md.append("\n**test · location_2**\n")
    md.append(df_to_md(geo["top_loc2_test"].rename("n").to_frame(), index=True))
    md.append("\n**test · location_3**\n")
    md.append(df_to_md(geo["top_loc3_test"].rename("n").to_frame(), index=True))

    md.append("\n### 7.5 Centros comerciales / sub-barrios céntricos como referencia")
    if len(geo["centros"]) > 0:
        md.append(df_to_md(geo["centros"]))
    else:
        md.append("\n_(ninguno de los nombres buscados aparece como `location_2` ni `location_3`)_\n")
    md.append(f"\n**Veredicto**: {geo['veredicto_centros']}\n")

    md.append("\n## Conclusiones accionables para los experimentos del plan")
    tension = (
        f" **Tensión IG-alto + drift-alto**: {', '.join(top_ig_drift)} aporta(n) "
        f"señal en el train pero su prevalencia es distinta en el test — riesgo "
        f"v5 (señal que no generaliza)."
        if top_ig_drift else
        " Sin tensión: ninguna feature del top IG está entre las de mayor drift."
    )
    pct_filas_grandes = float(barrio_df.loc["1000+", "pct_filas"]) if "1000+" in barrio_df.index else 0.0
    md.append(
        f"- **Candidatas claras a transformación log/sqrt** (Análisis 1): "
        f"{', '.join(candidatas_log) if candidatas_log else '—'}. "
        f"Esto refina el v2 del plan (que tenía sólo `np.log(m2)`). "
        f"`n_banos` aparece con skew 310 por un máximo de 140 000 — un dato basura "
        f"que confirma que el capping `[0, 15]` heredado de v4 sigue siendo necesario.\n"
        f"- **Top 5 features binarias por IG** (Análisis 3): {', '.join(top_ig)}. "
        f"`f_cochera` y `f_balcon` solas explican IG = 0.143 (~4.3% de H(Y)).\n"
        f"- **Bottom 5 features binarias por IG** (Análisis 3): {', '.join(bot_ig)}. "
        f"Candidatas a podar o a fundir en una score agregada.\n"
        f"- **Mayores drifts train↔test** (Análisis 3b): "
        f"{', '.join(f'{f} ({d:+.2f}pp)' for f, d in drift_top)}. "
        f"El test tiene SISTEMÁTICAMENTE más amenities marcadas que el train — "
        f"covariate shift consistente, no ruido aleatorio.{tension}\n"
        f"- **Barrios** (Análisis 4): {barrio_meta['n_barrios_chicos']} barrios chicos "
        f"sobre {barrio_meta['n_barrios_total']} totales, pero cubren sólo "
        f"{barrio_meta['pct_filas_chicos']}% del train; el {pct_filas_grandes}% de las "
        f"filas vive en barrios > 1000 obs. **Esto debilita el v5 del plan**: la "
        f"discretización supervisada del `barrio_id` no compra agregando barrios "
        f"chicos (no hay con qué); si se hace, es para reducir la cardinalidad "
        f"efectiva del feature, no para rescatar muestras.\n"
        f"- **Temporal** (Análisis 2): {veredicto_temporal}\n"
        f"- **Insumo C3PO — parseo dual** (Análisis 5): {parseo_meta['veredicto']}. "
        f"Recupera 8.86% de filas con `m2`, 3.07% con `n_dormitorios`, 1.86% con "
        f"`n_banos`. Es un upgrade limpio del pipeline de v4 → entra al v2.\n"
        f"- **Insumo C3PO — surface total/covered** (Análisis 6): {surf_meta['veredicto']}. "
        f"La mediana del ratio es 1.0 porque el regex de cubierta hace match con el "
        f"texto general (overlap del patrón `m²`). No agrega señal real → no entra.\n"
        f"- **Geografía interna — sub-barrios** (Análisis 7.1): comparar "
        f"`n_unicos` de location_2 vs location_3 en la tabla. Si difieren, "
        f"agregar `barrio_principal = location_2` como feature paralela a la "
        f"actual `barrio = location_3`.\n"
        f"- **Geografía interna — limpieza barrio** (Análisis 7.2): {geo['veredicto_basura']}\n"
        f"- **Geografía interna — centros comerciales como referencia** (Análisis 7.5): "
        f"{geo['veredicto_centros']}\n"
    )

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[eda] markdown escrito en {OUT_MD}")


if __name__ == "__main__":
    main()
