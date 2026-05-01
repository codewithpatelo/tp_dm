# --- §2.4 Feature Engineering — v5: distancias a centros de referencia (sin API) ---
#
# Motivacion: lat/lon crudos dan al RF gradiente de precio por zona pero con
# discontinuidades no lineales. Distancias a centros nombrados capturan la
# estructura radial del mercado de CABA sin usar APIs ni datos externos.
#
# Metodo: centroide (median lat/lon) por barrio en train, top-N barrios con
# >= MIN_CENTRO_OBS obs y lat/lon validos. Distancia euclidea * 111 km/grado.
# Imputacion: mediana train para filas con lat/lon NaN.

import numpy as np, re as _re5, unicodedata as _ucd5

_MIN_CENTRO_OBS = 50
_N_CENTROS_MAX  = 10

def _slug(s):
    s = _ucd5.normalize("NFKD", str(s))
    s = "".join(c for c in s if not _ucd5.combining(c))
    s = _re5.sub(r"[^a-z0-9]", "_", s.lower())
    s = _re5.sub(r"_+", "_", s).strip("_")
    return s[:20]

# Construir centroides desde train (barrios limpios con suficientes obs)
_bc_valid = df_ent[
    (~df_ent["barrio"].isin({"desconocido", "barrio_raro"})) &
    df_ent["lat"].notna() & df_ent["lon"].notna()
].copy()
_barrio_stats = (
    _bc_valid.groupby("barrio")
    .agg(lat_med=("lat","median"), lon_med=("lon","median"), n=("lat","count"))
    .reset_index()
    .query("n >= @_MIN_CENTRO_OBS")
    .sort_values("n", ascending=False)
    .head(_N_CENTROS_MAX)
    .reset_index(drop=True)
)

print(f"[FE v5] {len(_barrio_stats)} centros de referencia (>={_MIN_CENTRO_OBS} obs):")
for _, r in _barrio_stats.iterrows():
    print(f"  {r['barrio']}: lat={r['lat_med']:.4f} lon={r['lon_med']:.4f} n={r['n']}")

# Calcular distancias
_dist_cols_v5 = []
for _, centro in _barrio_stats.iterrows():
    col_name = "dist_" + _slug(centro["barrio"])

    dist_train = np.sqrt(
        (df_ent["lat"] - centro["lat_med"]) ** 2 +
        (df_ent["lon"] - centro["lon_med"]) ** 2
    ) * 111.0
    _med = float(dist_train.median())
    df_ent[col_name] = dist_train.fillna(_med)

    dist_test = np.sqrt(
        (df_ap["lat"] - centro["lat_med"]) ** 2 +
        (df_ap["lon"] - centro["lon_med"]) ** 2
    ) * 111.0
    df_ap[col_name] = dist_test.fillna(_med)

    _dist_cols_v5.append(col_name)

print(f"[FE v5] columnas agregadas: {_dist_cols_v5}")
print(df_ent[_dist_cols_v5].describe().round(2))

EXPERIMENT_LOG["params"]["fe_distancias"] = {
    "descripcion": f"distancias euclideas (*111 km) a top-{_N_CENTROS_MAX} barrios por frecuencia (>={_MIN_CENTRO_OBS} obs)",
    "n_centros": len(_barrio_stats),
    "centros": _barrio_stats[["barrio","lat_med","lon_med","n"]].to_dict("records"),
    "columnas_output": _dist_cols_v5,
}
