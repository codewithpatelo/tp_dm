"""
Patch notebook for v5: insert quality+temporal features cell at index 16,
update NUM_MED in imputation cell to include floor/pub_year/pub_month.
"""
import json, sys, re
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "Colab_Base_para_el_Trabajo_Práctico_(Entrega_2).ipynb"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

# ── 1. New cell: quality + temporal features ─────────────────────────────────
NEW_CELL_SOURCE = '''\
# --- v5: features de calidad / estado y temporales desde description + features ---
# Hipótesis: floor (piso), a_estrenar, reciclado, suite, subte y pub_year/pub_month
# capturan señal de calidad y ciclo de mercado sin depender de agregados del train
# (evitamos el patrón de leakage confirmado en v2/v3).

def extract_quality_features(df_):
    """Extrae indicadores de calidad/estado y cercanía desde description + features."""
    import numpy as np
    out = {}
    desc = df_["description"].fillna("").astype(str).str.lower()
    feat = df_["features"].fillna("").astype(str).str.lower()
    for src, dst in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n")]:
        desc = desc.str.replace(src, dst, regex=False)
        feat = feat.str.replace(src, dst, regex=False)
    combined = desc + " " + feat

    # Número de piso: "piso 8", "8° piso", "8 piso", "8º piso"
    # En CABA pisos altos → vista/precio premium
    _fp1 = combined.str.extract(r"piso\s+(\d{1,2})", expand=False)
    _fp2 = combined.str.extract(r"(\d{1,2})\s*(?:er|do|ro|to|[°º])\s*piso", expand=False)
    floor_vals = _fp1.fillna(_fp2).astype(float)
    # Capamos fuera de rango válido [1,50]
    floor_vals[~floor_vals.between(1, 50)] = float("nan")
    out["floor"] = floor_vals

    # Calidad / estado
    out["is_a_estrenar"]    = (desc.str.contains("a estrenar", regex=False) |
                               desc.str.contains("estrenar",   regex=False)).astype(int)
    out["is_reciclado"]     = desc.str.contains("reciclado",   regex=False).astype(int)
    out["has_suite"]        = combined.str.contains("suite",   regex=False).astype(int)
    out["has_service_room"] = (
        combined.str.contains("dependencia de servicio", regex=False) |
        combined.str.contains("cuarto de servicio",      regex=False)
    ).astype(int)

    # Accesibilidad transporte
    out["is_near_subway"] = (
        desc.str.contains("subte",    regex=False) |
        desc.str.contains("metrobus", regex=False)
    ).astype(int)

    return pd.DataFrame(out, index=df_.index)


def extract_temporal_features(df_):
    """Parsea publication_date (ej. '15 oct 2023') → pub_year, pub_month."""
    MESES = {"ene":1,"feb":2,"mar":3,"abr":4,"may":5,"jun":6,
             "jul":7,"ago":8,"sept":9,"sep":9,"oct":10,"nov":11,"dic":12}
    years, months = [], []
    pat = re.compile(r"(\\d{1,2})\\s+(\\w+)\\s+(\\d{4})")
    for v in df_["publication_date"].fillna(""):
        m = pat.match(str(v).strip())
        if m:
            _, mes, a = m.groups()
            mes_n = MESES.get(mes.lower())
            years.append(int(a) if mes_n else float("nan"))
            months.append(mes_n if mes_n else float("nan"))
        else:
            years.append(float("nan"))
            months.append(float("nan"))
    return pd.DataFrame({"pub_year": years, "pub_month": months}, index=df_.index)


qual_ent  = extract_quality_features(df_ent)
qual_ap   = extract_quality_features(df_ap)
temp_ent  = extract_temporal_features(df_ent)
temp_ap   = extract_temporal_features(df_ap)

df_ent = pd.concat([df_ent, qual_ent, temp_ent], axis=1)
df_ap  = pd.concat([df_ap,  qual_ap,  temp_ap],  axis=1)

# Resumen rápido para verificar cobertura
print("Quality / temporal features — train:")
cols_show = ["floor","is_a_estrenar","is_reciclado","has_suite","pub_year","pub_month"]
print(df_ent[cols_show].describe().round(2))
print("\\nCobertura pub_year train :", df_ent["pub_year"].notna().mean().round(3))
print("Cobertura floor train     :", df_ent["floor"].notna().mean().round(3))
'''

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": NEW_CELL_SOURCE,
}

# ── 2. Find insertion point: after the EDA cell (which contains "Percentil") ─
# The plan says insert at index 16 (0-based), after the EDA cell added as cell 15.
# We find the EDA cell by content to be robust to index shifts.
eda_idx = None
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "Percentil" in src and "m2" in src and "n_dormitorios" in src:
        eda_idx = i
        break

if eda_idx is None:
    # Fallback: insert after parse_features cell (contains "def parse_features")
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "def parse_features" in src:
            eda_idx = i
            break

if eda_idx is None:
    print("ERROR: no se encontró la celda EDA ni parse_features.", file=sys.stderr)
    sys.exit(1)

insert_at = eda_idx + 1
print(f"Insertando celda v5 en índice {insert_at} (después de celda {eda_idx})")

# Check we're not inserting twice
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "extract_quality_features" in src:
        print(f"ADVERTENCIA: ya existe una celda con extract_quality_features (idx {i}). Reemplazando.")
        cells[i] = new_cell
        break
else:
    cells.insert(insert_at, new_cell)

# ── 3. Update NUM_MED in the imputation cell ─────────────────────────────────
# NUM_MED should include floor, pub_year, pub_month
impute_idx = None
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "NUM_MED" in src and "SimpleImputer" in src:
        impute_idx = i
        break

if impute_idx is None:
    print("ERROR: no se encontró la celda de imputación (NUM_MED).", file=sys.stderr)
    sys.exit(1)

src_lines = cells[impute_idx]["source"]
full_src = src_lines if isinstance(src_lines, str) else "".join(src_lines)

NEW_FEATURES = '"floor", "pub_year", "pub_month"'

if "floor" in full_src:
    print(f"Celda de imputación (idx {impute_idx}): floor ya presente, sin cambios.")
else:
    # Find the NUM_MED = [...] block and add new features
    # Pattern: NUM_MED = [ ... ]  (possibly multiline)
    # Strategy: find the closing bracket of NUM_MED assignment and insert before it
    def insert_into_num_med(src):
        # Find NUM_MED = [ and then the matching ]
        start = src.find("NUM_MED")
        if start == -1:
            return src, False
        bracket_open = src.find("[", start)
        if bracket_open == -1:
            return src, False
        bracket_close = src.find("]", bracket_open)
        if bracket_close == -1:
            return src, False
        # Insert before the closing bracket
        # Check if last item already has a trailing comma
        inner = src[bracket_open+1:bracket_close].rstrip()
        if inner.endswith(","):
            new_inner = inner + f"\n    {NEW_FEATURES}"
        else:
            new_inner = inner + f",\n    {NEW_FEATURES}"
        return src[:bracket_open+1] + new_inner + "\n" + src[bracket_close:], True

    new_src, ok = insert_into_num_med(full_src)
    if ok:
        cells[impute_idx]["source"] = new_src
        print(f"Celda de imputación (idx {impute_idx}): floor/pub_year/pub_month añadidos a NUM_MED.")
    else:
        print(f"ADVERTENCIA: no se pudo modificar NUM_MED en celda {impute_idx}.")

# ── 4. Save ──────────────────────────────────────────────────────────────────
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook guardada: {NB_PATH}")
print("Patch v5 completado.")
