"""Validación end-to-end con la consulta REAL del usuario.

Pregunta: "Si deseo comprar una propiedad de 3 ambientes de 50 a 80 m2 en
2026 por San Cristobal, Boedo o Parque Patricios, a qué precio debería
considerar un buen trato / negocio?"

Verifica:
1. El system prompt ahora tiene schema REAL (no inventado).
2. El helper `infer_real_estate_fields` está disponible en sandbox y rápido.
3. El sandbox responde dentro del timeout y publica artifacts útiles.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gui.data_loader import DataRegistry  # noqa: E402
from gui.sandbox import execute_analysis  # noqa: E402
from gui.system_prompt import codegen_system_prompt  # noqa: E402

print("=== VALIDACIÓN BUYER QUERY ===\n")

# 1) System prompt: chequeo que no tenga las columnas inventadas y SÍ las reales.
print("[1/3] System prompt...")
sp = codegen_system_prompt()
assert "infer_real_estate_fields" in sp, "el helper no aparece en el prompt"
assert "_rooms_est" in sp, "el prompt no menciona _rooms_est"
assert "_surface_m2_est" in sp, "el prompt no menciona _surface_m2_est"
assert "NO existen columnas estructuradas como `rooms`" in sp, \
    "el prompt no advierte sobre columnas inexistentes"
# Verificar que el snapshot incluya las columnas reales.
for col in ("description", "features", "location_2", "location_3",
            "property_type", "price", "pub_yyyymm"):
    assert col in sp, f"el snapshot no menciona la columna real `{col}`"
# Y que NO promete columnas inexistentes como columnas estructuradas.
assert "rooms (int" not in sp.lower(), "el prompt sigue listando rooms como estructurada"
print(f"  OK ({len(sp):,} chars de prompt)\n")

# 2) Cargar datasets.
print("[2/3] Cargando datasets...", end=" ", flush=True)
t0 = time.time()
reg = DataRegistry.get()
train = reg.table("train_filtered")
test = reg.table("test")
print(f"OK ({time.time() - t0:.1f}s)\n")

# 3) Simular el script que el LLM debería generar.
print("[3/3] Ejecutando análisis tipo-LLM...")
ANALYSIS_CODE = r"""
import unicodedata

def _norm(s):
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip()

# Paso 1: usar helper para tener _rooms_est y _surface_m2_est ya inferidos.
df = infer_real_estate_fields(train_filtered)

# Paso 2: filtrar deptos.
mask_dep = df["property_type"].astype(str).str.lower().str.contains(
    "departamento", na=False, regex=False
)
df = df[mask_dep]
print("Tras property_type=departamento:", len(df))

# Paso 3: filtrar barrios pedidos.
barrios = ["san cristobal", "boedo", "parque patricios"]
loc2_n = df["location_2"].astype(str).map(_norm)
loc3_n = df["location_3"].astype(str).map(_norm)
mask_b = pd.Series(False, index=df.index)
for needle in barrios:
    mask_b = mask_b | loc2_n.str.contains(needle, na=False, regex=False)
    mask_b = mask_b | loc3_n.str.contains(needle, na=False, regex=False)
df = df[mask_b]
print("Tras filtro de barrios:", len(df))

# Paso 4: filtros de 3 ambientes y 50-80 m2.
df_amb = df[df["_rooms_est"] == 3]
print("Tras 3 ambientes:", len(df_amb))

work = df_amb[df_amb["_surface_m2_est"].between(50, 80)]
print("Tras 50-80 m2:", len(work))

# Paso 5: filtro temporal 2026 con fallback si la muestra queda chica.
work_2026 = work[work["pub_yyyymm"].between(202601, 202612)]
note = ""
if len(work_2026) >= 25:
    work_final = work_2026
    note = "Muestra: publicaciones de 2026."
else:
    print(f"En 2026 quedan solo {len(work_2026)}; amplío a 2024-2026 para tener señal estable.")
    work_recent = work[work["pub_yyyymm"].between(202401, 202612)]
    if len(work_recent) >= 25:
        work_final = work_recent
        note = "Muestra: publicaciones 2024-2026 (en 2026 no había suficientes comparables)."
    else:
        work_final = work
        note = f"Muestra: todo el histórico ({len(work)} filas, sin filtro temporal)."

work_final = work_final.copy()
work_final["price"] = pd.to_numeric(work_final["price"], errors="coerce")
work_final = work_final[work_final["price"].notna() & (work_final["price"] > 0)]
work_final["usd_m2"] = work_final["price"] / work_final["_surface_m2_est"]

n = len(work_final)
q25 = float(work_final["price"].quantile(0.25))
med = float(work_final["price"].median())
q75 = float(work_final["price"].quantile(0.75))
usd_m2_med = float(work_final["usd_m2"].median())

publish_artifact(
    "metrics",
    title="Rango justo (USD) - 3 amb, 50-80 m2 en San Cristobal/Boedo/Parque Patricios",
    data={
        "Buen trato (q25)": round(q25, 0),
        "Mediana": round(med, 0),
        "Caro (q75)": round(q75, 0),
        "USD/m2 mediano": round(usd_m2_med, 0),
        "Comparables (n)": n,
    },
    caption=note,
)

sample = (
    work_final[["location_3", "_rooms_est", "_surface_m2_est", "price", "usd_m2", "pub_yyyymm"]]
    .sort_values("price")
    .head(20)
    .round(0)
    .rename(columns={"_rooms_est": "ambientes", "_surface_m2_est": "m2"})
)
publish_artifact(
    "table",
    title="Muestra de comparables (20 más baratos)",
    data=sample,
    caption=f"Total comparables: {n}.",
)

try:
    fig = px.histogram(work_final, x="price", nbins=30,
                       title="Distribución de precios de comparables (USD)")
    publish_artifact("chart", title="Distribución de precios", figure=fig)
except Exception as e:
    print("histograma falló:", e)

print("Resumen:", {"q25": q25, "median": med, "q75": q75, "usd_m2_median": usd_m2_med, "n": n})
"""

t0 = time.time()
result = execute_analysis(
    ANALYSIS_CODE,
    {"train_filtered": train, "test": test},
    timeout_s=45.0,
)
elapsed = time.time() - t0
print(f"  ok={result.ok} en {result.elapsed_s:.2f}s (wall {elapsed:.2f}s)")

if not result.ok:
    print("  ERROR:", result.error)
    print(result.traceback[-1500:])
    sys.exit(1)

print("\n  STDOUT:")
for line in (result.stdout or "").strip().splitlines():
    print("   ", line)

print(f"\n  ARTIFACTS publicados: {len(result.artifacts)}")
for i, art in enumerate(result.artifacts):
    head = f"   [{i}] kind={art['kind']} title={art['title']!r}"
    if art["kind"] == "metrics":
        head += f" items={len(art.get('items', []))}"
        print(head)
        for it in art.get("items", []):
            try:
                print(f"        - {it['label']}: {it['value']}")
            except UnicodeEncodeError:
                print(f"        - {it['label'].encode('ascii', 'ignore').decode()}: {it['value']}")
        continue
    elif art["kind"] == "table":
        head += f" rows={len(art.get('data', []))}"
    elif art["kind"] == "chart":
        head += f" figure={art.get('figure') is not None}"
    print(head)

# Aceptación
assert result.ok, "ejecución falló"
assert result.elapsed_s < 40, f"tardo demasiado: {result.elapsed_s:.1f}s"
metrics_art = next(a for a in result.artifacts if a["kind"] == "metrics")
n_item = next(it for it in metrics_art["items"] if "Comparables" in it["label"])
assert int(n_item["value"]) >= 25, f"pocos comparables: {n_item['value']}"
print(f"\nOK -> {n_item['value']} comparables, ejecutado en {result.elapsed_s:.2f}s.")
print("=== OK ===")
