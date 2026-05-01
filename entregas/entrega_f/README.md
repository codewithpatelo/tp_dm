# entrega_f — workspace inicial

Aún no arrancamos Entrega Final. Esta carpeta contiene material preparatorio
("starter scripts") para ideas que sabemos que vamos a explorar pero que no
encajan en E3 / E4 por restricción técnica del TP.

Ver `.claude/CONTEXT.md` → "Backlog de ideas para entregas futuras" para la
lista completa con justificación.

## Contenido

### `drift_correct.py` — Corrección de data drifting

Biblioteca de **7 métodos de corrección** de data drifting, port a Python +
regresión del workflow visto en la materia hermana (Laboratorio de
Implementación II — UBA Exactas Maestría DM, script `z1401_DR_corregir_drifting.r`).

Ataca el problema concreto del dataset: el train cubre `oct-2021` a `jun-2026`
con fuerte inflación + dolarización heterogénea; sin normalización temporal,
las variables monetarias (`price`) no son comparables entre 2022 y 2026. El
`error_analysis_v4.md` de E3 confirmó que ~85 % del RMSE se concentra en
segmentos cuya principal característica común es la separación temporal
respecto al train.

Métodos disponibles:

| metodo | requiere YAML | status |
|---|---|---|
| `rank_simple` | no | ✅ funcional |
| `rank_cero_fijo` | no | ✅ funcional |
| `estandarizar` | no | ✅ funcional |
| `deflacion` | sí (IPC) | ⏳ requiere completar serie |
| `dolar_oficial` | sí (FX) | ⏳ requiere completar serie |
| `dolar_blue` | sí (FX) | ⏳ requiere completar serie |
| `uva` | sí (UVA) | ⏳ requiere completar serie |

Uso:

```bash
# Sin datos externos — corre ya
python entregas/entrega_f/drift_correct.py --metodo rank_simple --cols price

# Con datos macro — completar el YAML primero
python entregas/entrega_f/drift_correct.py --metodo dolar_blue --cols price \
    --indices entregas/entrega_f/indices_macro_arg.yml
```

Output: `dataset_corregido_<metodo>.train.parquet`,
`dataset_corregido_<metodo>.test.parquet` y `<prefix>.report.md` con la
mediana del campo corregido por período.

### `indices_macro_arg.yml` — Stub de tabla de índices macro

Estructura esperada para los métodos `deflacion`, `dolar_oficial`,
`dolar_blue` y `uva`. Está vacío de datos reales — sólo muestra el formato
con valores placeholder. **Antes de usar en EF: completar con las series
reales hasta jun-2026** desde fuentes públicas:

| Serie | Fuente |
|---|---|
| IPC | INDEC, índice de precios al consumidor mensual |
| dolar_oficial | BCRA, serie A3500 |
| dolar_blue | ámbito.com / bluelytics.com.ar (API pública gratuita) |
| UVA | BCRA |

Cargarlas es trabajo de ~30 min: descarga + parseo + extender hasta
jun-2026. En E4 ya habría que tener las series si se aborda la idea
"Normalización temporal con FX paralelo / índice CABA" del backlog.

## Pendientes para EF (referencia rápida)

Resumen de las ideas del backlog que probablemente terminen siendo
experimentos concretos en EF. La lista completa con justificación está en
`.claude/CONTEXT.md`:

- **Anti-drift cuantitativo + visual** (PSI / KS / Wasserstein / JS por feature
  + CDFs por período): es el paso DIAGNÓSTICO previo a `drift_correct.py`.
  Diseño en discusión, no implementado todavía. Workflow ideal:
  `drift_detect.py` → "estas N features driftean" → `drift_correct.py` →
  dataset corregido por método → comparar RMSE en holdout temporal.
- **Embeddings densos de description** (`sentence-transformers` + PCA).
- **RAG semántico de comparables** (kNN sobre embeddings + FAISS).
- **LLM structured extraction** local (Llama / Mistral con Ollama).
- **Visión satelital / StreetView con VLM** pre-entrenado.
- **Geocoding enriquecido** vía OSM Overpass + datos abiertos GCBA / INDEC
  (este probablemente cae en E4, no en EF).
- **Stacking de modelos** (RF + GBM + LightGBM) si EF habilita cambiar de
  modelo.
- **Mixture of experts** (un modelo por cluster del feature space enriquecido)
  — atacaría directo el ~85 % del RMSE concentrado en 2 segmentos.

## Convenciones heredadas

Aplican las mismas reglas que en E1-E3 (ver `.claude/CONTEXT.md`):

- Sin sobreescribir datasets originales.
- Procesamiento train/test en paralelo (estadísticas fittean SOLO en train).
- Si se transforma el target, aplicar la inversa antes de subir a Kaggle.
- Predicciones finales redondeadas a múltiplos de 1 000 USD.
- Experimentos largos con checkpointing (`entregas/_resumable.py`).
