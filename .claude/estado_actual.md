# Estado actual del TP — resumen ejecutivo

> Snapshot del progreso del trabajo práctico. **Mantener este archivo al día**
> cuando haya avances (nuevo campeón, entrega cerrada, decisión metodológica
> relevante). Ver también la nota en `[CONTEXT.md](CONTEXT.md)`.

## Rol y meta

Competencia Kaggle de **predicción de precio (USD)** de propiedades, métrica
**RMSE**. Restricción habitual del curso: solo técnicas vistas hasta la clase
previa a cada entrega; modelo base **`RandomForestRegressor`**, con más
libertad en E4 y en la final.

- **Competencia:** [fcen-dm-2026-prediccion-precio-de-propiedades](https://www.kaggle.com/competitions/fcen-dm-2026-prediccion-precio-de-propiedades)

## Benchmarks RMSE (Kaggle)

| Referencia | Valor | Notas |
|---|---:|---|
| Mejor público leaderboard | **62 821.209** | Snapshot 2026-04-20; aspiracional, no es consigna. |
| Campeón propio (v4) | **93 151** | Baseline actual de trabajo. |
| Robot E3 | *pendiente* | Aún no publicado; sin RMSE de referencia hasta entonces. Detalle en `CONTEXT.md`. |

## Entregas (tabla de estado)

| Entrega | Foco | Estado |
|--------|------|--------|
| **E1** | Filtros | Aprobada (mejor Kaggle entorno **166 979**) |
| **E2** | Outliers + faltantes | Aprobada (mejor Kaggle **93 151**) |
| **E3** | Ingeniería de atributos + reducción de dimensionalidad | **En curso** |
| **E4** | Datos no estructurados + APIs + datos geográficos | Pendiente |
| **Final** | Integración completa | Pendiente |

**Nota:** En `CONTEXT.md` la sección *"Estado de las entregas → Entrega 2"*
puede seguir con detalle histórico y lecciones. El **campeón de Kaggle
registrado** sigue siendo **v4** (RMSE **93 151**). El foco operativo declaro
en el repo es **E3**.

## Campeón actual (baseline de trabajo): v4 (E2)

Pipeline en síntesis:

1. Filtros E1 + ampliaciones CABA / `ph` / `cochera`.
2. Parseo de **`features`** (m², dormitorios, baños, amenities).
3. Outliers univariados (winsorizar a NaN) + **`IsolationForest`** multivariado solo en train.
4. Imputación (`SimpleImputer`); **`m2_was_na`**.
5. **Hot Deck** por descripción normalizada.
6. **`factorize`** barrio / tipo.
7. **RF** `n_estimators=500`, `max_depth=50`; predicción + override Hot Deck.

**Métricas de referencia (tabla “Resultados registrados” en CONTEXT):**

| versión | RMSE CV5 (mean ± std) | RMSE holdout temporal | RMSE Kaggle |
|---|---:|---:|---:|
| **v4 (campeón)** | **117 219 ± 2 392** | — | **93 151** |

## Experimentos que marcaron el rumbo

- **v2 / v3:** mejoras locales con **leakage** (encoding / imputación que arrastran señal del precio) → Kaggle peor.
- **v5:** mejora fuerte en CV5 pero **peor en Kaggle** → **distribution shift**; features temporales y booleanos de texto engañosos con KFold aleatorio.
- **v6:** corrida de **diagnóstico** (no submit): holdout temporal + mini-ablaciones; confirma que **`pub_year` / `pub_month`** son el principal problema de v5. Regla: **nuevo FE se valida con vista temporal**, no solo CV aleatorio.

**Política de submit (v6+):** mejora simultánea > **1 500** en `rmse_cv5_mean` (multi-seed) **y** > **1 500** en `rmse_holdout_temporal` respecto al mejor previo de cada una; registrar Kaggle cuando corresponda.

## E3 — foco actual

- Puede ajustarse **`n_estimators`** y **`max_depth`** del RF (el resto según consigna del curso).
- Robot E3: notebook **pendiente**; **RMSE Kaggle del robot** aún **desconocido**
  (tabla y párrafo en `CONTEXT.md` → *Robots de la cátedra*).
- Notebook / artefactos: `entregas/entrega_3/`, `leaderboard.md` local.

## Repositorio más allá del notebook Kaggle

Carpeta **`gui/`** (Streamlit, agente code-first, sandbox): herramienta de exploración; no sustituye por sí sola la entrega en formato notebook/consigna.

## Historial de actualizaciones

| Fecha | Cambio |
|------|--------|
| 2026-04-30 | Creación del archivo; contenido alineado al resumen ejecutivo del repositorio. |
| 2026-04-30 | Benchmarks: leaderboard público **62 821.209**, campeón v4 **93 151**, robot E3 sin RMSE hasta publicación (`CONTEXT.md` + esta tabla). |

Al avanzar el TP: actualizar la tabla de entregas, el campeón, métricas y esta
línea de historial; reflejar lo mismo en las tablas detalladas de `CONTEXT.md`
cuando corresponda.
