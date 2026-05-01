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
| **Campeón propio (v2, E3)** | **91 399** | Baseline actual de trabajo. |
| Robot E3 | *pendiente* | Aún no publicado; sin RMSE de referencia hasta entonces. |

## Entregas (tabla de estado)

| Entrega | Foco | Estado |
|--------|------|--------|
| **E1** | Filtros | Aprobada (mejor Kaggle **166 979**) |
| **E2** | Outliers + faltantes | Aprobada (mejor Kaggle **93 151**, v4) |
| **E3** | Ingeniería de atributos + reducción de dimensionalidad | **En curso** — campeón parcial v2 (Kaggle **91 399**) |
| **E4** | Datos no estructurados + APIs + datos geográficos | Pendiente |
| **Final** | Integración completa | Pendiente |

## Campeón actual: E3-v2

Pipeline en síntesis:

1. Filtros E1 + ampliaciones CABA / `ph` / `cochera`.
2. Parseo de **`features`** con fallback a **`description`** para m2/dormitorios/baños cuando NaN (**parseo dual**); nueva feature `rooms` (ambientes).
3. Outliers univariados (winsorizar a NaN) + **`IsolationForest`** multivariado solo en train.
4. Imputación (`SimpleImputer` mediana); **`m2_was_na`**.
5. **Hot Deck** por descripción normalizada.
6. **`factorize`** barrio / tipo.
7. **RF** `n_estimators=500`, `max_depth=50`; predicción + override Hot Deck.

**Métricas:**

| versión | RMSE CV5 (mean ± std) | RMSE holdout temporal | RMSE Kaggle |
|---|---:|---:|---:|
| v1 (baseline) | 116 670 ± 2 521 | 117 236 | 93 147 |
| **v2 (campeón)** | **112 777 ± 2 173** | **111 536** | **91 399** |

## E3 — experimentos en curso / pendientes

| orden | versión | contenido | estado |
|---|---|---|---|
| — | v2 | parseo dual + rooms | **campeón** |
| 1 | v3 | log-transforms (log1p m2/dormitorios/baños/len_desc/n_features) | **completado** — NO mejoró (CV5 112 884 vs 112 777; RF invariante a transforms monótonas) |
| 2 | v4 | barrio cleanup (Hot Deck → KNN → desconocido) | **corriendo ahora** |
| 3 | v5 | distancias a centros de referencia (sin API) | pendiente |
| 4 | v6 | log(price) target transform | pendiente |
| 5 | v7 | reducción de dimensionalidad (VarianceThreshold / PCA) | pendiente |

**Política de submit:** mejora simultánea > 1 500 en CV5 multi-seed **y** > 1 500 en holdout temporal vs el mejor previo de cada métrica.

## Clases disponibles (E3)

- Clase 07 — Ingeniería de atributos (`diapos_clase/FCEN MD Clase 07...pdf`, `colabs_clase/Clase_07_...ipynb`)
- Clase 08 — Reducción de dimensionalidad (`diapos_clase/FCEN MD Clase 08...pdf`, `colabs_clase/Clase_08_...ipynb`)

## Historial de actualizaciones

| Fecha | Cambio |
|------|--------|
| 2026-04-30 | Creación del archivo; contenido alineado al resumen ejecutivo del repositorio. |
| 2026-04-30 | Benchmarks: leaderboard público **62 821.209**, campeón v4 E2 **93 151**, robot E3 sin RMSE hasta publicación. |
| 2026-04-30 | E3-v1 completado (Kaggle 93 147); E3-v2 completo (Kaggle **91 399**, nuevo campeón). Clases 07 y 08 disponibles. |
| 2026-05-01 | E3-v3 completado (CV5 112 884 ± 2 185, holdout 111 556) — log-transforms NO mejoran al RF (invariante a transforms monótonas). Aprendizaje: no agregar columnas redundantes en escala log. |
| 2026-05-01 | Timeout nbclient (3600s) en primera corrida de v3; fix: timeout=7200s en run_entrega.py. V4 (barrio cleanup) iniciada. |
