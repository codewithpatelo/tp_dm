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

## Campeón actual: E3-v4 (local)

Pipeline en síntesis:

1. Filtros E1 + ampliaciones CABA / `ph` / `cochera`.
2. Parseo de **`features`** con fallback a **`description`** para m2/dormitorios/baños cuando NaN (**parseo dual**); nueva feature `rooms` (ambientes).
3. Outliers univariados (winsorizar a NaN) + **`IsolationForest`** multivariado solo en train.
4. Imputación (`SimpleImputer` mediana); **`m2_was_na`**.
5. **Barrio cleanup**: cascade Hot Deck (desc) → KNN(lat/lon, k=5) → "desconocido"; barrios ≤10 obs → "barrio_raro".
6. **Hot Deck** precio por descripción normalizada.
7. **`factorize`** barrio / tipo.
8. **RF** `n_estimators=500`, `max_depth=50`; predicción + override Hot Deck.

**Métricas:**

| versión | RMSE CV5 (mean ± std) | RMSE holdout temporal | RMSE Kaggle | nota |
|---|---:|---:|---:|---|
| v1 (baseline) | 116 670 ± 2 521 | 117 236 | 93 147 | |
| v2 | 112 777 ± 2 173 | 111 536 | **91 399** | mejor Kaggle |
| v3 | 112 884 ± 2 185 | 111 556 | — | log-transforms — NO mejoró |
| **v4 (campeón local)** | **112 300 ± 2 191** | **110 492** | 91 595 | barrio cleanup; Kaggle −196 vs v2 (shift leve) |
| v5 | 112 441 ± 2 138 | 111 293 | — | distancias + ratios + density — NO mejoró (redundante con lat/lon) |
| v6 | 114 512 (parcial 9/15) | — | — | log(price) target — cancelado por early stopping (peor que campeón al fold 9) |

## E3 — experimentos en curso / pendientes

| orden | versión | contenido | estado |
|---|---|---|---|
| — | v2 | parseo dual + rooms | **campeón** |
| 1 | v3 | log-transforms (log1p m2/dormitorios/baños/len_desc/n_features) | **completado** — NO mejoró (CV5 112 884 vs 112 777; RF invariante a transforms monótonas) |
| 2 | v4 | barrio cleanup (Hot Deck → KNN → desconocido) | **campeón local** — CV5 +477, holdout +1044; Kaggle −196 (shift leve) |
| 3 | v5 | distancias dinámicas + m2_per_room ratio + BallTree density | **completado** — NO mejoró (CV5 +141, holdout +801 vs v4). RF ya captura geografía vía lat/lon crudo; distancias a centroides son redundantes. |
| 4 | v6 | log(price) target transform | **en curso** — mayor potencial esperado (decil 9 = 59.4% del RMSE) |
| 5 | v7 | SelectKBest(mutual_info) + luxury/thematic scores + TF-IDF+SVD | pendiente |

**Política de submit:** el **campeón se define por las métricas locales** — mejora simultánea en CV5 multi-seed **y** holdout temporal vs el campeón anterior, sin umbral mínimo. Qualquier mejora consistente en ambas → nuevo campeón + submit a Kaggle. El score Kaggle es referencia para detectar distribution shift, no el árbitro.

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
| 2026-05-01 | E3-v4 completado: CV5 112 300 ± 2 191, holdout 110 492, Kaggle 91 595. Nuevo campeón local. Kaggle levemente peor que v2 (shift leve en barrio cleanup). Política de submit actualizada: umbral 0 (cualquier mejora local consistente). |
| 2026-05-01 | E3-v5 completado: CV5 112 441 ± 2 138, holdout_temporal 111 293. NO campeón — ambas métricas peores que v4. Distancias a centroides redundantes con lat/lon; m2_per_room y density_k10 no aportan señal nueva al RF. |
| 2026-05-01 | E3-v6 iniciada: log(price) target transform (fit en log(y), predict en exp(ŷ)). Mayor potencial esperado por subestimación sistemática del decil 9. |
