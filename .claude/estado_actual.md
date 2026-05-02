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
| Mejor **privado** leaderboard | **68 908** | Rank 1 — Ariel Rojas. Fuente: private LB filtrado por profe (2026-04-29). |
| Mejor público leaderboard | **62 821.209** | Snapshot 2026-04-20 — era público, probablemente coincide con rank 1 privado. |
| **Nuestro mejor privado (v2, E3)** | **95 419** | Rank 16/70 snapshot 29/04 (solo E2). |
| **Nuestro mejor público (v8, E3)** | **89 877** | Nuevo campeón — submit automático 2026-05-01. |
| Robot E3 (C3PO) | **132 425** | Rank 0 (fuera de competencia) — nuestro baseline a superar está bien por debajo. |
| Gap nuestro privado vs rank 1 | **+26 511** | Diferencia real viene de features/modelo (E4/EF), no de overfitting. RF sin APIs tiene techo ~88-92K en E3. |

## Entregas (tabla de estado)

| Entrega | Foco | Estado |
|--------|------|--------|
| **E1** | Filtros | Aprobada (mejor Kaggle **166 979**) |
| **E2** | Outliers + faltantes | Aprobada (mejor Kaggle **93 151**, v4) |
| **E3** | Ingeniería de atributos + reducción de dimensionalidad | **En curso** — campeón parcial v2 (Kaggle **91 399**) |
| **E4** | Datos no estructurados + APIs + datos geográficos | Pendiente |
| **Final** | Integración completa | Pendiente |

## Campeón actual: E3-v11 (criterio: métricas locales)

Pipeline en síntesis:

1. Filtros E1 + ampliaciones CABA / `ph` / `cochera`.
2. Parseo de **`features`** con fallback a **`description`** para m2/dormitorios/baños cuando NaN (**parseo dual**); nueva feature `rooms` (ambientes).
3. Outliers univariados (winsorizar a NaN) + **`IsolationForest`** multivariado solo en train.
4. Imputación (`SimpleImputer` mediana); **`m2_was_na`**.
5. **Barrio cleanup**: cascade Hot Deck (desc) → KNN(lat/lon, k=5) → "desconocido"; barrios ≤10 obs → "barrio_raro".
6. **Scores temáticos** (Clase 07): 5 scores sumando binarias por tema (lujo, servicios, estacionamiento, espacios extra, seguridad).
7. **SelectKBest** (Clase 08, `mutual_info_regression`, k=10): conserva 10 de 16 amenities binarias; elimina `f_alarma`, `f_bodega`, `f_calefaccion`, `f_cocina_equipada`, `f_gas_natural`, `f_internet`.
8. **TF-IDF + TruncatedSVD** (20 componentes) sobre `description` — fit en train, transform en ambos.
9. **Hot Deck** precio por descripción normalizada.
10. **`factorize`** barrio / tipo.
11. **RF** `n_estimators=500`, `max_depth=50`, `USE_LOG_TARGET=False`; predicción + override Hot Deck.

**Métricas:**

| versión | RMSE CV5 (mean ± std) | RMSE holdout temporal | RMSE Kaggle | nota |
|---|---:|---:|---:|---|
| v1 (baseline) | 116 670 ± 2 521 | 117 236 | 93 147 | |
| v2 | 112 777 ± 2 173 | 111 536 | **91 399** | parseo dual + rooms |
| v3 | 112 884 ± 2 185 | 111 556 | — | log-transforms — NO mejoró |
| v4 | 112 300 ± 2 191 | 110 492 | 91 595 | barrio cleanup |
| v5 | 112 441 ± 2 138 | 111 293 | — | distancias + ratios + density — NO mejoró |
| v6 | 114 512 (parcial 9/15) | — | — | log(price) target — cancelado early stopping |
| v7 | 115 552 (parcial 8/15) | — | — | KNN m2 imputation — cancelado early stopping |
| v8 | 111 160 ± 3 375 | 102 112 | 89 877 | TF-IDF+SVD 20 comp — campeón previo |
| v9 | — | — | — | TF-IDF+SVD 50 comp 10k vocab — abortado timeout |
| v10 | 118 574 (parcial 2/15) | — | — | dedup price tests — cancelado early stopping |
| **v11 (CAMPEÓN)** | **106 840 ± 2 765** | **101 041** | 90 614 | scores temáticos + SelectKBest(k=10) + USE_LOG_TARGET=False. Kaggle +736 vs v8 investigar shift de SelectKBest |

## E3 — experimentos en curso / pendientes

| orden | versión | contenido | estado |
|---|---|---|---|
| — | v2 | parseo dual + rooms | **campeón** |
| 1 | v3 | log-transforms (log1p m2/dormitorios/baños/len_desc/n_features) | **completado** — NO mejoró (CV5 112 884 vs 112 777; RF invariante a transforms monótonas) |
| 2 | v4 | barrio cleanup (Hot Deck → KNN → desconocido) | **campeón local** — CV5 +477, holdout +1044; Kaggle −196 (shift leve) |
| 3 | v5 | distancias dinámicas + m2_per_room ratio + BallTree density | **completado** — NO mejoró (CV5 +141, holdout +801 vs v4). RF ya captura geografía vía lat/lon crudo; distancias a centroides son redundantes. |
| 4 | v6 | log(price) target transform | **cancelado** — media parcial 114 512 (9/15 folds). RF en log-space predice mediana geométrica, no media aritmética → sesgo en RMSE original. |
| 5 | v7 | KNN imputation para m2 usando lat/lon (k=5) | **cancelado** — media parcial 115 552 (8/15 folds, +3 252 vs campeón). KNN introduce ruido: propiedades cercanas no siempre tienen m2 similares (distintos tipos, alturas, antigüedades). Mediana global más robusta. |
| 6 | v8 | TF-IDF + TruncatedSVD sobre description (Clase 08) | **CAMPEÓN** — CV5 111 160 ± 3 375, holdout 102 112, Kaggle 89 877 |
| 7 | v9 | TF-IDF+SVD escalado (50 comp, 10k vocab) | **abortado** — timeout repetido (>6h por fold). TF-IDF con bigrams+10k vocab demasiado lento para el pipeline actual. Pendiente workflow más resiliente. |
| 8 | v10 | Dedup price tests + base v8 (20 SVD) | **cancelado** — media parcial fold 2/15: 118 574 (+7 414 vs campeón). Eliminar duplicados reduce el volumen de train ~10%; el RF pierde más por menos datos que lo que gana por labels más limpias. |
| 9 | **v11** | Scores temáticos (Clase 07) + SelectKBest(mutual_info, k=10) (Clase 08) + USE_LOG_TARGET=False | **CAMPEÓN** — CV5 106 840 ± 2 765, holdout temporal 101 041, Kaggle pendiente |

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
| 2026-05-01 | E3-v6 cancelada por early stopping (fold 9/15, media parcial 114 512 vs campeón 112 300). Aprendizaje: RF entrenado en log(y) minimiza RMSE en log-space (≈ MAPE), predice mediana geométrica al aplicar exp(), no la media aritmética que minimiza RMSE en escala original. La corrección σ²/2 no es sencilla con sklearn RF. Política de early stopping agregada: cancelar si media parcial al fold 10 > campeón. |
| 2026-05-01 | E3-v7 cancelada por early stopping (fold 8/15, media parcial 115 552 vs campeón 112 300). Aprendizaje: KNN(lat/lon) para imputar m2 introduce ruido — propiedades geográficamente cercanas no comparten m2 (distintos tipos, pisos, antigüedad). La mediana global captura mejor la distribución marginal de m2. |
| 2026-05-01 | E3-v8 completada: **NUEVO CAMPEÓN**. CV5 111 160 ± 3 375, holdout temporal 102 112, Kaggle **89 877** (mejor score propio hasta ahora). TF-IDF+TruncatedSVD (20 componentes) sobre description aporta información textual genuinamente nueva. Holdout temporal baja 8 380 puntos — mayor mejora de E3. Informe regenerado automáticamente. |
| 2026-05-01 | Private LB revelado por el profe (snapshot 2026-04-29). Rank 16/70, private 95 419. Gap público/privado: +4 020 (leve overfitting con 14 subs, no alarmante). Gap vs rank 1 (68 908): +26 511 — atribuible a features externas (E4/EF), no a overfitting. Archivo: `fcen-dm-2026-prediccion-precio-de-propiedades-privateleaderboard-2026-04-29T223756 (1).csv` en raíz del repo. |
| 2026-05-02 | v10 (dedup) cancelado fold 2 (media 118 574, +7 414 vs campeón) — RF pierde más por volumen de datos que lo que gana en pureza de labels. |
| 2026-05-02 | v11 completado — CV5 106 840 ± 2 765 (−4 320 vs v8), holdout temporal 101 041 (−1 071), Kaggle 90 614 (+736 vs v8). Distribution shift: mejora local consistente pero Kaggle retrocede levemente. Campeón local = v11; mejor Kaggle = v8 (89 877). |
