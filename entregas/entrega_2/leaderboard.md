# Leaderboard local — entrega_2

Track de cada submission generada con `entregas/run_entrega.py`. La columna
**RMSE Kaggle** se completa con `python entregas/run_entrega.py --record-kaggle
--entrega entrega_2 --nombre <v> --kaggle-rmse <valor>` después de subirla.

**Métricas locales (v6+)**:
- `RMSE CV5 (mean ± std)` — multi-seed (3 seeds × 5 folds = 15 evaluaciones)
  con `n_estimators=500, max_depth=50`. Métrica primaria desde v4; en v6 se
  pasó a multi-seed para estabilizar la varianza intra-KFold.
- `RMSE holdout temporal` — RMSE sobre el último 20 % de las filas con
  `publication_date` conocida ordenadas cronológicamente. Métrica
  secundaria agregada en v6 después de la lección de v5: KFold aleatorio
  no detecta distribution shift entre train y test público (mezcla años
  uniformemente); el split por fecha sí lo aproxima. v1-v5 quedan con
  "—" porque no la calcularon.
- `RMSE holdout` — split único `train_test_split(test_size=0.2,
  random_state=42)`. Se descartó como métrica primaria desde v4 porque
  tiene std ≈ 2 K (mayor que las diferencias entre versiones); queda como
  columna informativa.

**Auto-submit (v6+)**: requiere mejora simultánea > 1 500 puntos en
`RMSE CV5 (mean)` y > 1 500 en `RMSE holdout temporal` respecto al mejor
previo. Si una sola métrica mejora, no submit.

Baseline a superar: **153 060.094**.

| nombre | fecha | RMSE CV5 (mean ± std) | RMSE holdout temporal | RMSE holdout | RMSE Kaggle | hotdeck % | descripción | csv md5 |
|---|---|---:|---:|---:|---:|---:|---|---|
| v1 | 2026-04-20 17:43 | — | — | 118866.97 | **93167.324** | 6.9% | Filtro CABA ampliado + parseo de features (m2/dorm/baños/amenities) + outliers IQR + IsolationForest(0.01) + imputación mediana/moda + Hot Deck por description | `pre-orq` |
| v2 | 2026-04-20 18:17 | — | — | 116922.89 | **96891.059** | 6.9% | H2: precio_mediano_barrio y precio_mediano_barrio_tipo desde train | `8aef4216` |
| v3 | 2026-04-21 11:16 | — | — | 117080.27 | **97423.121** | 6.9% | H3: imputacion lat/lon por mediana de BARRIO (no global). Justifica: lat+lon = 18% de feat_importances en v2; mediana global aplasta señal geográfica. | `2ba4d1d7` |
| v4 | 2026-04-21 12:12 | 117,218.50 ± 2,392 | — | 118866.97 | **93151.157** | 7.8% | v1 limpio (sin H3, sin precio_mediano_barrio*) + Hot Deck con descripcion normalizada (lower/sin acentos/sin puntuacion/min 20 chars) + CV5 como metrica primaria | `fb37bcaf` |
| v5 | 2026-04-21 14:23 | 112,340.70 ± 1,659 | — | 113951.86 | **97498.065** ✗ | 7.8% | v5: floor + calidad texto (a_estrenar, reciclado, suite, cuarto servicio, subte) + temporales (pub_year, pub_month) | `465a63e6` |
| v6 | 2026-04-21 15:54 | 112,303.62 ± 2,794 | 122794.86 | 113951.86 | — | 7.8% | v6: validacion multi-seed CV5 + holdout temporal + mini-ablation features v5 (sin cambios al pipeline; pipeline = v5) | `b7fd328d` |
