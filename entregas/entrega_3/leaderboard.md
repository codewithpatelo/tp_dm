# Leaderboard local — entrega_3

| nombre | fecha | RMSE CV5 (mean ± std) | RMSE holdout temporal | RMSE holdout | RMSE Kaggle | hotdeck % | descripción | csv md5 |
|---|---|---:|---:|---:|---:|---:|---|---|
| v1 | 2026-04-30 18:58 | 116,669.79 ± 2,521 | 117,235.85 | 118866.97 | **93146.749** | 7.8% | E2 pipeline exact + CV5 multiseed (3 seeds x 5 folds) + holdout temporal | `42f5b5ee` |
| v2 | 2026-04-30 21:41 | 112,776.86 ± 2,173 | 111,536.16 | 116706.86 | **91399.286** | 7.8% | Parseo dual (features+description fallback): rescata m2/dormitorios/banos + nueva feature rooms | `f2ddf22f` |
| v3 | 2026-05-01 09:19 | 112,884.00 ± 2,185 | 111,555.58 | 116821.38 | _pendiente_ | 7.8% | Log-transforms de variables sesgadas: log1p(m2/dormitorios/banos/len_desc/n_features) | `5a67e41c` |
| v4 | 2026-05-01 10:44 | 112,300.31 ± 2,191 | 110,491.77 | 116114.84 | **91594.919** | 7.8% | Barrio cleanup: cascade HotDeck(desc) -> KNN(lat/lon k=5) -> desconocido; barrios escasos<=10 -> barrio_raro | `89e09083` |
| v5 | 2026-05-01 11:55 | 112,441.41 ± 2,138 | 111,293.00 | 116949.35 | _pendiente_ | 7.8% | Distancias a top-10 barrios + m2_per_room/bano ratios + BallTree density k=10 | `fff18e84` |
| v8 | 2026-05-01 16:50 | 111,160.24 ± 3,375 | 102,111.96 | 115442.10 | **89877.580** | 7.8% | TF-IDF + TruncatedSVD (20 componentes) sobre description — tecnica Clase 08; agrega informacion textual nueva no capturada por features actuales | `899b0abe` |
| v11 | 2026-05-02 11:48 | 106,839.97 ± 2,765 | 101,041.35 | 110806.45 | **90613.645** | 7.8% | Scores temáticos de amenities (Clase 07) + SelectKBest(mutual_info, k=10) sobre amenities (Clase 08) + TF-IDF+SVD v8. USE_LOG_TARGET=False. Base: v8. | `684980b1` |
