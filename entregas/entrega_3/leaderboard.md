# Leaderboard local — entrega_3

| nombre | fecha | RMSE CV5 (mean ± std) | RMSE holdout temporal | RMSE holdout | RMSE Kaggle | hotdeck % | descripción | csv md5 |
|---|---|---:|---:|---:|---:|---:|---|---|
| v1 | 2026-04-30 18:58 | 116,669.79 ± 2,521 | 117,235.85 | 118866.97 | **93146.749** | 7.8% | E2 pipeline exact + CV5 multiseed (3 seeds x 5 folds) + holdout temporal | `42f5b5ee` |
| v2 | 2026-04-30 21:41 | 112,776.86 ± 2,173 | 111,536.16 | 116706.86 | **91399.286** | 7.8% | Parseo dual (features+description fallback): rescata m2/dormitorios/banos + nueva feature rooms | `f2ddf22f` |
| v3 | 2026-05-01 09:19 | 112,884.00 ± 2,185 | 111,555.58 | 116821.38 | _pendiente_ | 7.8% | Log-transforms de variables sesgadas: log1p(m2/dormitorios/banos/len_desc/n_features) | `5a67e41c` |
