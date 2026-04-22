# Mini-ablation de las 8 features de v5

Objetivo: identificar qué bloque de las features que v5 introdujo es responsable del distribution shift que se vio en Kaggle (97 498 vs los 93 151 de v4). Sin submits a Kaggle; sólo CV5 + holdout temporal local.

Baseline v6.A = v4 (sin features de v5). Deltas se computan vs v6.A. **Negativo = mejora; positivo = empeora.**

| Sub-conjunto | Features extra | n_features | CV5 mean ± std | Holdout temporal | Δ CV5 vs v4 | Δ holdout temporal vs v4 |
|---|---|---:|---:|---:|---:|---:|
| v6.A_baseline_v4 | — | 26 | 117,219 ± 2,392 | 122,780 | +0 | +0 |
| v6.B_solo_floor | floor | 27 | 116,752 ± 2,510 | 121,678 | -466 | -1,103 |
| v6.C_solo_temporales | pub_year, pub_month | 28 | 117,266 ± 2,361 | 126,165 | +48 | +3,385 |
| v6.D_solo_booleanos | is_a_estrenar, is_reciclado, has_suite, has_service_room, is_near_subway | 31 | 112,767 ± 1,554 | 122,187 | -4,452 | -593 |

_Generado automáticamente por la celda de mini-ablation de v6._