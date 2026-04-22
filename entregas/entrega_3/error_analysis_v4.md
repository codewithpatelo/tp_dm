# Análisis de errores del campeón v4 (E2) sobre holdout temporal
Reproducción exacta del pipeline v4 (filtros, parsing, IsolationForest, imputación por mediana, Hot Deck por description normalizada, RF(500, 50)) con **split temporal 80/20** por `publication_date`. El Hot Deck del análisis se construye **sólo con el 80 % train del split** (no con todo el train original) para evitar leakage en la evaluación del holdout.

**Setup**: 7,695 filas en el holdout temporal.

**RMSE holdout temporal** (sólo modelo): **157,717**.
**RMSE holdout temporal** (modelo + Hot Deck): **157,925**.

## 1. Errores por barrio
Comparación crítica para validar **v6 (limpieza de barrio)**: ¿el modelo se equivoca más en filas con etiqueta basura ("Ciudad Autónoma de Buenos Aires", etc.) que en barrios limpios?

### 1.1 Limpio vs basura
| barrio_basura | n_obs | mean_signed | mean_abs | rmse_grupo | contrib_rmse_pct |
|---|---|---|---|---|---|
| limpio | 5,622 | -15,235 | 96,381 | 171,197 | 85.86 |
| basura | 2,073 | -980 | 54,061 | 114,427 | 14.14 |

### 1.2 Top 15 barrios por contribución al RMSE total
| barrio | n_obs | mean_signed | mean_abs | rmse_grupo | contrib_rmse_pct |
|---|---|---|---|---|---|
| desconocido | 5,067 | -14,408 | 93,440 | 164,276 | 71.25 |
| Ciudad Autónoma de Buenos Aires | 2,073 | -980 | 54,061 | 114,427 | 14.14 |
| Palermo Chico | 42 | 58,220 | 331,728 | 531,297 | 6.18 |
| Las Cañitas | 47 | -51,503 | 246,037 | 395,988 | 3.84 |
| Palermo Hollywood | 133 | -19,561 | 75,853 | 116,915 | 0.95 |
| Caballito Sur | 45 | -33,679 | 121,943 | 170,368 | 0.68 |
| Belgrano C | 26 | -113,473 | 143,080 | 215,954 | 0.63 |
| Palermo Soho | 63 | -9,045 | 99,898 | 137,521 | 0.62 |
| Belgrano Chico | 11 | 48,677 | 239,607 | 275,397 | 0.43 |
| Botánico | 16 | -63,785 | 158,547 | 196,007 | 0.32 |
| Palermo Nuevo | 8 | 73,009 | 206,454 | 276,039 | 0.32 |
| Belgrano R | 38 | -13,382 | 55,996 | 80,709 | 0.13 |
| Caballito Norte | 37 | -22,824 | 58,001 | 82,885 | 0.13 |
| Almagro Norte | 23 | -48,932 | 65,669 | 85,528 | 0.09 |
| Flores Sur | 5 | -37,991 | 126,792 | 137,541 | 0.05 |

## 2. Errores por property_type
| property_type | n_obs | mean_signed | mean_abs | rmse_grupo | contrib_rmse_pct |
|---|---|---|---|---|---|
| departamento | 6,793 | -13,002 | 85,505 | 160,456 | 91.13 |
| casa | 402 | -1,344 | 118,178 | 188,451 | 7.44 |
| ph | 396 | 4,580 | 59,356 | 78,973 | 1.29 |
| cochera | 104 | -6,094 | 19,942 | 51,548 | 0.14 |

## 3. Errores por decil de `price` real
`mean_signed` positivo = el modelo **subestima** el precio. Si el sesgo cambia de signo entre deciles bajos y altos, el RF está pegado al promedio y mete sesgo regressivo.
| decil_price | n_obs | mean_signed | mean_abs | rmse_grupo | contrib_rmse_pct | price_p50 |
|---|---|---|---|---|---|---|
| 0 | 785 | -51,576 | 52,423 | 87,291 | 3.12 | 59,000 |
| 1 | 796 | -56,254 | 56,984 | 87,206 | 3.15 | 80,000 |
| 2 | 728 | -54,616 | 57,771 | 91,056 | 3.15 | 98,000 |
| 3 | 785 | -43,618 | 50,743 | 96,567 | 3.81 | 118,000 |
| 4 | 761 | -33,788 | 48,060 | 80,466 | 2.57 | 137,000 |
| 5 | 784 | -29,201 | 53,347 | 104,619 | 4.47 | 160,000 |
| 6 | 747 | -23,686 | 64,062 | 101,653 | 4.02 | 193,000 |
| 7 | 801 | -8,866 | 82,900 | 126,102 | 6.64 | 245,000 |
| 8 | 738 | 13,097 | 119,095 | 158,648 | 9.68 | 350,000 |
| 9 | 770 | 175,738 | 266,198 | 384,749 | 59.39 | 650,000 |

## 4. Errores por decil de `m2`
| decil_m2 | n_obs | mean_signed | mean_abs | rmse_grupo | contrib_rmse_pct | m2_p50 |
|---|---|---|---|---|---|---|
| 0 | 772 | -2,178 | 17,527 | 31,210 | 0.39 | 39 |
| 1 | 6,018 | -14,153 | 94,876 | 169,403 | 89.99 | 56 |
| 2 | 144 | -8,060 | 28,081 | 38,251 | 0.11 | 60 |
| 3 | 761 | 437 | 85,914 | 154,867 | 9.51 | 110 |

## 5. Errores por año de publicación
Valida la decisión **v3** de sacar `pub_year`/`pub_month`: si los años más recientes (2025-2026) tienen RMSE muy distinto a 2022-2024, el lookup año→precio que el RF aprende no generaliza.
| pub_yyyymm | n_obs | mean_signed | mean_abs | rmse_grupo | contrib_rmse_pct |
|---|---|---|---|---|---|
| 202,601 | 5,622 | -15,235 | 96,381 | 171,197 | 85.86 |
| 202,606 | 2,073 | -980 | 54,061 | 114,427 | 14.14 |

## 6. Errores en filas con m2 imputado vs observado
Si el RMSE en `m2_imputado` es mucho mayor, la imputación por mediana global está rompiendo predicciones (motiva imputación más sofisticada).
| m2_was_na_label | n_obs | mean_signed | mean_abs | rmse_grupo | contrib_rmse_pct |
|---|---|---|---|---|---|
| m2_imputado | 6,009 | -14,143 | 94,980 | 169,526 | 89.98 |
| m2_observado | 1,686 | -1,601 | 49,340 | 106,777 | 10.02 |

## 7. Errores en filas con lat imputado vs observado
Mismo análisis para `lat`. El 63 % del train tiene lat NaN antes de imputar; si el RMSE es muy distinto entre los dos grupos, la imputación por mediana global está rompiendo el feature geográfico.
| lat_was_na_label | n_obs | mean_signed | mean_abs | rmse_grupo | contrib_rmse_pct |
|---|---|---|---|---|---|
| lat_observado | 7,695 | -11,395 | 84,980 | 157,925 | 100 |

## 8. Errores cerca de centros nombrables
Para cada centro del Análisis 7.5 del EDA, RMSE en filas donde `location_3` contiene ese nombre. **Valida v8** (`dist_a_<centro>`): si los centros tienen RMSE alto, vale invertir esfuerzo en derivar distancias.
_(vacío)_

## 9. Errores por presencia de feature binaria
Para cada `f_*`, RMSE cuando vale 1 vs cuando vale 0. `delta_rmse_1_menos_0` positivo = el modelo se equivoca MÁS cuando la feature está presente (= la propiedad es atípica respecto a su grupo).
| feature | rmse_si_0 | rmse_si_1 | delta_rmse_1_menos_0 |
|---|---|---|---|
| f_cochera | 109,476 | 294,284 | 184,808 |
| f_gimnasio | 139,504 | 247,728 | 108,224 |
| f_jardin | 153,705 | 235,474 | 81,769 |
| f_pileta | 142,025 | 208,428 | 66,403 |
| f_balcon | 141,877 | 177,285 | 35,408 |
| f_bodega | 157,045 | 186,584 | 29,539 |
| f_parrilla | 153,074 | 172,113 | 19,039 |
| f_garage | 156,974 | 171,234 | 14,260 |
| f_internet | 158,657 | 154,872 | -3,785 |
| f_seguridad | 158,687 | 148,024 | -10,663 |
| f_cuarto_de_servicio | 158,772 | 147,982 | -10,790 |
| f_calefaccion | 159,880 | 135,855 | -24,025 |
| f_aire_acondicionado | 161,666 | 130,791 | -30,875 |
| f_alarma | 158,162 | 125,322 | -32,840 |
| f_gas_natural | 165,751 | 110,548 | -55,203 |
| f_cocina_equipada | 166,929 | 110,561 | -56,368 |

## Conclusiones accionables y reordenamiento de prioridades
- **Barrio basura** (Sección 1.1): contribuye **14.1%** al RMSE total con un **delta de RMSE de -56,770** vs barrios limpios. CONFIRMA prioridad media de v6.
- **Sesgo por decil de price** (Sección 3): mirar la columna `mean_signed`. Si los deciles bajos tienen sesgo positivo (subestimamos) y los altos negativo (sobreestimamos), el RF está haciendo regression to the mean — eso JUSTIFICA `np.log(price)` como target transform (no estaba en el plan, agregar como v9).
- **Sesgo por año** (Sección 5): si 2025-2026 tienen RMSE muy distinto a 2022-2024, refuerza la decisión v3 de sacar `pub_year`/`pub_month`.
- **Imputación de m2 / lat** (Secciones 6-7): si el RMSE de las filas imputadas duplica al de las observadas, la imputación global por mediana es subóptima — agregar al plan: imputación por barrio o KNN sobre features completas.
- **Centros nombrables** (Sección 8): los que tienen RMSE alto + buena `n_obs` son los candidatos más fuertes para v8 (`dist_a_<centro>`). Reordenar la lista del plan v8 según esta contribución al RMSE.
- **Features con delta positivo** (Sección 9): las propiedades con `f_X = 1` que tienen RMSE más alto son atípicas respecto a su grupo — candidatas para feature engineering específico (p.ej. interacciones `f_pileta * m2`).
