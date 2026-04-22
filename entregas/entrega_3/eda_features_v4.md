# EDA preparatorio para Entrega 3 (sobre el pipeline de v4)

Este EDA reproduce los filtros + parsing de la corrida campeona de Entrega 2 (v4) y mide cuatro cosas distintas que necesito antes de diseñar los experimentos de E3 anclados en Clase 7. **No** corre el modelo: las decisiones de transformación / discretización se toman sobre las distribuciones crudas, antes de outliers / imputación.

- Filtro v4: train post-filtro = 123,102 filas, test = 13,471.


## 1. Skewness y percentiles de las numéricas continuas
Identifica candidatas a `np.log` / `np.sqrt` (transformaciones de Clase 7). La recomendación es heurística: skew > 2 con valores positivos → log; skew 1-2 con no-negativos → log1p / sqrt; skew leve o cuasi-simétrica → nada.

| columna | n | p01 | p50 | p99 | max | skew | reco |
|---|---|---|---|---|---|---|---|
| price | 123,102 | 12,500 | 139,000 | 1,300,000 | 3,000,000 | 4.71 | np.log (skew alto, valores positivos) |
| m2 | 90,764 | 10 | 56 | 351 | 999 | 3.71 | np.log1p o np.sqrt (skew moderado) |
| lat | 45,597 | -34.6644 | -34.597 | -34.5414 | 40.0704 | 53.87 | evaluar (skew leve o valores no-positivos) |
| lon | 45,593 | -58.5191 | -58.436 | -58.37 | 58.3953 | 47.84 | evaluar (skew leve o valores no-positivos) |
| n_dormitorios | 83,810 | 1 | 2 | 5 | 51 | 5.13 | np.log (skew alto, valores positivos) |
| n_banos | 96,497 | 1 | 1 | 4 | 140,000 | 310.63 | np.log (skew alto, valores positivos) |
| len_descripcion | 123,102 | 102 | 1,411 | 5,472.98 | 21,299 | 1.44 | np.log1p o np.sqrt (skew moderado) |
| n_features | 123,102 | 0 | 15 | 33 | 121 | 1.07 | np.log1p o np.sqrt (skew moderado) |

## 2. Distribución temporal train vs test (shift de v6)
Cuantifica el distribution shift cronológico que rompió el holdout temporal en v6 cuando se metieron `pub_year`/`pub_month`. Si la distribución por año difiere fuerte entre train y test, usar el año como nivel del RF es un lookup que no generaliza.

### Por año
| publication_date | train_n | test_n | train_pct | test_pct | delta_pct |
|---|---|---|---|---|---|
| 2,022 | 3,687 | 536 | 22.59 | 23.51 | 0.92 |
| 2,023 | 781 | 72 | 4.78 | 3.16 | -1.62 |
| 2,024 | 423 | 73 | 2.59 | 3.2 | 0.61 |
| 2,025 | 9,343 | 1,315 | 57.23 | 57.68 | 0.45 |
| 2,026 | 2,090 | 284 | 12.8 | 12.46 | -0.34 |

### Por mes calendario (estacionalidad pura, agregando años)
| mes | train_n | test_n | train_pct | test_pct | delta_pct |
|---|---|---|---|---|---|
| 1 | 2,372 | 338 | 14.53 | 14.82 | 0.29 |
| 2 | 364 | 41 | 2.23 | 1.8 | -0.43 |
| 3 | 193 | 16 | 1.18 | 0.7 | -0.48 |
| 4 | 225 | 27 | 1.38 | 1.18 | -0.2 |
| 5 | 45 | 5 | 0.28 | 0.22 | -0.06 |
| 6 | 303 | 21 | 1.86 | 0.92 | -0.94 |
| 7 | 666 | 83 | 4.08 | 3.64 | -0.44 |
| 8 | 494 | 49 | 3.03 | 2.15 | -0.88 |
| 9 | 653 | 93 | 4 | 4.08 | 0.08 |
| 10 | 1,320 | 184 | 8.09 | 8.07 | -0.02 |
| 11 | 5,793 | 862 | 35.49 | 37.81 | 2.32 |
| 12 | 3,896 | 561 | 23.87 | 24.61 | 0.74 |

## 3. Information Gain de las features binarias contra `pd.qcut(price, 10)`
Implementación manual del cálculo de Clase 7 (entropía → IG). Mide cuánta información da cada `f_*` para predecir el decil de precio. Las de IG bajo son candidatas a podar; las de IG alto son las que más pesa el RF.

| feature | tasa_1_train_pct | IG | IG_pct_Hy |
|---|---|---|---|
| f_cochera | 18.03 | 0.0927 | 2.79 |
| f_balcon | 39.05 | 0.0501 | 1.51 |
| f_pileta | 6.8 | 0.0259 | 0.78 |
| f_gimnasio | 4.97 | 0.023 | 0.69 |
| f_parrilla | 8.25 | 0.0227 | 0.68 |
| f_cuarto_de_servicio | 4.68 | 0.0143 | 0.43 |
| f_jardin | 2.57 | 0.0129 | 0.39 |
| f_seguridad | 3.8 | 0.0121 | 0.36 |
| f_cocina_equipada | 9.82 | 0.0117 | 0.35 |
| f_garage | 5.22 | 0.0116 | 0.35 |
| f_bodega | 1.58 | 0.0099 | 0.3 |
| f_calefaccion | 4.22 | 0.0063 | 0.19 |
| f_aire_acondicionado | 5.86 | 0.0056 | 0.17 |
| f_internet | 6.63 | 0.0047 | 0.14 |
| f_gas_natural | 7.02 | 0.0038 | 0.11 |
| f_alarma | 0.33 | 0.0013 | 0.04 |

### 3b. Covariate shift de las binarias entre train y test
Tasa de 1s en train vs test, ordenadas por |delta|. Si una `f_*` tiene tasa muy distinta entre train y test, su contribución no generaliza aunque tenga IG alto.

| feature | tasa_train | tasa_test | delta |
|---|---|---|---|
| f_cocina_equipada | 9.82 | 14.64 | 4.82 |
| f_cochera | 18.03 | 21.16 | 3.12 |
| f_aire_acondicionado | 5.86 | 8.54 | 2.68 |
| f_gas_natural | 7.02 | 9.55 | 2.54 |
| f_garage | 5.22 | 7.35 | 2.13 |
| f_cuarto_de_servicio | 4.68 | 6.54 | 1.86 |
| f_calefaccion | 4.22 | 6.06 | 1.85 |
| f_parrilla | 8.25 | 9.56 | 1.31 |
| f_pileta | 6.8 | 7.59 | 0.79 |
| f_bodega | 1.58 | 2.26 | 0.67 |
| f_seguridad | 3.8 | 4.44 | 0.64 |
| f_gimnasio | 4.97 | 4.51 | -0.46 |
| f_alarma | 0.33 | 0.59 | 0.25 |
| f_internet | 6.63 | 6.81 | 0.18 |
| f_balcon | 39.05 | 39.17 | 0.12 |
| f_jardin | 2.57 | 2.52 | -0.05 |

## 4. Distribución del tamaño de los barrios
Identifica barrios chicos (< 30 obs) — candidatos a colapsar via discretización supervisada. Reduce dimensionalidad efectiva de `barrio_id` sin perder señal.

| tamano_barrio | n_barrios | pct_barrios | filas_cubiertas | pct_filas |
|---|---|---|---|---|
| 1-10 | 14 | 14.29 | 57 | 0.05 |
| 11-30 | 11 | 11.22 | 218 | 0.18 |
| 31-100 | 9 | 9.18 | 534 | 0.43 |
| 101-500 | 23 | 23.47 | 5,690 | 4.62 |
| 501-1000 | 12 | 12.24 | 7,950 | 6.46 |
| 1000+ | 29 | 29.59 | 108,653 | 88.26 |

**Resumen barrios**: 98 barrios totales, de los cuales 25 tienen < 30 obs (cubren 275 filas = 0.22% del train post-filtro).


## 5. Cobertura del parseo dual `features` + `description` (insumo Robot 2 C3PO)
Para cada atributo numérico clave (`m2`, `n_dormitorios`, `n_banos`), cuántas filas vienen NaN en `features` pero el dato sí está en `description`. Si el rescate es > 1% en al menos un atributo, vale incorporar el parseo dual al pipeline; si no, el costo de complejidad no se justifica.

| atributo | n_total | nan_en_features | pct_nan_features | rescatadas_por_desc | pct_rescatadas |
|---|---|---|---|---|---|
| m2 | 123,102 | 40,537 | 32.93 | 10,906 | 8.86 |
| n_dormitorios | 123,102 | 39,292 | 31.92 | 3,775 | 3.07 |
| n_banos | 123,102 | 26,605 | 21.61 | 2,293 | 1.86 |

**Veredicto**: ENTRA en v2 (parseo dual rescata > 1% en al menos un atributo) (máximo de rescate observado: 8.86%).


## 6. Cobertura `surface_total_m2` vs `surface_covered_m2` (insumo Robot 2 C3PO)
Cuántas filas traen los dos valores en el texto y cuán distintos son. El robot deriva `pct_surface_covered = covered / total`. Vale agregarla sólo si la cobertura conjunta es > 5% **y** los dos valores difieren > 10% en una porción significativa de las filas (si fueran casi siempre iguales, la feature derivada sería ruido constante = 1).

- Filas totales: 123,102
- Con `surface_total_m2` extraíble: 91,424 (74.27%)
- Con `surface_covered_m2` extraíble: 64,537 (52.43%)
- Con ambos valores extraídos: 64,505 (52.4%)
- Mediana de `covered / total` (sólo donde hay ambos): 1.0
- Filas donde `covered` < 90% de `total`: 1,129 (1.75% de las que tienen ambos)


**Veredicto**: DESCARTAR (cobertura 52.4% o difieren sólo en 1.75% — no agrega señal).


## 7. Geografía interna sin APIs (Notas 3 y 4 del user)
Tres preguntas en una: (a) ¿es `location_3` realmente sub-barrio o es lo mismo que `location_2`? (b) ¿cuántas filas tienen barrio mal etiquetado y vale el cleanup vía Hot Deck/KNN? (c) ¿hay algún centro comercial / sub-barrio céntrico nombrado en `location_3` que permita derivar `dist_a_microcentro` sin APIs externas?


### 7.1 Cardinalidad de location_1..4 (¿hay sub-barrios?)
| nivel | n_unicos_train | n_unicos_test | pct_nan_train | pct_nan_test |
|---|---|---|---|---|
| location_1 | 3 | 2 | 0 | 0 |
| location_2 | 79 | 40 | 0 | 0 |
| location_3 | 97 | 61 | 11.44 | 0.4 |
| location_4 | 34 | 34 | 93.39 | 94.82 |

Si `n_unicos` de `location_2` y `location_3` son distintos, tenemos dos granularidades geográficas reales y vale tener ambas como features paralelas en el modelo.


### 7.2 Barrios mal etiquetados / NaN en location_2 y location_3
| nivel | split | n_filas | n_basura | pct_basura | n_nan | pct_nan | n_total_a_imputar | pct_a_imputar |
|---|---|---|---|---|---|---|---|---|
| location_2 | train | 123,102 | 99,261 | 80.63 | 4 | 0 | 99,265 | 80.64 |
| location_2 | test | 13,471 | 12,805 | 95.06 | 0 | 0 | 12,805 | 95.06 |
| location_3 | train | 123,102 | 25,128 | 20.41 | 14,080 | 11.44 | 39,208 | 31.85 |
| location_3 | test | 13,471 | 2,865 | 21.27 | 54 | 0.4 | 2,919 | 21.67 |

**Veredicto**: PRIORIDAD ALTA: hasta 80.64% del train y 95.06% del test tienen barrio mal etiquetado o NaN. Vale la cascada Hot Deck por description → KNN sobre lat/lon (Nota 3).


### 7.3 Intersección train↔test (¿el vocabulario de train cubre el test?)
| nivel | n_train | n_test | n_comunes | pct_test_cubierto | n_solo_train | n_solo_test |
|---|---|---|---|---|---|---|
| location_2 | 79 | 40 | 40 | 100 | 39 | 0 |
| location_3 | 97 | 61 | 61 | 100 | 36 | 0 |

Si `pct_test_cubierto` es alto (> 95%), podemos usar el train como diccionario para imputar etiquetas raras del test. Si es bajo, el test trae barrios nuevos y necesitamos KNN sobre lat/lon para esos casos.


### 7.4 Top barrios por frecuencia (sanity check)

**train · location_2**

| location_2 | n |
|---|---|
| Capital Federal | 74,226 |
| Buenos Aires | 25,035 |
| Palermo | 2,538 |
| Caballito | 1,636 |
| General Pueyrredon | 1,598 |
| Belgrano | 1,447 |
| Recoleta | 1,310 |
| Almagro | 1,152 |
| Balvanera | 950 |
| Villa Crespo | 948 |

**train · location_3**

| location_3 | n |
|---|---|
| Ciudad Autónoma de Buenos Aires | 25,034 |
| Palermo | 9,629 |
| Belgrano | 6,041 |
| Caballito | 4,991 |
| Ciudad de Buenos Aires | 4,318 |
| Recoleta | 3,877 |
| Centro | 3,726 |
| Villa Urquiza | 3,218 |
| Nuñez | 3,206 |
| Almagro | 3,118 |

**test · location_2**

| location_2 | n |
|---|---|
| Capital Federal | 9,947 |
| Buenos Aires | 2,858 |
| Palermo | 132 |
| Belgrano | 85 |
| Recoleta | 78 |
| Caballito | 71 |
| Flores | 42 |
| Almagro | 38 |
| Villa Crespo | 34 |
| Villa Urquiza | 24 |

**test · location_3**

| location_3 | n |
|---|---|
| Ciudad Autónoma de Buenos Aires | 2,859 |
| Palermo | 894 |
| Caballito | 702 |
| Ciudad de Buenos Aires | 599 |
| Belgrano | 591 |
| Recoleta | 573 |
| Almagro | 523 |
| Villa Urquiza | 438 |
| Villa Crespo | 411 |
| Villa Devoto | 384 |

### 7.5 Centros comerciales / sub-barrios céntricos como referencia
| centro | nivel | n_train | n_test |
|---|---|---|---|
| Microcentro | location_2 | 130 | 0 |
| Centro | location_2 | 130 | 0 |
| Centro | location_3 | 3,726 | 33 |
| Catalinas | location_2 | 1 | 0 |
| Catalinas | location_3 | 9 | 0 |
| Tribunales | location_2 | 67 | 0 |
| Once | location_2 | 121 | 0 |
| Once | location_3 | 318 | 11 |
| Retiro | location_2 | 406 | 9 |
| Retiro | location_3 | 775 | 107 |
| Puerto Madero | location_2 | 234 | 0 |
| Puerto Madero | location_3 | 753 | 263 |

**Veredicto**: VIABLE en E3: 10 centros con ≥ 50 obs en train (Centro, Microcentro, Once, Puerto Madero, Retiro, Tribunales). Se puede calcular su centroide y derivar `dist_a_<centro>` sin APIs.


## Conclusiones accionables para los experimentos del plan
- **Candidatas claras a transformación log/sqrt** (Análisis 1): price, m2, n_dormitorios, n_banos, len_descripcion, n_features. Esto refina el v2 del plan (que tenía sólo `np.log(m2)`). `n_banos` aparece con skew 310 por un máximo de 140 000 — un dato basura que confirma que el capping `[0, 15]` heredado de v4 sigue siendo necesario.
- **Top 5 features binarias por IG** (Análisis 3): f_cochera, f_balcon, f_pileta, f_gimnasio, f_parrilla. `f_cochera` y `f_balcon` solas explican IG = 0.143 (~4.3% de H(Y)).
- **Bottom 5 features binarias por IG** (Análisis 3): f_calefaccion, f_aire_acondicionado, f_internet, f_gas_natural, f_alarma. Candidatas a podar o a fundir en una score agregada.
- **Mayores drifts train↔test** (Análisis 3b): f_cocina_equipada (+4.82pp), f_cochera (+3.12pp), f_aire_acondicionado (+2.68pp), f_gas_natural (+2.54pp), f_garage (+2.13pp). El test tiene SISTEMÁTICAMENTE más amenities marcadas que el train — covariate shift consistente, no ruido aleatorio. **Tensión IG-alto + drift-alto**: f_cochera aporta(n) señal en el train pero su prevalencia es distinta en el test — riesgo v5 (señal que no generaliza).
- **Barrios** (Análisis 4): 25 barrios chicos sobre 98 totales, pero cubren sólo 0.22% del train; el 88.26% de las filas vive en barrios > 1000 obs. **Esto debilita el v5 del plan**: la discretización supervisada del `barrio_id` no compra agregando barrios chicos (no hay con qué); si se hace, es para reducir la cardinalidad efectiva del feature, no para rescatar muestras.
- **Temporal** (Análisis 2): NO hay desbalance grosero (delta_pct máximo = 1.62pp). El shift de v6 NO viene de 'el test cae en años raros'. La hipótesis más sólida es que el RF aprendió correlación año↔precio que no generaliza dentro de cada año (lookup vacío). v3 sigue valiendo, pero la justificación cambia: no es 'sustituir año por estacionalidad' sino 'eliminar el lookup año al RF'.
- **Insumo C3PO — parseo dual** (Análisis 5): ENTRA en v2 (parseo dual rescata > 1% en al menos un atributo). Recupera 8.86% de filas con `m2`, 3.07% con `n_dormitorios`, 1.86% con `n_banos`. Es un upgrade limpio del pipeline de v4 → entra al v2.
- **Insumo C3PO — surface total/covered** (Análisis 6): DESCARTAR (cobertura 52.4% o difieren sólo en 1.75% — no agrega señal). La mediana del ratio es 1.0 porque el regex de cubierta hace match con el texto general (overlap del patrón `m²`). No agrega señal real → no entra.
- **Geografía interna — sub-barrios** (Análisis 7.1): comparar `n_unicos` de location_2 vs location_3 en la tabla. Si difieren, agregar `barrio_principal = location_2` como feature paralela a la actual `barrio = location_3`.
- **Geografía interna — limpieza barrio** (Análisis 7.2): PRIORIDAD ALTA: hasta 80.64% del train y 95.06% del test tienen barrio mal etiquetado o NaN. Vale la cascada Hot Deck por description → KNN sobre lat/lon (Nota 3).
- **Geografía interna — centros comerciales como referencia** (Análisis 7.5): VIABLE en E3: 10 centros con ≥ 50 obs en train (Centro, Microcentro, Once, Puerto Madero, Retiro, Tribunales). Se puede calcular su centroide y derivar `dist_a_<centro>` sin APIs.
