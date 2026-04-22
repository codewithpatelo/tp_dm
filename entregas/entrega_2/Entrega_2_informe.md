# Entrega Parcial 2 — Patricio Gerpe

Partí del pipeline de la Entrega 1 y mantuve sus filtros (venta, dólares, CABA, precio entre USD 5 000 y 3 000 000), con dos ajustes mínimos guiados por `a_predecir`: amplié CABA a casos mal rotulados como `location_1="Buenos Aires"` pero con barrio válido en `location_2/3`, recuperando 3 341 filas (+2.8 %), y extendí `property_type` con `ph` y `cochera`, presentes en el test. Quedaron 123 102 filas antes de tratar atípicos y faltantes.

## A. Datos Atípicos (outliers)

**A.1** Al crear atributos desde `features`, la primera alarma apareció en `m2`: el percentil esperable quedaba muy por debajo de algunos valores absurdos. Sospeché errores de extracción desde texto, no propiedades raras, así que mi hipótesis fue preservar la fila y corregir el atributo, conservando la señal de ubicación y tipo.

**A.2** Esa distinción me llevó a separar “columna sucia” de “fila sucia”. Capé `m2 ∈ [10,1500]`, `n_dormitorios ∈ [0,15]` y `n_banos ∈ [0,15]` mandando el resto a `NaN` para que un parseo imposible no dominara los splits del Random Forest, y para fila sucia apliqué `IsolationForest` sobre `(price, m2, n_dormitorios, n_banos)` escalados con `contamination=0.01` (el EDA mostraba una minoría clara de casos raros, no un 5 % sospechosamente alto). Eliminé 719 filas, quedando 122 383.

**A.3** La variable con más outliers fue **`m2`**, justamente la más expuesta a errores de parseo. El enfoque adecuado para esa columna no fue borrar registros sino **capar extremos a `NaN` e imputar con mediana**, preservando la observación completa.

## B. Datos Faltantes

**B.1** Los `NaN` que dejó A se sumaron a faltantes previos en `lat/lon` y derivadas, así que el problema pasó de limpiar a completar sin inventar estructura. Como en clase la ausencia también puede cargar información, busqué imputaciones sobrias y, cuando aportaba, agregué un marcador.

**B.2** Mediana global (ajustada en train) para `m2`, `lat`, `lon`, `n_dormitorios`, `n_banos`, `len_descripcion` y `n_features`; moda para `property_type` y `barrio`; más `m2_was_na`. Después un **Hot Deck** por `description` normalizada (Clase 4) para casos del test casi idénticos a publicaciones del train: el override pasó de 925 (6.87 %) en v1 a 1 058 (7.85 %) en v4 — 133 publicaciones más donde dejé de extrapolar y copié un precio observado.

**B.3** El fracaso de v2 y v3 me enseñó por qué una imputación puede subir el error: agrego una señal demasiado optimista en train que no generaliza. v2 incorporó `precio_mediano_barrio*` y Kaggle empeoró de 93 167 a 96 891; v3 imputó `lat/lon` por mediana de barrio y volvió a empeorar a 97 423. Ambas introdujeron leakage o una localización artificial, así que en v4 volví a una imputación conservadora.

## C. Modelo (Predicción)

| Entrega | Mejor submission | RMSE Kaggle |
|---|---|---:|
| Entrega 1 | `solucion (12).csv` | 166 979.008 |
| Entrega 2 | `v4` | **93 151.157** |

La mejora de **73 827.851** puntos no vino de tocar el modelo —la consigna lo prohibía— sino de darle una entrada más creíble: 719 filas anómalas removidas, 26 atributos útiles y un Hot Deck más preciso. Como el holdout único oscilaba demasiado (std ≈ 2 K, mayor que la diferencia entre versiones), adopté **CV5** como criterio principal: la corrida campeona dio **117 218.50 ± 2 392** antes de confirmar Kaggle.

## D. Entrega y próximos pasos

Entrego **v4** como campeona, con RMSE Kaggle **93 151.157**: tratar atípicos y faltantes no fue agregar la mayor cantidad posible de información, sino conservar sólo la que sigue siendo verosímil fuera del train.

Después de v4 probé en v5 ocho features nuevas: bajaron CV5 en 4 877 puntos pero subieron Kaggle en 4 347. Sospeché *distribution shift* y en v6 lo confirmé con un experimento diagnóstico (CV5 multi-seed + holdout temporal por `publication_date` + mini-ablation): el holdout temporal quedó **+10 491 puntos por encima del CV5** (122 795 vs 112 304). La ablation aisló culpables: `pub_year`/`pub_month` rompen el holdout temporal en +3 385 sin mover CV (shift puro); los cinco booleanos textuales mejoran CV5 en −4 452 pero el holdout temporal apenas en −593 (mejora ~7× engañada por el split aleatorio); sólo `floor` mejora ambas, aunque por debajo del margen de submit. Por eso v4 sigue siendo el envío.

**Próximos pasos — Entrega 3 (ingeniería de atributos + reducción de dimensionalidad)**: incorporar `floor`; rediseñar los booleanos como **interacciones** con `barrio`/`m2` para que dejen de overfittear al CV; sustituir los temporales crudos por encodings que no carguen el año como nivel (estación, días desde el primer scrape); y aplicar reducción de dimensionalidad —PCA o selección por importancia **out-of-fold**— sobre las 26 features para podar las que no aporten al holdout temporal, métrica primaria de ahora en más.
