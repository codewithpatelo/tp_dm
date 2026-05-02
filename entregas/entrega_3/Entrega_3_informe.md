# Entrega Parcial 3 - Patricio Gerpe

Partí del campeón de la Entrega 2 y mantuve sus filtros, outliers, imputación y Hot Deck porque no tenía evidencia de que tocar esa base mejorara fuera de muestra. Esa herencia dejó 123 102 filas post-filtro, eliminó 760 outliers multivariados y terminó en 122 342 filas; además, el Hot Deck por `description` siguió cubriendo 1 058 casos del test, el 7.85 %.

## A. Ingeniería de atributos

Al cerrar E2 me había quedado una duda concreta: cuando faltaban `m2`, `n_dormitorios` y `n_banos`, ¿el dato no existía o estaba mal ubicado? Mi hipótesis fue que seguía escrito en `description`, así que en v2 usé un parseo dual: primero `features` y, sólo si quedaba `NaN`, fallback a la descripción; además sumé `rooms` desde patrones de "ambientes". El rescate fue grande en train: 10 844 `m2`, 3 168 dormitorios y 2 293 baños; en test recuperé 29, 80 y 55. Esa señal fue genuina: el CV5 bajó de 116 670 a 112 777, el holdout temporal de 117 236 a 111 536 y Kaggle de 93 146.749 a 91 399.286. Aprendí que recuperar valores reales de variables estructurales vale más que seguir agregando proxies.

Con esa base más completa, me pregunté si convenía transformar variables sesgadas. Probé en v3 `log1p` sobre `m2`, dormitorios, baños y longitudes de texto, pero el resultado fue peor: el CV5 subió a 112 884 y el holdout temporal a 111 556. Eso confirmó la teoría vista en clase: en árboles, una transformación monótona no agrega información nueva porque los splits ya encuentran umbrales útiles en la escala original. El fracaso de v3 me empujó a buscar señal nueva, no otra escala de la misma señal.

Esa pista me llevó al texto libre. Si v2 había mejorado por rescatar datos explícitos de `description`, todavía podía quedar información semántica sobre estado, calidad o amenities que mis variables manuales no capturaban. Por eso en v8 representé `description` con TF-IDF de unigramas y bigramas, limité el vocabulario a 5 000 términos y lo comprimí con TruncatedSVD a 20 componentes, que explicaron 25.64 % de la varianza. A diferencia de v5, donde distancias, ratios y densidad empeoraron el CV5 de 112 300 a 112 441 por redundar con `lat/lon`, acá sí apareció señal nueva: el CV5 bajó a 111 160, el holdout temporal a 102 112 y Kaggle a 89 877.580.

Sobre esa base, en v11 apliqué la técnica de ingeniería de atributos vista en Clase 07: construí cinco scores temáticos a partir de las 16 features binarias de amenities, agrupando por semántica —lujo (`f_pileta`, `f_gimnasio`, `f_parrilla`, `f_jardin`), servicios (`f_aire_acondicionado`, `f_calefaccion`, `f_gas_natural`, `f_internet`), estacionamiento (`f_cochera`, `f_garage`), espacios extra (`f_balcon`, `f_bodega`, `f_cuarto_de_servicio`) y seguridad (`f_seguridad`, `f_alarma`)—. La hipótesis era que el score pre-computado captura el efecto aditivo de las amenities directamente, reduciendo la profundidad de árbol necesaria para detectar la combinación, y que la señal agregada es más robusta frente al ruido que cada binaria individual.

## B. Reducción de dimensionalidad

Apliqué dos técnicas de Clase 08 en v11. La primera fue TruncatedSVD sobre la representación TF-IDF del campo `description`: comprimí miles de términos sparse en 20 componentes densos, manteniendo la señal semántica mientras eliminaba el ruido léxico. La segunda fue SelectKBest con `mutual_info_regression` sobre las 16 amenities binarias, conservando las 10 más informativas respecto al precio y descartando `f_alarma`, `f_bodega`, `f_calefaccion`, `f_cocina_equipada`, `f_gas_natural` e `f_internet`.

El fit de ambas transformaciones se hizo exclusivamente sobre el conjunto de entrenamiento y luego se aplicó al test, sin leakage. El resultado de v11 fue el mejor en métricas locales: CV5 bajó a 106 840 (−4 320 respecto a v8) y el holdout temporal a 101 041 (−1 071).

## C. Modelo (Predicción)

| versión | CV5 RMSE (mean ± std) | holdout temporal | Kaggle RMSE |
|---|---:|---:|---:|
| E2-v4 (base) | 112 300 ± 2 191 | 110 492 | 93 151.000 |
| v8 (TF-IDF+SVD) | 111 160 ± 3 375 | 102 112 | 89 877.580 |
| **v11 (campeón)** | **106 840 ± 2 765** | **101 041** | **90 613.645** |

La mejora de v11 contra E2 fue de 5 460 puntos en CV5 y 9 451 en holdout temporal. Mantuve `RandomForestRegressor` con 500 árboles y profundidad 50 durante todos los experimentos; la mejora proviene exclusivamente de las features y de la reducción de dimensionalidad.

## D. Entrega

El recorrido de esta entrega me dejó dos lecciones complementarias. La primera: cuando una feature nueva sólo reordena información ya presente —logs o distancias derivadas de `lat/lon`— el score no se mueve; cuando agrega señal genuina —texto libre comprimido, scores de amenities— sí aparece una mejora robusta en validación. La segunda: la reducción de dimensionalidad no es un paso cosmético, sino la forma de volver utilizable una fuente de alta dimensión como el texto; sin TruncatedSVD, las 5 000 columnas sparse del TF-IDF habrían sido inutilizables para el Random Forest. Eso me deja bien posicionado para la Entrega 4, donde el paso natural es sumar información externa y geográfica que hoy no está en las columnas estructuradas.
