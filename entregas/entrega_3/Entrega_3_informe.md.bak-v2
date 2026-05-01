# Entrega Parcial 3 - Patricio Gerpe

Partí exactamente del campeón de la Entrega 2 y, como todavía no encontré una ingeniería de atributos ni una reducción de dimensionalidad que mejoren a la vez validación local y Kaggle, mantuve sin cambios filtros, outliers, imputación y Hot Deck. En números, seguí con 123 102 filas post-filtro, eliminé 719 outliers multivariados y terminé con 122 383 filas; además, el Hot Deck por `description` siguió cubriendo 1 058 casos del test (7.85 %).

## A. Ingeniería de atributos

### A.1
Al cerrar la Entrega 2 me quedó una tensión clara: v5 había mejorado en validación aleatoria pero empeorado en Kaggle, así que antes de sumar atributos nuevos necesité separar señal genuina de sobreajuste temporal. Esa lección me hizo conservar sólo los derivados que ya habían mostrado estabilidad fuera de muestra: `m2`, `n_dormitorios`, `n_banos`, 17 flags de amenities, `len_descripcion`, `n_features` y `m2_was_na`, para 26 features. Mi hipótesis fue que, sin una validación temporal confiable, agregar variables “atractivas” podía repetir el error de v5. El resultado sostuvo esa cautela: con este set obtuve 116 670 de CV5 y 117 236 en holdout temporal, muy cerca entre sí, en lugar de la brecha de más de 10 000 puntos que había aparecido en E2.

### A.2
Con esa base estabilizada, el problema dejó de ser “agregar más” y pasó a ser qué atributos convenía no tocar. El fracaso previo de las variables temporales me enseñó que `pub_year` y `pub_month` capturaban drift macroeconómico más que valor del inmueble; por eso no las reincorporé, aunque en KFold parecían útiles. También descarté volver a estadísticas de precio por barrio: en E2, v2 empeoró Kaggle de 93 167 a 96 891 y v3 a 97 423 porque esa familia de variables filtraba información del target. Aprendí entonces que, en esta entrega, una buena ingeniería de atributos no era la más ambiciosa sino la que preservaba señal estructural sin reabrir leakage ni shift.

## B. Reducción de dimensionalidad

### B.1
Esa prudencia me llevó a una decisión menos vistosa pero más sólida: no apliqué reducción de dimensionalidad en la corrida campeona. Con sólo 26 variables, la mayoría interpretables y varias binarias, no tenía evidencia de maldición de dimensionalidad ni de ruido por alta correlación que justificara comprimir el espacio. Mi hipótesis fue que reducir dimensiones acá podía mezclar superficie, ubicación y amenities, y quitarle capacidad de partición al Random Forest.

### B.2
Con esa pregunta resuelta, preferí usar la entrega para fijar una línea de base robusta antes de probar PCA u otras variantes. El aprendizaje fue concreto: en E3 la principal mejora no vino de transformar el espacio de atributos, sino de endurecer el criterio de validación que decide qué merece llegar a Kaggle.

## C. Modelo (Predicción)

| Entrega | Kaggle RMSE |
|---|---:|
| Entrega 2 | 93 151 |
| Entrega 3 | **93 146.749** |

Esa comparación muestra una mejora marginal de 4.251 puntos. No la atribuyo a una feature nueva, sino a haber consolidado como campeón el pipeline de E2 bajo un esquema de validación más exigente: CV5 multi-seed de 116 670 ± 2 521 y holdout temporal de 117 236. Mi conclusión es que E3, por ahora, aportó más en control experimental que en ganancia de score.

## D. Entrega

El principal resultado de esta entrega fue metodológico: convertí en campeón una versión que casi replica E2 pero ya está evaluada con un criterio compatible con el shift temporal del test público. Eso me deja una base más confiable para la Entrega 4, donde espero sumar señal nueva desde texto, geografía y datos externos, en lugar de seguir forzando atributos estructurados con rendimiento decreciente.
