# Entrega Parcial 3 - Patricio Gerpe

Partí del campeón de la Entrega 2 y mantuve filtros, outliers, imputación y Hot Deck porque no tenía evidencia de que tocar esa base mejorara fuera de muestra. Eso dejó 123 102 filas post-filtro, eliminó 760 outliers multivariados y terminó en 122 342 filas; además, el Hot Deck por `description` siguió resolviendo 1 058 casos del test, el 7.85 %.

## A. Ingeniería de atributos

### A.1
Al cerrar la Entrega 2 me había quedado una contradicción: v5 mejoraba en validación aleatoria pero empeoraba en Kaggle, así que antes de sumar atributos nuevos necesité separar señal genuina de atajos espurios. Mi hipótesis fue que faltaba información estructural en filas donde `features` venía incompleta, pero que esa señal seguía escrita en `description`. Por eso probé en v2 un parseo dual: extraer `m2`, `n_dormitorios` y `n_banos` desde `features` y, sólo si faltaban, recuperarlos desde la descripción; además sumé `rooms`. Esa decisión rescató en train 10 844 valores de `m2`, 3 168 de dormitorios y 2 293 de baños, y en test otros 29, 80 y 55. El resultado fue claro: pasé de 116 670 a 112 777 en CV5, de 117 236 a 111 536 en holdout temporal y de 93 146.749 a 91 399.286 en Kaggle.

### A.2
Esa mejora me obligó a preguntarme qué atributos convenía seguir dejando afuera. El fracaso de v2 y v3 de la entrega anterior ya me había enseñado que las estadísticas de precio por barrio no agregaban contexto sino leakage: Kaggle había empeorado de 93 167 a 96 891 y 97 423. A la vez, v5 mostró que `pub_year` y `pub_month` parecían útiles en KFold pero capturaban drift temporal. Por eso la corrida campeona no buscó la ingeniería más ambiciosa, sino la más defendible: 27 variables, casi todas interpretables, donde la novedad real fue rescatar información faltante del mismo aviso en lugar de inyectar target o tiempo.

## B. Reducción de dimensionalidad

### B.1
Con ese rescate resuelto, el siguiente problema era si convenía comprimir el espacio. Decidí no aplicar reducción de dimensionalidad en la versión campeona porque no vi el patrón que la justificara: trabajé con 27 features, muchas binarias y semánticamente distintas, y el Random Forest tolera bien esa escala. Mi hipótesis fue que una compresión prematura podía mezclar superficie, ubicación y amenities, justo cuando la mejora de v2 vino de volver más explícitas esas señales, no de fusionarlas. El aprendizaje fue que, antes de reducir dimensiones, primero tengo que agotar atributos nuevos con significado claro.

## C. Modelo (Predicción)

| Entrega | Kaggle RMSE |
|---|---:|
| Entrega 2 | 93 151 |
| Entrega 3 | **91 399.286** |

Esa comparación muestra una mejora de 1 751.714 puntos. No la explico por cambiar el modelo, porque seguí con `RandomForestRegressor` de 500 árboles y profundidad 50, sino por haber recuperado variables básicas que faltaban en parte del dataset. La consistencia entre CV5 y holdout temporal también mejoró: la brecha quedó en 1 241 puntos, muy lejos del desalineamiento que había delatado el shift en E2.

## D. Entrega

El resultado principal de esta entrega fue confirmar que la mejor ingeniería de atributos no era sumar variables vistosas, sino rescatar señal perdida dentro del propio aviso. Eso me deja una base más fuerte para la Entrega 4: si quiero volver a mejorar, ya no alcanza con exprimir columnas estructuradas; necesito incorporar información nueva desde texto libre y geografía.
