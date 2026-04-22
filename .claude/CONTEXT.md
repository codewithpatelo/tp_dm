# Contexto del Trabajo Práctico — Data Mining (UBA Exactas)

## Datos del curso

- **Materia:** Data Mining
- **Carrera:** Esp. en Explotación de Datos y Descubrimiento de Conocimiento (UBA Exactas)
- **Alumno:** Patricio Gerpe
- **Modalidad:** TP individual estructurado como **competencia de Kaggle**.
- **Competencia:** [fcen-dm-2026-prediccion-precio-de-propiedades](https://www.kaggle.com/competitions/fcen-dm-2026-prediccion-precio-de-propiedades)
- **Objetivo:** Predecir el **precio (USD)** de publicaciones de propiedades en Argentina.
- **Métrica:** RMSE (Raíz del Error Cuadrático Medio).
- **Formato de envío:** CSV con header `id,price`.

## Estructura del TP

Hay **4 entregas parciales (10 % cada una)** y una **entrega final (60 %)**. Cada entrega parcial pide:

1. Un CSV con la predicción subido a Kaggle.
2. Un informe corto (PDF/DOCX) en el campus, **máx. 1 carilla**, dirigido a un "jefe / supervisor" — explica decisiones y resultados, **no código**.
3. (Sólo en la final) la notebook `.ipynb` completa.

### Cronograma temático de entregas

| Entrega | Foco temático |
|---|---|
| **Entrega 1** | Filtros |
| **Entrega 2** | Manejo de valores atípicos + manejo de faltantes |
| **Entrega 3** | Ingeniería de atributos + reducción de dimensionalidad |
| **Entrega 4** | Datos no estructurados + APIs y web scraping + datos geográficos |
| **Entrega Final** | Integración completa según `Enunciado General de Trabajo Práctico de Posgrado.md` |

Esta secuencia importa porque las restricciones técnicas de cada entrega
se acumulan: la Entrega ⟨n⟩ puede usar cualquier técnica vista en
entregas anteriores, pero **no** puede anticipar las de entregas
futuras. Y la sección "próximos pasos" del informe de cada entrega debe
apuntar al foco temático de la siguiente: el cierre de E2 ya tiene que
estar mirando hacia ingeniería de atributos + reducción de
dimensionalidad, el de E3 hacia datos no estructurados + geográficos,
etc.

### Restricción central

> Cada entrega **solo puede usar técnicas vistas hasta la clase previa a esa entrega**. Nada de XGBoost, nada de librerías externas no vistas en clase, nada de datos externos.

Para aprobar una entrega hay que superar el RMSE de los docentes en el leaderboard correspondiente.

### Reglas sobre el modelo por entrega

El modelo "oficial" del curso es `RandomForestRegressor`. La libertad
para tocarlo se va abriendo entrega por entrega:

| Entrega | Algoritmo | Hiperparámetros tuneables |
|---|---|---|
| 1 | `RandomForestRegressor` | ❌ ninguno (fijos: `n_estimators=500`, `max_depth=50`) |
| 2 | `RandomForestRegressor` | ❌ ninguno (fijos: `n_estimators=500`, `max_depth=50`) |
| **3** | **`RandomForestRegressor`** | **✅ `n_estimators` y `max_depth`. El resto sigue fijo / por defecto.** |
| 4 | libre (incluye otros modelos vistos en clase) | libre (dentro de lo visto) |
| Final | libre | libre (dentro de lo visto) |

Esto cambia la sección "⛔ NO TOCAR ⛔" del notebook base: a partir de
E3, sí se puede tocar pero **solamente** esas dos perillas. Tunearlas
introduce una decisión metodológica nueva: ¿se tunea sobre el pipeline
de features fijo y después se hace FE, o al revés? El plan de E3 fija
el orden y lo justifica.

### Robots de la cátedra (umbral de aprobación)

Para cada entrega parcial, la cátedra publica una notebook "robot" con
una solución mínima. Es el **umbral de aprobación**: hay que ganarle al
robot en Kaggle para aprobar la entrega. Conviene leerlo apenas se
publica para (i) descartar ideas que ya tenemos cubiertas y (ii)
detectar técnicas o features que el robot use y que nosotros no estemos
explotando.

| Entrega | Robot (notebook) | Estado |
|---|---|---|
| 1 | `Robot_1_R2D2.ipynb` | superado |
| 2 | `Robot_2_C3PO.ipynb` | superado ampliamente |
| 3 | _(pendiente de publicación)_ | — |
| 4 | _(pendiente de publicación)_ | — |
| Final | _(pendiente de publicación)_ | — |

Cada vez que llegue un robot nuevo se anota acá su archivo y el estado
("pendiente de comparar / superado / no superado"); las ideas
aprovechables van a la sección *"Lecciones aprendidas"* de la entrega
correspondiente.

## Datasets locales (en `datasets/`)

- `entrenamiento.db` — SQLite (~2 GB, 1 292 674 filas, 18 columnas) con la tabla `entrenamiento`.
- `a_predecir.csv` — 13 471 filas (test).
- `ejemplo_de_solucion.csv` — formato de envío.

### Esquema de columnas

`id, description, address, lat, lon, publication_date, publisher_id, features, location_0, location_1, location_2, location_3, location_4, operation_type, property_type, source, price, currency_type`

### Diferencia clave train vs test (¡motiva el filtrado!)

- **`a_predecir`** es exclusivamente **CABA**, **siempre `venta`**, casi 100 % en **dólares**, y `property_type ∈ {departamento, casa, cochera}`.
- **`train`** abarca todo el país, varias monedas, todas las operaciones, todos los tipos de propiedad. → Hay que **filtrar el train** para acercarlo al universo del test.

### Particularidades

- La columna **`features`** es texto con `;`-separated tokens del estilo `"3 dormitorios;2 baños;91 m²;balcón;..."`. Se puede parsear para obtener `n_dormitorios`, `n_banos`, `m2` y flags binarios de amenities (pileta, garage, parrilla, etc.).
- Categorías redundantes: `departamento`/`departamentos`, `casa`/`casas`, etc.
- Muchos faltantes en `lat`, `lon`, `publication_date`, `publisher_id`, `description`, `address` y en la propia `price` (hay 308 197 filas sin price en train).
- En train hay duplicados de la misma publicación: ~13 676 descripciones repetidas y un buen número de coincidencias **train ↔ test** por `description` (~925 filas de test cubiertas) o por `address` (~6 789, pero más ruidoso).

## Referencia de scores

- **Target nice-to-have**: RMSE < 62 821.209 (mejor score actual en el leaderboard de la competencia, al 2026-04-20).
- No es un objetivo rígido; no forzar decisiones técnicas para perseguirlo si comprometen la claridad o la justificación del TP.

### Resultados registrados (entrega_2)

| versión | RMSE CV5 (mean ± std) | RMSE holdout temporal | RMSE holdout | RMSE Kaggle | estado |
|---|---:|---:|---:|---:|---|
| v1 | — | — | 118 867 | 93 167 | |
| v2 | — | — | 116 923 | 96 891 | leakage: barrio price stats |
| v3 | — | — | 117 080 | 97 423 | leakage: lat/lon centroide barrio |
| **v4** | **117 219 ± 2 392** | — | 118 867 | **93 151** | **campeón actual** |
| v5 | 112 341 ± 1 659 | — | 113 952 | 97 498 | distribution shift: temporales/booleanos/floor |
| v6 | 112 304 ± 2 794 | 122 795 | 113 952 | — (no submit) | diagnóstico: multi-seed CV5 + holdout temporal + mini-ablation v5 |

Umbral de submit (v6+, doble criterio): mejora simultánea > 1 500 en
`rmse_cv5_mean` (multi-seed) **y** > 1 500 en `rmse_holdout_temporal`. El
holdout único queda como columna informativa, no decide.

## Directiva: experimentos fallidos como aprendizaje

Cada experimento que no mejora el score **es un aprendizaje valioso**. Al registrar un experimento fallido, explicitar:
- Qué hipótesis se planteó y por qué se esperaba que funcionara.
- Qué resultado se obtuvo y en qué difirió de lo esperado.
- Por qué creemos que no funcionó (leakage, ruido, señal ya capturada por el modelo, etc.).

Esto evita repetir los mismos errores y construye conocimiento acumulado sobre el dataset.

## Estado de las entregas

### Entrega 1 — ENTREGADA

Hecho:

- AID exploratorio (countplots, boxplots, medidas de posición y dispersión por tipo de propiedad).
- Modelo Random Forest baseline.
- Filtros que se mantienen para todas las entregas siguientes:
  - `operation_type == "venta"`.
  - `currency_type == "dolares"`.
  - `location_1 ∈ {Capital Federal, Ciudad Autónoma de Buenos Aires}`.
  - `property_type` unificado (`departamento`/`departamentos` y `casa`/`casas`).
  - `price ∈ [USD 5 000, USD 3 000 000]`.
  - Imputación rápida con `fillna(0)` (a mejorar en esta entrega).

Detalle en `Entrega Parcial 1 - Patricio Gerpe (1).pdf`.

**Scores Kaggle de Entrega 1** (para la comparación pedida en C de Entrega 2):

| submission | Kaggle RMSE |
|---|---:|
| `solucion (12).csv` ← mejor | **166 979.008** |
| `solucion (1).csv` | 171 308.383 |
| `solucion (3).csv` | 173 224.183 |
| `solucion (2).csv` | 183 584.683 |
| `solucion (11).csv` | 192 650.506 |
| `solucion (10).csv` | 204 146.779 |
| `solucion.csv` | 331 069.059 |
| `solucion (4).csv` | 538 205.576 |

La que corresponde al informe entregado de Entrega 1 es **`solucion (12).csv` = 166 979.008**.
Este es el **número de referencia** para la comparación del punto **C. Modelo (Predicción)**
de la Entrega 2 (evaluar evolución del modelo al aplicar las técnicas nuevas).

### Entrega 2 — EN CURSO

#### Lecciones aprendidas

- **v2 → leakage explícito**. `precio_mediano_barrio*` mejoraba el holdout
  en ~1.9 K pero EMPEORABA Kaggle en +3.7 K. La mediana incluía cada fila
  en su propio agregado (target encoding mal hecho). Lección: cualquier
  feature que use el target tiene que computarse out-of-fold.
- **v3 → leakage del mismo tipo, otra dimensión**. Imputar `lat`/`lon` por
  mediana del barrio metía señal del precio agregado del barrio dentro
  de las coordenadas. Local "mejoraba" 1.8 K (dentro del ruido del
  holdout) y Kaggle pasó de 93 167 → 97 423. Lección: revisar si la
  imputación condicional inyecta información del target indirectamente.
- **v5 → distribution shift, NO leakage**. Las 8 features añadidas
  (`floor`, `is_a_estrenar`, `is_reciclado`, `has_suite`,
  `has_service_room`, `is_near_subway`, `pub_year`, `pub_month`)
  mejoraron CV5 en 4 877 puntos (>2× el std intra-fold) y EMPEORARON
  Kaggle en 4 347 puntos. No fue varianza de validación: KFold
  aleatorio reparte los años uniformemente entre folds, así que **no
  detecta** que el test público puede ser de un período distinto.
  Lección: a partir de v6, validación = **CV5 multi-seed + holdout
  temporal** (último 20 % por `publication_date`). La regla "decidir por
  CV" sigue valiendo, pero la CV tiene que **imitar el test**.
- **v6 → diagnóstico (no submit), aísla la causa de v5**. Mini-ablation
  de las 8 features de v5 sobre v4, midiendo CV5 single-seed +
  holdout temporal (deltas vs v4):
  - `floor`: ΔCV5 = −466, Δholdout temporal = −1 103. Única feature
    que mejora **ambas** métricas en la misma dirección, aunque por
    debajo del margen de submit (1 500). Señal genuina, no shift.
  - `pub_year, pub_month`: ΔCV5 = +48, Δholdout temporal = **+3 385**.
    No mueve CV pero rompe el holdout temporal — firma textbook de
    distribution shift: el RF aprende correlación año↔precio que se
    invierte fuera de muestra. **Culpable principal del +4 347 de v5.**
  - 5 booleanos de calidad textual (`a_estrenar, reciclado, suite,
    service_room, subway`): ΔCV5 = **−4 452**, Δholdout temporal = −593.
    Brecha de ~7×: el KFold aleatorio se "comía" un shift que el split
    temporal sí revela. Mejora local **engañosa**.
  El holdout temporal queda **+10 491 puntos por encima del CV5** en
  v6 (122 795 vs 112 304), lo que cuantifica el shift estructural
  train↔test público. Esta lección define la dirección de la Entrega 3:
  el feature engineering nuevo se valida primero contra el holdout
  temporal, no contra el CV5.
- **Robot 2 (C3PO) — qué hace que vale la pena revisar para E3**.
  El robot usa pipeline más naive que v4 (sin `ph`/`cochera`, sin
  IsolationForest, `fillna(0)`, target encoding `price_m2` por
  `(property_type, location_2)` con leakage, modelo idéntico). Lo
  superamos ampliamente, pero hay tres ideas que NO tenemos en v4 y
  encajan en feature engineering / Clase 7 (vienen al plan de E3, no a
  E2):
  - Parsea regex sobre **`features` + `description` concatenados** —
    nosotros sólo `features`. Recupera info en filas donde el dato
    crítico (m², dormitorios, baños) está en la descripción libre.
  - Extrae **`surface_total_m2` y `surface_covered_m2` por separado**
    (con `m²?\s*cubiert[ao]s?` para la cubierta) — nosotros tenemos un
    único `m2`. Si el m² total y el cubierto difieren, perdemos esa
    diferencia.
  - Deriva **`pct_surface_covered = covered / total`** y **`rooms`**
    (ambientes) como feature separada de `n_dormitorios`.
  Lo que **no** vale la pena copiar: `price_m2` por
  `(property_type, location_2)` (es exactamente el leakage de v2),
  filtro por `price_m2 ∈ [300, 9000]` (lo cubre nuestro
  IsolationForest), `fillna(0)` (nuestra mediana + Hot Deck es mejor),
  bug en cell 13 que sobreescribe `bedrooms` con `bool` (es un error
  del robot).

#### Delta respecto a Entrega 1

Mantengo los filtros de E1 (`operation_type == "venta"`,
`currency_type == "dolares"`, `price ∈ [USD 5 000, USD 3 000 000]`) y
agrego dos ajustes mínimos justificados por mirar `a_predecir`:

1. **CABA ampliado vía `location_2/3`**: además del filtro estricto
   `location_1 ∈ {Capital Federal, Ciudad Autónoma de Buenos Aires}` de
   E1, incluyo las filas con `location_1 = "Buenos Aires"` cuyo
   `location_2` o `location_3` coincide con un barrio observado en el
   test. Recupero **+3 341 filas (+2.8 %)** de entrenamiento legítimo
   que E1 perdía por mal etiquetado.
2. **`property_type` extendido con `ph` y `cochera`**: E1 sólo unificaba
   `{casa/casas, departamento/departamentos}`. Sumo `ph` y `cochera`
   porque están presentes en `a_predecir` (los cuatro tipos juntos
   cubren 100 % del test).

Consigna (`Consigna Entrega Parcial 2.pdf`):

- **A. Datos atípicos (outliers):** análisis, estrategia, identificar la variable con más outliers.
- **B. Datos faltantes:** evaluar proporción, definir e implementar imputaciones; razonar qué pasa cuando la imputación empeora el error.
- **C. Modelo:** comparar predicción nueva vs entrega anterior, evaluar evolución.
- **D. Entrega:** informe de ≤ 1 carilla + código fuente.

Restricción explícita:

> "El código solamente debe tratar los datos faltantes y atípicos y crear nuevos atributos en base a los datos dados. No se pueden usar datos externos ni otras librerías que las vistas en clase."

Notebook base provista: `Colab_Base_para_el_Trabajo_Práctico_(Entrega_2).ipynb`. Tiene una sección de modelo marcada **⛔ NO TOCAR ⛔** (RandomForestRegressor con `n_estimators=500`, `max_depth=50`).

## Material de clase disponible

### Slides (`diapos_clase/`)

- Clase 01 — Presentación + Qué es la ciencia de datos.
- Clase 02 — Preprocesamiento (tipos de atributos, limpieza, discretización, numerización).
- Clase 04 — Análisis de valores atípicos (IQR, Z-score, Mahalanobis, LOF, IsolationForest).
- Clase 04 — Datos faltantes (MCAR/MAR/MNAR; eliminar, imputar; sustitución por media/mediana/moda; **Hot Deck**, **Cold Deck**; regresión; **MICE**; KNN; marcadores de ausencia).

### Colabs prácticos (`colabs_clase/`)

- Clase 01 — Qué es la ciencia de datos.
- Clase 02 — Preprocesamiento.
- Clase 04 — Análisis de valores atípicos.
- Clase 04 — Datos faltantes.
- Clase 05 — Práctica de outliers (penguins + IsolationForest).
- Clase 05 — Práctica de datos faltantes (Ames Housing + KNN/MICE).

### Librerías permitidas (vistas en clase)

`pandas`, `numpy`, `matplotlib`, `seaborn`, `sqlite3`, `scipy.stats.zscore`, y de `sklearn`:

- `model_selection`, `ensemble.RandomForestRegressor`, `ensemble.IsolationForest`.
- `metrics.root_mean_squared_error`.
- `preprocessing.StandardScaler`.
- `neighbors.LocalOutlierFactor`.
- `impute.SimpleImputer`, `KNNImputer`, `IterativeImputer` (MICE), `MissingIndicator`.

## Estrategia ganadora (sugerida por amigo que ya cursó)

> Tomar con pinzas — usar **sólo** lo que cuadre con el contenido de la clase a la fecha de la entrega.

1. Mergear train y test.
2. Sacar duplicados (buscar por `description` o `url`, **no por `id`**); ordenar y quedarse con el registro más cercano al test. (Hay muchos duplicados por las inmobiliarias pueden subir la misma propiedad varias veces con distintos precios en un lapso relativamente cercano de tiempo, tienen distinto ID, pero en si son la misma propiedad).
3. Del train, traer el `price` e imputarlo en test (= **Hot Deck**, técnica vista en Clase 4 → válido para esta entrega).
4. Guardar el override en un diccionario y aplicarlo al final sobre la predicción del modelo.
5. Tratar outliers.
6. Imputar faltantes (promedios / mediana / KNN).

## Plan de la Entrega 2 (lo que estoy implementando)

Notebook reorganizada manteniendo la estructura base, en este orden lógico:

1. **0. Lectura** — cargar train (DB) y test (CSV) **juntos**, marcando `__src__`.
2. **1. AID** — recap de la Entrega 1 + tabla de % nulos por columna.
3. **2.1. Filtrado** — filtros de Entrega 1.
4. **2.4. Creación de nuevos atributos** (anticipado): parseo de `features` → `m2`, `n_dormitorios`, `n_banos`, flags binarios de amenities; `barrio` (`location_3`), `len_descripcion`, `n_features`. Hace falta antes de outliers porque las nuevas columnas son las que se evalúan.
5. **2.2. Outliers**:
    - Univariado: IQR + Z-score sobre `price` y `m2`.
    - Tratamiento: `m2 ∉ [10, 1500]`, `n_dormitorios ∉ [0, 15]`, `n_banos ∉ [0, 15]` → `NaN`.
    - Multivariado: `IsolationForest(contamination=0.01)` sobre `(price, m2, n_dormitorios, n_banos)` escalados — sólo elimina filas en train.
6. **2.3. Imputación**:
    - Numéricas: mediana (rápido) y/o KNNImputer; comparar.
    - Categóricas: moda.
    - Marcar nulos con `MissingIndicator` cuando aporte.
7. **Hot Deck por descripción duplicada**: dict `{id_test → price_train}` — visto explícitamente en Clase 4.
8. **3. Modelo** (NO TOCAR) — RandomForestRegressor + (opcional) CV; definir `mejor_combinacion` para que el código posterior funcione.
9. **4. Predicción**:
    - Aplicar mismas transformaciones a `df_ap`.
    - Predecir.
    - Override con el dict de Hot Deck.
    - Guardar CSV.

## Metodología de experimentación

Antes de incorporar cualquier mejora al notebook de entrega, seguir este ciclo:

1. **Hipótesis**: enunciar explícitamente qué se espera y por qué. Formato sugerido: *"Si implementamos X, el RMSE debería bajar porque Y"*.
2. **Diseño del experimento**: definir qué se cambia, qué queda igual (baseline) y cómo se mide (RMSE en validación local, misma semilla).
3. **Prueba local**: correr el experimento y reportar los resultados numéricos (score baseline vs. score nuevo).
4. **Decisión**: en base a los números, decidir si la mejora se incorpora al submission de Kaggle o se descarta.

No se sube nada a Kaggle sin haber pasado por los pasos 1-3. Esto evita gastar submissions en cambios que empeoran el score.

## Informe de entrega

Cada entrega tiene un único informe (≤ 1 carilla) que se trabaja como
Markdown en `entregas/<entrega>/Entrega_<n>_informe.md` y se exporta a PDF
en el mismo directorio.

### Naming del archivo final

El PDF que se sube al campus **debe** llamarse exactamente:

```
Entrega Parcial <n> - Patricio Gerpe.pdf
```

con `<n>` el número de la entrega (1, 2, 3, …). Ejemplo: para Entrega 2 es
`Entrega Parcial 2 - Patricio Gerpe.pdf`. Es el mismo formato del PDF de
Entrega 1 (`Entrega Parcial 1 - Patricio Gerpe (1).pdf`) y lo que espera la
cátedra.

El `.md` se mantiene con el nombre técnico `Entrega_<n>_informe.md` (sin
espacios, ASCII-puro) porque es archivo de trabajo del orquestador
(`entregas/_lib.py::regenerate_informe` y `md_to_pdf`); sólo el PDF final
adopta el nombre humano.

### Cuándo reescribirlo

El informe **solo se reescribe / actualiza cuando una nueva corrida supera a
todas las anteriores en las TRES métricas a la vez**:

- **RMSE CV5 (mean)** — multi-seed (3 seeds × 5 folds) con
  `n_estimators=500, max_depth=50`. Métrica primaria desde v4. El holdout
  único `random_state=42` se descartó porque la CV exploratoria sobre v3
  mostró std ≈ 2 K (mayor que la diferencia entre v1, v2 y v3); el
  multi-seed se agregó en v6 para estabilizar la varianza intra-KFold.
- **RMSE holdout temporal** — último 20 % de las filas con
  `publication_date` conocida, ordenadas cronológicamente. Métrica
  secundaria agregada en v6 después de la lección de v5: KFold aleatorio
  no captura distribution shift entre train y test público porque mezcla
  años uniformemente; un split por fecha sí lo aproxima.
- **RMSE Kaggle** (una vez confirmado vía `--record-kaggle`).

Si una corrida nueva mejora una métrica pero empeora otra, el informe
**no** se toca: ese experimento queda registrado en el `leaderboard.md` y
en su JSON, pero el informe sigue describiendo la mejor combinación
ganadora hasta ese momento. Esto evita reescribir el informe con cada
experimento intermedio y asegura que lo que se entrega refleje la corrida
campeona.

**Excepción — actualización dirigida de "próximos pasos"**: la sección
final del informe (resumen ejecutivo + próximos pasos; en E2 es el
sub-ítem `D. Entrega`) **sí** puede actualizarse manualmente sin que
haya un nuevo campeón cuando un experimento posterior aporta evidencia
diagnóstica accionable para la entrega siguiente. Casos típicos:
mini-ablations que aíslan qué feature rompió una corrida, análisis de
distribución que cuantifican un shift, validaciones cruzadas
adicionales que cambian el plan. El cuerpo del informe (sub-ítems
correspondientes a la consigna) sigue intacto y describiendo al
campeón; sólo se enriquece la sección de cierre con el aprendizaje
nuevo y se reorienta el "qué viene después" a la luz de esa evidencia.
La fuente de los aprendizajes nuevos es la sub-sección
*"Estado de las entregas → Entrega ⟨n⟩ → Lecciones aprendidas"* de este
documento.

### Auto-submit a Kaggle

`entregas/run_entrega.py` decide submit automático así (v6+):

1. La corrida actual debe tener **CV5 multi-seed** (`rmse_cv5_mean`)
   **y** **holdout temporal** (`rmse_holdout_temporal`); si falta
   cualquiera, no submit (queda manual).
2. Tiene que existir al menos un previo con cada una de las dos métricas
   para poder comparar; en la primera corrida del nuevo régimen, no
   submit (queda manual).
3. Submit sólo si MEJORA SIMULTÁNEAMENTE > `MARGEN_CV5` (= 1 500) en
   `rmse_cv5_mean` y > `MARGEN_HOLDOUT_TEMPORAL` (= 1 500) en
   `rmse_holdout_temporal` respecto al mejor previo de cada una. Una
   mejora en una sola métrica nunca dispara submit (caso v5).

### Disciplina experimental (válida para cualquier entrega)

Reglas de proceso, independientes de qué entrega esté en curso. Surgen
del consejo de los "9 errores típicos en Kaggle" filtrado contra las
restricciones del TP (clases ≤ 5, modelo fijo, sin externos):

- **Cambiar máximo 1-2 cosas por experimento.** Si una corrida cambia
  features nuevas + nueva imputación + nuevo Hot Deck a la vez, no se
  puede atribuir éxito o fracaso a ninguna en particular. Cada
  hipótesis aislada se prueba aislada.
- **Baseline congelado = última corrida que mejora Kaggle real.** Esa
  baseline no se pisa en el `leaderboard.md` ni se reescribe el informe
  hasta que aparezca otra corrida que la supere en CV5 multi-seed +
  holdout temporal + Kaggle simultáneamente. En Entrega 2 la baseline
  congelada es **v4** (Kaggle 93 151) hasta nuevo aviso.
- **No sobre-optimizar al leaderboard público.** El leaderboard público
  es ~30 % del test; ganar 500 puntos contra esa muestra puede ser
  ruido. La validación local (CV5 multi-seed + holdout temporal) es la
  fuente de verdad, no el LB público.
- **Procesar train y test en simultáneo, NUNCA mergearlos antes de
  procesar** (regla del profe, generalizable). Dos formas de hacerlo
  bien — adoptamos la segunda:
  1. _Merge train + test → procesar → volver a separar_: válido para
     transformaciones que no dependen del target (parsear `features`,
     normalizar strings, marcar `m2_was_na`). El profe avisa que es
     fácil meter la pata acá: cualquier estadístico ajustado sobre el
     merge (mediana, moda, cuantiles, KNN, encoding) usa información
     del test → leakage. **No usamos esta forma.**
  2. _Procesamiento en paralelo, fit on train + transform on both_:
     todos los estimadores con estado (`SimpleImputer`,
     `StandardScaler`, `IsolationForest`, `KNNImputer`, `factorize`,
     vocabularios de Hot Deck, centroides geográficos, etc.) se
     fittean **sólo con train** y se aplican a ambos. Los pasos sin
     estado (parsing, regex, marcadores) corren con el mismo código
     dos veces, una sobre cada dataframe. **Esta es la forma que
     usamos en v4 de E2 y a partir de ahí.**
  - **Nunca sobreescribir los archivos originales** (`entrenamiento.db`,
    `a_predecir.csv`). Las mutaciones in-place sobre los DataFrames en
    RAM son OK siempre que la lectura se haga al inicio de cada corrida.
  - **Caso límite a vigilar**: el Hot Deck por `description` arma su
    diccionario sobre el train completo. En CV5 eso da al fold de
    validación acceso indirecto al `price` del fold de train del
    propio Hot Deck — sobreestima el rendimiento local. En producción
    (Kaggle test sin price) no hay leak. No urgente; cuando importe,
    armar el Hot Deck dict por fold (más caro). Heredamos el patrón a
    v6+ de E3 (cleanup de barrio cascada Hot Deck → KNN).
- **Outliers ridículos en una celda → impute esa celda, no descartes la
  fila** (regla del profe, generalizable). Si una propiedad tiene
  `n_banos = 800` o `n_cocheras = 1 000`, el resto de las columnas
  (precio, m², barrio, descripción, amenities) sigue siendo
  informativo. Tirar toda la fila por un valor ridículo en una
  columna desperdicia el resto. Patrón a seguir: detectar el rango
  legítimo en el EDA (percentiles, value_counts), winsorizar la
  variable a ese rango, mandar lo que cae fuera a `NaN` y dejar que
  la imputación posterior (mediana / KNN / Hot Deck) lo reemplace.
  Implementación de referencia en v4 de E2 (cell-level): para `m2`,
  `n_dormitorios` y `n_banos`, `df.loc[~col.between(low, high), col]
  = np.nan` (aplicado tanto a train como a test). **Eliminar la fila
  entera** queda reservado para outliers **multivariados**
  estructurales — combinaciones imposibles tipo `price = alto + m2 =
  chico + n_banos = 20` que ningún `NaN` puntual rescata —, y sólo en
  train. La detección multivariada se hace con un algoritmo apropiado
  (IsolationForest, LOF, Mahalanobis), no extendiendo el filtro
  univariado.

### Cómo debe estar redactado

El informe es para un "supervisor" / docente: explica **decisiones**, no
código. Cada decisión técnica que aparezca debe estar **justificada** con
algún ancla concreta:

- **EDA previo**: percentiles, distribuciones, conteos, comparaciones train
  vs test (como las que están en `eda_train_vs_test.py`).
- **Preguntas a los datos**: enunciar la pregunta que motivó el análisis
  (ej: *"¿qué proporción del test queda cubierta si imputamos por
  `description`?"*) y responderla con el número observado.
- **Hipótesis explícita**: *"Si X, entonces el RMSE debería bajar porque Y"*.
- **Experimentos previos y sus resultados numéricos**: baseline vs variante,
  con los RMSE concretos del `leaderboard.md`.
- **Teoría vista en clase**: referenciar el método y por qué corresponde
  (ej: Hot Deck → Clase 4 — Datos Faltantes).

No se aceptan en el informe afirmaciones del estilo *"se eligió la mediana
porque es más robusta"* sin un número o un análisis previo que lo respalde.
Cualquier umbral, hiperparámetro o algoritmo elegido debe poder responder
**¿por qué este valor y no otro?** apuntando a evidencia del JSON del
experimento, del EDA o de un colab/diapositiva de clase.

### Estructura del informe (OBLIGATORIA — debe coincidir con la consigna)

El informe **debe** seguir la **misma denotación y la misma lista de sub-ítems**
que `Consigna Entrega Parcial <n>.pdf` correspondiente a la entrega en curso.
La consigna puede usar letras (A / B / C / …), números romanos, o cualquier
otra numeración; el informe replica esa numeración **tal cual**, sin inventar
una propia.

Reglas estructurales aplicables a **cualquier entrega**:

1. **Los encabezados del informe copian los de la consigna**. Si la consigna
   de esta entrega dice "A. Datos Atípicos (outliers)", el informe tiene
   "## A. Datos Atípicos (outliers)". Nada de renombrar a "1) Contexto",
   "2) Filtros", etc. Si hay contenido que no encaja en ningún punto
   explícito de la consigna (contexto, objetivo, filtros heredados de la
   entrega anterior), va en una introducción breve **antes** del primer
   punto, o se integra dentro del punto donde sea más natural.
2. **Cada sub-pregunta explícita de la consigna tiene que responderse de
   forma directa** — con el nombre concreto de la variable, causa, umbral o
   valor que corresponda, respaldado por algún número del JSON de la
   corrida campeona o del `leaderboard.md`. Si la consigna hace una
   pregunta retórica tipo "¿cuál es el problema cuando…?", hay que
   contestarla en el sub-ítem correspondiente, no diferirla.
3. **El punto de la consigna que pide comparar contra la entrega anterior**
   (suele haberlo: en E2 es "C. Modelo (Predicción)") se resuelve con una
   tabla **Entrega (n-1) vs Entrega (n)** usando los scores Kaggle reales
   — ver *"Estado de las entregas → Entrega (n-1) → Scores Kaggle"* de este
   documento — y una narración breve de por qué cambió lo que cambió. Las
   comparaciones entre versiones *dentro* de la misma entrega son contexto
   secundario, no el foco.
4. **Tono**: primera persona del **SINGULAR** (el TP es individual). Ver
   regla 4 abajo.
5. **Toda** decisión técnica tiene que estar anclada en números del JSON de
   la corrida campeona o del `leaderboard.md`. Nada de afirmaciones
   genéricas ("es más robusta", "mejora el score") sin dato.
6. **Longitud**: ≤ 1 carilla (≈ 500 palabras sin contar la/s tabla/s).
7. **Párrafo introductorio (aplica a cualquier entrega)**: el informe
   abre con un párrafo breve, antes del primer punto numerado de la
   consigna, que cubre dos cosas: (i) **de dónde parto** — qué pipeline
   o decisiones heredo de la entrega anterior — y (ii) **si ajusté
   parámetros heredados** (filtros, hiperparámetros, criterios), los
   listo con sus números (ej. *"amplié el filtro X recuperando +N filas
   (+M %)"*). Si no hubo ajustes respecto de la entrega anterior, lo
   digo explícitamente (*"mantengo idénticos los filtros de la
   Entrega ⟨n-1⟩"*) en lugar de omitirlo. La fuente de verdad para los
   deltas siempre es la sub-sección *"Estado de las entregas →
   Entrega ⟨n⟩ → Delta respecto a Entrega ⟨n-1⟩"* de este documento;
   si esa sub-sección no existe para la entrega en curso, asumir que no
   hubo deltas y decirlo así.

### Tono

Primera persona del **SINGULAR** (*decidí, implementé, mi estrategia,
esperaba, aprendí, analicé, observé, elegí*). Evitar:

- Voz pasiva impersonal (*se implementó, se procedió a*) cuando no agrega
  información sobre quién decide.
- Primera persona del **plural** (*decidimos, nuestra estrategia, aprendimos,
  observamos*): el TP lo firma una sola persona, el plural suena impostado.

### Narrativa y storytelling (regla crítica)

El informe **NO** es un catálogo de acciones ("hice X, después hice Y"). Es
una **historia** donde cada decisión aparece como consecuencia de lo anterior.
El lector (supervisor) tiene que poder reconstruir el **proceso de pensamiento**:
qué se observó, qué pregunta disparó, qué hipótesis se formuló, qué se
probó, qué falló, qué se aprendió del fracaso, y qué se probó después.

**Elementos narrativos que cada sub-ítem (cualquiera sea su numeración) debe
combinar**, sin necesidad de usarlos todos ni en orden fijo:

- **Hallazgo** — qué se vio en los datos, con número concreto (ej. percentil,
  conteo, porcentaje de nulos).
- **Pregunta** — qué duda disparó ese hallazgo (*"eso me hizo preguntarme
  si…"*, *"empecé a sospechar que…"*).
- **Hipótesis** — *"si X, entonces el RMSE debería bajar porque Y"*.
- **Experimento y resultado** — qué versión del pipeline probó esa hipótesis
  y qué pasó, citando la fila del `leaderboard.md` que corresponde.
- **Aprendizaje del fracaso** — las hipótesis **refutadas** (corridas anteriores
  que empeoraron el score) son material narrativo de primera clase, no
  ruido. Tienen que aparecer integradas en la historia explicando por qué
  motivaron la decisión siguiente.
- **Conexión con el próximo paso** — cómo lo aprendido justifica lo que
  viene en el sub-ítem siguiente.

**Conexión entre sub-ítems**: cada nuevo bloque arranca enganchando con el
anterior. Conectores aceptables: *"Eso me llevó a…"*, *"Con ese problema
resuelto, me encontré con…"*, *"El fracaso de v⟨k⟩ me enseñó que…"*, *"Esa
pista me empujó a probar…"*. La prueba de fuego: si el lector puede permutar
el orden de los sub-ítems sin romper la lectura, la narrativa está rota.

**Señales de redacción robótica que el informe NO debe tener**:

- Bullets adentro de un sub-ítem cuando el contenido se puede contar como
  párrafo corrido. Los bullets fragmentan la narrativa; reservarlos para
  listas estrictamente enumerativas (p. ej. "los 3 tipos de operaciones
  consideradas" o "las 4 columnas imputadas con mediana").
- Cadenas de oraciones cortas declarativas sin conectores
  (*"Analicé X. Observé Y. Apliqué Z."*).
- Referencias parentéticas al JSON pegadas al final de cada frase sin
  integrarlas en el discurso. Preferir "*terminé con 122 383 filas*" (número
  dentro de la oración) sobre un paréntesis "*(JSON v4: `data.train_post_outliers`)*"
  que repite lo mismo.
- Encabezados con variación estética pero sin sustancia ("Análisis", "Marco
  conceptual", "Resultados obtenidos") cuando la consigna ya define la
  estructura.

### Ejemplo ilustrativo (referencia genérica)

Para fijar la diferencia entre un informe "catálogo" y uno narrativo, este
es un ejemplo **tomado de Entrega 2**, pero la estructura se replica en
cualquier entrega con sub-ítem de tratamiento de outliers:

❌ MAL (catálogo descriptivo, sin thought process):

> A.1 Analicé atípicos en variables numéricas. Observé colas extremas en
> superficie/ambientes. En Random Forest estos puntos dominan splits.
>
> A.2 Implementé una estrategia mixta: capado a NaN + IsolationForest
> con `contamination=0.01`. Eliminé 719 filas.

✅ BIEN (hallazgo → pregunta → hipótesis → experimento → aprendizaje):

> A.1 Al mirar las distribuciones de `m2`, `n_dormitorios` y `n_banos` me
> llamó la atención que el percentil 99.9 de `m2` era ~1 500 m² pero había
> valores de 40 000 y hasta 10⁶, imposibles para un departamento. Sospeché
> que venían de errores de parseo del texto de `features`, no de
> propiedades reales, así que me pregunté si el problema era "columna
> sucia" o "fila sucia".
>
> A.2 Probé primero lo más agresivo — borrar la fila — pero perdía señal
> en barrios chicos donde ya tenía pocas observaciones. Eso me llevó a
> separar dos problemas: errores de atributo, que resolví winsorizando
> `m2 ∈ [10, 1500]` y mandando el resto a NaN para que la imputación de
> B.2 los manejara; y filas globalmente incoherentes, donde el outlier no
> es de una sola variable. Para estos últimos usé IsolationForest sobre
> `(price, m2, n_dormitorios, n_banos)` escalados, con `contamination=0.01`
> porque el EDA mostraba una minoría clara de filas raras, no un 5 %
> sospechosamente alto, y terminé eliminando 719 filas del train.

El template se regenera automáticamente cuando el auto-submit declara un
nuevo campeón; por eso estas reglas viven acá para que cada regeneración
las respete.

### Sección operativa: parámetros específicos de la entrega en curso

Esta sub-sección es la **única** parte que cambia entrega a entrega y la que
el prompt del LLM debe leer para particularizar las reglas genéricas de
arriba. Hay que actualizarla antes de cada entrega nueva.

**Entrega 2 — parámetros del informe**:

- **Denotación de la consigna**: literales `A / B / C / D` con sub-ítems
  `A.1, A.2, A.3, B.1, B.2, B.3`.
- **Encabezados que debe usar el informe**:
  `## A. Datos Atípicos (outliers)`, `## B. Datos Faltantes`,
  `## C. Modelo (Predicción)`, `## D. Entrega`.
- **Sub-preguntas explícitas de la consigna que requieren respuesta directa**:
  - **A.3** — "¿Cuál variable presenta mayor cantidad de outliers? ¿Qué
    enfoque se considera adecuado para tratar dicha columna?"
  - **B.3** — "En caso que una técnica de imputación suba el error ¿cuál
    se considera que es el problema?"
- **Punto que exige comparación con la entrega anterior**: **C. Modelo**.
  Comparar contra Entrega 1 (mejor Kaggle E1 = 166 979.008,
  `solucion (12).csv`; el detalle completo está en *Estado de las
  entregas → Entrega 1 → Scores Kaggle*).
- **Restricción técnica de la consigna**: el código sólo puede tratar
  outliers, datos faltantes y crear nuevos atributos. No tocar el bloque
  de modelo del notebook (`n_estimators=500`, `max_depth=50`).
- **Arco narrativo esperado** (guía, no libreto):
  - Introducción breve: qué filtro heredé de Entrega 1 y por qué se
    mantiene.
  - A: del EDA de outliers a la estrategia de tratamiento, respondiendo
    A.3 dentro del arco.
  - B: enganchando con los NaN que dejó el tratamiento de A, contar las
    imputaciones elegidas; en B.3 usar las hipótesis refutadas (corridas
    que empeoraron Kaggle, ver `leaderboard.md`) para responder por qué
    una imputación puede subir el error.
  - C: la tabla **E1 vs E2** se lee como cierre de todo lo anterior.
  - D: resumen ejecutivo y próximos pasos, breve.

## Directiva de calidad técnica

Toda decisión técnica en el código del TP — elección de métrica, algoritmo, modelo, herramienta de análisis, umbral numérico o parámetro — **debe estar justificada** a partir de análisis previos (EDA, percentiles, distribuciones, comparación de alternativas) o de la teoría vista en clase. No se acepta elegir valores o métodos de manera trivial o arbitraria.

Ejemplos de lo que NO es aceptable:
- Usar `contamination=0.01` sin referenciar la proporción de outliers observada en el EDA.
- Definir umbrales como `[10, 1500]` para m² sin mostrar percentiles que los respalden.
- Elegir mediana sobre moda sin explicar por qué es más adecuada para esa variable.
- Tener una función definida que nunca se usa (código muerto).
- Parsear texto con regex demasiado genérico sin verificar que no produzca falsos positivos.

Cuando Claude revise o escriba código para este TP, debe verificar activamente que cada decisión técnica esté anclada en evidencia del datos o en fundamentos teóricos de las clases.

### Reglas de integridad del notebook

- **Celdas marcadas ⛔ NO TOCAR**: no modificarlas bajo ninguna circunstancia.
- **Técnicas permitidas**: solo se pueden usar algoritmos, métricas y herramientas vistas en las clases disponibles para cada entrega. Para la Entrega 2: filtrado, outliers (IQR, Z-score, IsolationForest, LOF), datos faltantes (SimpleImputer, KNNImputer, MICE, MissingIndicator, Hot Deck, Cold Deck) y creación de atributos desde los datos propios. Nada de librerías externas no vistas ni datos externos.

## Flujo de trabajo (orquestador + logging)

**Regla central**: la única fuente de verdad del código es el notebook
`Colab_Base_para_el_Trabajo_Práctico_(Entrega_<n>).ipynb`. **No** se mantiene
ningún `.py` paralelo que duplique la lógica del notebook (eso garantiza que lo
que se valida localmente es exactamente lo que se sube al Colab del docente).

### Convención de notebooks por entrega

Para que un experimento que falla **no** pise el notebook ganador, cada
entrega mantiene tres notebooks distintos:

| Rol | Ubicación | Cuándo se actualiza |
|---|---|---|
| **Inicial** | `entregas/entrega_<n>/notebook_inicial.ipynb` | Una sola vez, al recibirlo del profe (se congela). |
| **Campeón** | `entregas/entrega_<n>/notebook_campeon.ipynb` | Sólo cuando aparece un nuevo campeón Kaggle (= mismas reglas que el informe; ver *"Informe de entrega → Cuándo reescribirlo"*). |
| **Último** | `Colab_Base_para_el_Trabajo_Práctico_(Entrega_<n>).ipynb` (raíz) | Constantemente, en cada experimento. Es el que ejecuta el orquestador. |

Reglas:

- El **campeón** es lo que se sube a la cátedra cuando se pide la
  notebook. Tiene que ser la versión exacta que generó el CSV ganador,
  con outputs limpios pero código intacto.
- La fuente de la versión campeona es siempre el `entregas/entrega_<n>/<entrega>_<vCampeon>.executed.ipynb`,
  que el orquestador genera automáticamente con cada corrida y queda
  como auditoría. Limpiarlo (sacar `outputs` y `execution_count`) y
  copiarlo a `notebook_campeon.ipynb`.
- El **último** evoluciona libremente con experimentos posteriores. Si
  un experimento rompe el modelo o introduce *distribution shift*, el
  campeón sigue intacto y se puede entregar sin reconstruir nada.
- La promoción de un experimento de "último" a "campeón" la dispara el
  auto-submit de `run_entrega.py` (en paralelo con la regeneración del
  informe). Si por alguna razón se hace manual, hay que regenerar
  también el `notebook_campeon.ipynb` desde el `.executed.ipynb`
  correspondiente para mantener la trinidad consistente.

### Cómo se corre una entrega localmente

```powershell
python entregas/run_entrega.py --entrega entrega_2 --nombre v2 \
    --desc "Sin IsolationForest, KNNImputer en m2, n_estimators=800"
```

`entregas/run_entrega.py` ejecuta el notebook con `nbconvert` (sin tocarlo) y
le pasa parámetros vía variables de entorno:

- `EXPERIMENT_ENTREGA` → carpeta destino bajo `entregas/` (ej: `entrega_2`).
- `EXPERIMENT_NAME`    → identificador corto del experimento (sufijo de archivos).
- `EXPERIMENT_DESC`    → texto libre que también se usa como **Submission
  Description** al subir a Kaggle.

El notebook tiene cerca del inicio una "celda de parámetros" que lee estas
variables (con defaults para correrla a mano) y arma un `dict` global llamado
`EXPERIMENT_LOG` que se va llenando paso a paso.

### Outputs por corrida (en `entregas/<entrega>/`)

- `solucion-<entrega>-<nombre>.csv` — submission para Kaggle (`id,price`).
- `solucion-<entrega>-<nombre>.json` — metadatos completos del experimento:
  filtros, umbrales de outliers, % de Hot Deck cubierto, hiperparámetros,
  RMSE de holdout, percentiles de las predicciones, md5 del CSV, mensaje
  sugerido para Kaggle, etc.
- `<entrega>_<nombre>.executed.ipynb` — notebook ya ejecutada con outputs
  (auditoría / debugging).
- `leaderboard.md` — tabla acumulada con una fila por corrida; la columna
  **RMSE Kaggle** queda como `_pendiente_` hasta que se anote a mano.

### Registrar el resultado real de Kaggle

Tras subir el CSV, anotar el RMSE devuelto por Kaggle (lo escribe en el JSON
y reemplaza el `_pendiente_` del leaderboard):

```powershell
python entregas/run_entrega.py --record-kaggle --entrega entrega_2 \
    --nombre v1 --kaggle-rmse 93167.324
```

### Resultados ya registrados

| entrega | nombre | RMSE holdout | RMSE Kaggle |
|---|---|---:|---:|
| entrega_2 | v1 | 118 866.97 | **93 167.324** |

### Notas operativas

- El notebook se ejecuta tal cual también en Colab (la celda de drive detecta
  `IN_COLAB` y usa `DIR = "/content/drive/MyDrive/datos/propiedades"`).
- Trabajo en **Windows + PowerShell**, encoding `utf-8` (forzar
  `$env:PYTHONIOENCODING="utf-8"` cuando se imprimen acentos).
- Datos pesados: **no commitear** `entrenamiento.db` (~2 GB) ni los `.executed.ipynb`.
- Dependencias locales para correr el orquestador: `nbconvert`, `nbformat`,
  `ipykernel`, `matplotlib`, `seaborn`. Kernel `python3` instalado vía
  `python -m ipykernel install --user --name python3`.
- Una corrida completa tarda ~10–15 min (cuello: dos `fit` del
  RandomForestRegressor con `n_estimators=500`, `max_depth=50`).
