# Contexto del Trabajo Práctico — Data Mining (UBA Exactas)

**Resumen ejecutivo del avance del TP:** [`estado_actual.md`](estado_actual.md).

**Política de mantenimiento:** cada vez que haya avances relevantes del TP
(nuevo campeón en Kaggle, cambio de entrega en curso, decisiones metodológicas
que muevan el foco), actualizar **`estado_actual.md`** con un snapshot breve
(tabla de entregas, campeón vigente, métricas clave) y sumar una línea al
historial del mismo archivo. Este `CONTEXT.md` sigue siendo la fuente canónica
de detalle (lecciones, backlog, consignas); el otro archivo existe para que
agentes y lectores rápidos vean el estado sin hojear novecientas líneas.

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

### Backlog de ideas para entregas futuras

Lugar para acumular ideas que aparecen en el medio del trabajo de una
entrega y NO encajan en la entrega actual (por restricción técnica o
porque el foco temático es otro). El objetivo es no perderlas y que
cuando arranque la entrega que sí las habilita ya estén con justificación
escrita.

| Idea | Entrega objetivo | Por qué no encaja antes | Estado |
|---|---|---|---|
| Embeddings de `description` con `sentence-transformers` (ej. `paraphrase-multilingual-MiniLM-L12-v2`) → reducción a ~20 dims con PCA → features para el RF | **Entrega Final** (o E4 si el alcance lo permite) | E3 sólo permite librerías y técnicas vistas en clase ≤ 8. `sentence-transformers` no es librería de clase y los pesos pre-entrenados violan la regla "sin datos externos" (el corpus de pre-training casi seguro contiene avisos de Properati / Zonaprop). E4 abre la puerta de "datos no estructurados", entrega final permite integración completa. Diagnóstico complementario: la versión "regex frágil → embeddings densos" sobre-atribuye el fallo de v5 (E2) a la fragilidad del regex; el error analysis mostró que el shift de v5 vino mayormente de `pub_year`/`pub_month`, no de las features textuales. Igual la idea de comprimir semántica de `description` con embeddings densos sigue siendo de las más prometedoras una vez levantada la restricción. | Pendiente. Versión "ML clásica" defendible en E3 si clase 8 lo banca: `TfidfVectorizer(ngram_range=(1,2), min_df=20, max_features=2000)` + `TruncatedSVD(n_components=20)` sobre `description_norm`. Misma forma del pipeline (vectorizar → reducir → features), técnicas dentro del scope. Decidir cuando llegue el material de clase 8. |
| **RAG semántico de comparables**: para cada fila, embebés `description + features estructuradas`, buscás los K vecinos más cercanos en train con FAISS, y el `precio_mediano_ponderado_vecinos` entra como feature al RF. Es la versión semántica del Hot Deck (similitud densa en vez de match exacto de texto). | **Entrega Final** (o E4 si el alcance lo banca) | Mismo problema de embeddings + librería externa (FAISS no es de clase). **Trampa crítica que NO mencionar como "zero leakage"**: es target encoding y necesita CV-aware. Para cada fold de train, los K vecinos deben venir del train *fuera del fold* (out-of-fold encoding al estilo `category_encoders.TargetEncoder`); si se hace naive (K vecinos en TODO el train, incluida la propia fila), el CV5 se infla artificialmente y se repite el desastre de v2 (`precio_mediano_barrio` con +3 740 RMSE en Kaggle). Para test sí es seguro siempre (no hay target). | Pendiente. Implementación correcta: KFold sobre train, para cada fold buscar vecinos sólo en los otros 4 folds, generar la feature out-of-fold, refittear FAISS sobre todo el train para inferir sobre test. |
| **LLM structured extraction batch (Llama 3 / Mistral local con Ollama)**: un LLM local procesa cada `description` y devuelve un JSON estructurado con `{floor, view_type, renovation_year, parking_covered, storage_room, orientation, calidad_materiales, ...}`. Captura todo lo que v5 intentó con regex frágil pero con comprensión semántica, sin costo de API y sin enviar datos afuera. | **Entrega Final** (puede caber en E4 si el enunciado lista LLMs locales como herramientas habilitadas) | Llama / Ollama son modelos externos (mismo argumento que embeddings). 130 K avisos × Llama 3 8B en CPU = ~30-50 horas; con GPU consumer ~3-5 h; con sampling de 20 K avisos representativos (estratificado por `barrio × tipo × decil_precio`) y propagación al resto por kNN sobre embeddings, ~30 min. Riesgo a controlar: el JSON puede salir inconsistente entre filas (mismo campo con valores en distintos formatos / idiomas). | Pendiente. Implementación segura: schema-enforced con `instructor` / `outlines` / `guidance` (forzar JSON válido contra un Pydantic model), `temperature=0` para reproducibilidad, validación post-hoc y reportar `% de filas con extracción exitosa` por feature. Persistir resultados a parquet (no llamar al LLM de nuevo si ya está hecho). |
| **Normalización temporal del precio con FX paralelo / índice CABA**: el dataset cubre oct-2021 a jun-2026, período de fuerte variación del dólar paralelo en Argentina. `price_USD_normalizado = price_USD / fx_paralelo[publication_date]` antes de entrenar; al predecir, multiplicar por `fx_paralelo[publication_date_test]` para volver a la escala original. Variante más simple: agregar `fx_paralelo_at_pub`, `fx_oficial_at_pub`, `inflacion_acum_12m_at_pub`, `tasa_badlar_at_pub` como features adicionales y dejar que el RF aprenda solo cómo usarlas. | **Entrega 4** | Requiere dataset externo (serie histórica del dólar paralelo / blue / MEP, índice CAMARA / RECC / INDEC). E4 habilita "APIs y web scraping" → series de BCRA, ámbito.com, Bluelytics son APIs públicas legítimas. **Es la única estrategia que ataca de raíz el shift temporal** que el `error_analysis_v4` confirmó como problema real (holdout entera en ene/jun 2026, train mayormente 2021-2024 con dólar muy distinto). | Pendiente. Implementación con cuidado de inversa: si normalizás dividiendo, **multiplicás al predecir** (regla del CONTEXT.md sobre target transforms aplicada a normalización por feature externa). Necesitás la serie de FX hasta jun-2026 inclusive. Empezar por la variante "FX como feature" (más simple, menor superficie de error) y comparar contra normalización completa. |
| **Biblioteca de corrección de data drifting (port del workflow de Lab II)**: 7 métodos parametrizados por período para neutralizar el shift temporal de variables monetarias. Métodos sin datos externos: `rank_simple` (rank percentil [0,1] por período), `rank_cero_fijo` (variante que mantiene 0→0), `estandarizar` (Z-score por período). Métodos macro (requieren tabla de índices): `deflacion` (×IPC), `dolar_oficial` / `dolar_blue` (÷FX del período), `uva` (×UVA). | **Entrega Final** (los 3 métodos sin datos externos podrían colarse antes; los macro caen en E4 junto con la idea anterior) | Es la implementación concreta de la idea anterior pero generalizada a cualquier feature monetaria, no sólo `price`. Para los métodos `rank_*` y `estandarizar` no hay restricción de datos externos; lo que NO encaja en E3 es la decisión de qué normalización aplicar (sobre el target o sobre features) y la falta de tabla macro lista. | Starter ya armado en `entregas/entrega_f/drift_correct.py` (port directo del R `z1401_DR_corregir_drifting.r` de la materia hermana, con el ajuste de fittear estadísticas SOLO en train). Stub del YAML de índices en `entregas/entrega_f/indices_macro_arg.yml` documentando fuentes (INDEC, BCRA, Bluelytics). En EF: completar series + comparar RMSE en holdout temporal de cada método contra baseline sin corrección. Decisión empírica, no teórica. |
| **Diagnóstico cuantitativo + visual de data drifting**: complemento al starter de corrección. Para cada feature, calcular PSI / KS / Wasserstein / JS divergence comparando "train viejo vs train nuevo vs test", + CDFs superpuestas en escala `sign(x)·log2(|x|+1)` (la transformación del script R original que comprime colas sin perder signo). Output: ranking de features por magnitud de drift + PDF con gráficos por feature. | **Entrega Final** (idealmente como paso 1 antes de aplicar `drift_correct.py`) | Sin datos externos en sí, pero requiere `scipy.stats` (KS, Wasserstein) y `matplotlib` para los CDFs — todas estándar pero la decisión metodológica de "diagnóstico antes de corrección" pertenece más a EF que a E3. | Pendiente. Workflow ideal: `drift_detect.py` → "estas N features driftean" → `drift_correct.py` → dataset corregido por método → comparar RMSE en holdout temporal. El R original (`densidades_<mes0>_<mes1>.pdf`) hace una versión visual; acá agregaría la cuantitativa con métricas de divergencia. |
| **Creacionismo: search evolutivo de features con canary pruning** (contribución original del autor en Lab II — Economía y Finanzas, materia hermana). Algoritmo iterativo: (1) ajustar modelo, ranquear features por importancia; (2) tomar top-20 y generar TODOS los pares cruzados con `+ - * /` → ~840 features candidatas; (3) inyectar features completamente random ("canaritos") al dataset, medir su importancia, y descartar TODA feature humana con importancia menor que `median(importancia_canaritos) + N·std` — control empírico de hipótesis nula contra ruido puro; (4) opcionalmente agregar leaf indices del modelo como features one-hot; (5) iterar. Encuentra interacciones de orden N sin enumeración manual y poda con criterio cuantitativo en lugar de intuición. | **Entrega Final** | E3-E4 está totalmente fuera de scope: es un meta-algoritmo de búsqueda de features que va más allá de lo enseñado en clases 7-8 (transformaciones, discretización, atributos derivados manuales). En EF tiene peso narrativo extra: **extender una contribución original propia** (publicada en colaborativo del Lab II) adaptándola de clasificación binaria de churn a regresión continua de precios. | Pendiente. Adaptaciones requeridas vs. el original R: (a) RF / LGBM regresión en lugar de LGBM binario, (b) `feature_importances_` de sklearn en lugar de "ganancia con meseta" custom, (c) lista negra explícita de variables target-derived antes de cruzar — evita el desastre v2/v3 de filtrar `precio_mediano_barrio` por división, (d) canary evaluado multi-seed o multi-fold para garantía estadística sobre miles de candidatas (el original single-shot puede dejar pasar features que ganan por azar), (e) manejo explícito de Inf por división por 0 (en R el script avisa con `NaN → 0` que es "decisión polémica"; en Python: `replace([Inf,-Inf], NaN)` + dejar que RF maneje), (f) recalibrar `min_data_in_leaf` y demás HP del modelo interno para nuestras ~80 K filas filtradas (el original está tuneado para ~150 K filas bancarias). Sinergia con resto del backlog: alimentar Creacionismo con embeddings densos (cruces entre dims latentes y features estructuradas que no se enumerarían a mano) o con features rolling por barrio (cruces tipo `precio_actual / avg6m_barrio`). |
| **Features históricas rolling por barrio / publisher con máscara causal**: para cada `(barrio, yyyymm)` precalcular `precio_m2_mediano`, `count_publicaciones`, `precio_min/max/avg`, `tendencia_6m` (pendiente OLS de los últimos 6 meses), `ratio_actual_vs_avg6m`, `ratio_actual_vs_max6m`. Para cada listing, attachear esas features evaluadas en su `publication_date`. Análogo para `(publisher_id, yyyymm)` (captura "inmobiliaria de lujo" vs "mass market"). Inspirado en `z1501_FE_historia.r` de la materia hermana de Economía y Finanzas (lags, tendencias, min/max/avg sobre series temporales de clientes bancarios), reformulado al cambiar la entidad ("propiedad" → "barrio" / "publisher") porque en nuestro problema cada fila es una publicación única, no un activo seguido en el tiempo. | **Entrega Final** | El port literal del script R (lag por propiedad) NO aplica: ~86 % de las propiedades aparecen una sola vez, y el ~10-14 % que se repite son los "price tests" del `v10_dedup` que justamente queremos colapsar — usar el lag entre ellas como feature contamina con leakage estilo v2. La reformulación (entidad = barrio / publisher) sí tiene la estructura `(entidad, período)` densa que necesita el algoritmo. **Trampa crítica (que el R no advierte porque churn binario no la sufre): es target encoding temporal y requiere máscara causal**. Para una fila publicada en mar-2024, las rolling stats deben usar SOLO datos `< mar-2024`; si incluyo el propio mes infla CV5 y rompe Kaggle (clásico de v2 / v3 con `precio_mediano_barrio`). Para test (2026) sí puedo usar TODO el train porque está todo en el pasado. Por la complejidad del diseño causal correcto + dependencia de barrios limpios (`v6_clean_barrio` debe estar resuelto), entra en EF. | Pendiente. Implementación segura: para cada listing en mes M usar expanding window sobre `train.publication_date < M`, con barrios con < N publicaciones en la ventana fallback al promedio CABA del período. Es el **complemento natural de `drift_correct.py`**: drift_correct neutraliza el shift macroeconómico del precio absoluto, las features rolling le dan al modelo la señal del trend local barrio-a-barrio que sí queremos preservar. Comparar contra baseline: ¿la información temporal del barrio aporta sobre lo que ya capturan los centroides geográficos (`v7_centroides`)? |
| **Geocoding enriquecido vía OSM Overpass + datos abiertos GCBA/INDEC**: para cada `(lat, lon)`, queries a OSM Overpass para sacar features de entorno (distancia al subte / colectivo más cercano, cantidad de escuelas / hospitales / comercios / parques en radio de 500 m), + match por radio censal contra datasets abiertos del GCBA / INDEC (nivel socioeconómico, densidad poblacional, m² verde por habitante, índice de inseguridad). | **Entrega 4** | Requiere APIs externas + datasets externos. E4 está literalmente diseñada para esto ("APIs y web scraping + datos geográficos"). Ataca directamente el problema "Palermo Soho ≠ Palermo Chico" del profe **sin necesidad de embeddings ni LLMs**, con datos auditables y reproducibles. Probablemente el mejor ROI (señal nueva / esfuerzo) de toda la lista para E4. | Pendiente. Una sola corrida cacheada por `(lat, lon)` única → ~50 K queries a OSM Overpass throttled = 1-2 días en background, después se reusa para siempre. Persistir a parquet. Las features OSM son inmediatamente interpretables → fáciles de defender en el informe. |
| **Visión por satélite / Street View con VLM pre-entrenado (CLIP / DINOv2 / SigLIP)**: con `(lat, lon)`, descargar tile de Mapillary o Sentinel-2 (gratis, públicos), extraer embedding visual del entorno con un VLM pre-entrenado, comprimir a ~10 dims con PCA, agregar como features al RF. Captura "edificios viejos", "manzana arbolada", "zona comercial", "skyline alto" — información que no está en ningún campo estructurado ni en `description`. | **Entrega Final** | Modelos externos pre-entrenados + datos visuales externos. Por costo computacional + complejidad de pipeline + tamaño del dataset (130 K imágenes a descargar y procesar) sólo entra en la última entrega. | Pendiente. Empezar con muestreo agresivo (1 imagen por barrio limpio, propagar por similaridad geográfica) antes de escalar. Caché obligatorio. Auditar visualmente N imágenes random para confirmar que la API devuelve algo útil para CABA (Mapillary tiene cobertura desigual). |
| LLM (Claude / GPT) razonando aviso por aviso para estimar precio y usar la estimación como feature | **Entrega Final** | Doble problema: (a) "sin datos / modelos externos" + costo de API sobre 130 K filas (impráctico salvo subsampling); (b) **riesgo serio de memorización**: Claude / GPT entrenados con CommonCrawl post-2024 muy probablemente vieron Properati / Zonaprop / Argenprop, y plausiblemente memorizaron precios listados de los avisos exactos de este dataset. La "prior" puede no ser razonamiento del LLM sino recall del precio real → feature semi-leak indistinguible de leakage real. | Mantener como **exploración honesta** en la entrega final, NO como feature de producción. Si se prueba, reportar con transparencia: "intenté esto, mejora X, riesgo de memorización Y, lo descarto/lo dejo con esta justificación". El profe va a valorar más la honestidad analítica que el RMSE final. |

Regla de uso del backlog: cada vez que aparezca una idea que pinta bien
pero no se puede usar ahora, se agrega una fila acá con (1) la entrega
en la que sí encajaría, (2) por qué no encaja antes y (3) el estado /
versión "downgrade" defendible en la entrega actual si la hay. Cuando
arranca la entrega objetivo, se revisa el backlog antes de planificar.

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

## Referencia de scores y benchmarks (Kaggle, RMSE)

Son **tres referencias distintas** — conviene no mezclarlas al evaluar avance:

| Rol | Valor (referencia) | Uso |
|---|---|---|
| **Benchmark público (leaderboard)** | **62 821.209** | Mejor RMSE **público** en la competencia a **2026-04-20**. Techo orientativo del *dataset* y del estado del arte visible en Kaggle; **no** es consigna de la cátedra ni criterio de aprobación. Actualizar el número si el tope del leaderboard cambia. |
| **Campeón propio** | **91 399** (v2, Entrega 3) | Baseline de trabajo e informe; ver tabla *Resultados registrados* abajo. |
| **Umbral de aprobación por entrega** | Robot de la entrega | Hay que **ganarle al robot** en Kaggle; el RMSE exacto del robot **E3** está **pendiente** hasta publicación (ver *Robots de la cátedra*). |

- No es obligatorio alcanzar el benchmark público; no forzar decisiones
  técnicas para perseguirlo si comprometen la claridad o la justificación del TP.

### Resultados registrados (entrega_2)

| versión | RMSE CV5 (mean ± std) | RMSE holdout temporal | RMSE holdout | RMSE Kaggle | estado |
|---|---:|---:|---:|---:|---|
| v1 | — | — | 118 867 | 93 167 | |
| v2 | — | — | 116 923 | 96 891 | leakage: barrio price stats |
| v3 | — | — | 117 080 | 97 423 | leakage: lat/lon centroide barrio |
| **v4** | **117 219 ± 2 392** | — | 118 867 | **93 151** | campeón E2 (entregado) |
| v5 | 112 341 ± 1 659 | — | 113 952 | 97 498 | distribution shift: temporales/booleanos/floor |
| v6 | 112 304 ± 2 794 | 122 795 | 113 952 | — (no submit) | diagnóstico: multi-seed CV5 + holdout temporal + mini-ablation v5 |

Umbral de submit (v6+, doble criterio): mejora simultánea > 1 500 en
`rmse_cv5_mean` (multi-seed) **y** > 1 500 en `rmse_holdout_temporal`. El
holdout único queda como columna informativa, no decide.

### Resultados registrados (entrega_3)

| versión | RMSE CV5 (mean ± std) | RMSE holdout temporal | RMSE holdout | RMSE Kaggle | estado |
|---|---:|---:|---:|---:|---|
| v1 | 116 669.79 ± 2 521 | 117 235.85 | 118 866.97 | 93 147 | baseline E2 pipeline + CV5 multi-seed |
| **v2** | **112 776.86 ± 2 173** | **111 536.16** | 116 706.86 | **91 399** | **campeón actual** — parseo dual + rooms |
| v3 | 112 884.00 ± 2 185 | 111 555.58 | 116 821.38 | pendiente | log-transforms — NO mejora (RF invariante a transforms monótonas) |
| v4 | — | — | — | — | en curso (barrio cleanup) |

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

### Entrega 2 — ENTREGADA

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

### Entrega 3 — EN CURSO

#### Lecciones aprendidas

- **v1 → baseline E3 + CV5 multi-seed**. Primera corrida E3 con el pipeline exacto de E2-v4 más la infraestructura nueva (CV5 multi-seed 3 seeds × 5 folds, holdout temporal, ResumableRunner). Kaggle 93 147 — alineado con E2-v4 (93 151), confirma que la infraestructura no introdujo ruido. HP fijos en `n_estimators=500`, `max_depth=50` (sweep habilitado pero no corrido todavía).
- **v2 → parseo dual + rooms (CAMPEÓN ACTUAL)**. Fallback de `features` a `description` para m2/dormitorios/baños cuando NaN; agrega `rooms` (ambientes desde patrón `X amb`). Rescata ~1 192 m2, ~88 dormitorios, ~186 baños en train. CV5 baja 3 893 puntos, holdout temporal baja 5 700 puntos — ambos por encima del umbral de 1 500. Kaggle: **91 399** (−1 748 vs campeón anterior). Lección: recuperar valores reales de m2 (en lugar de mediana imputable) mejora los splits del RF en la variable de mayor importancia — efecto real, no artefacto de validación.
- **v3 → log-transforms (DESCARTADO)**. `log1p(m2, n_dormitorios, n_banos, len_descripcion, n_features)` como columnas nuevas. CV5 empeoró ligeramente (112 884 vs 112 777, delta +107). **Aprendizaje clave**: los árboles de decisión son invariantes a transformaciones monótonas de las features — el split-finding ya encuentra los umbrales óptimos en cualquier escala. Agregar columnas `log1p_*` junto a las originales solo aumenta la dimensionalidad sin añadir información nueva. Log-transforms NO aplican a RF (sí aplican a regresión lineal, redes neuronales, etc.).
- **v4 → barrio cleanup (en curso)**. Cascada: Hot Deck por descripción normalizada → KNN(lat/lon, k=5) → "desconocido"; colapso barrios ≤10 obs → "barrio_raro". Justificación: análisis de errores muestra 71.25% del RMSE viene de "barrio desconocido".

#### Roadmap de experimentos (orden por potencial / complejidad)

| orden | versión | contenido | estado |
|---|---|---|---|
| 1 | v2 | parseo dual + rooms | campeón |
| 2 | v3 | log-transforms | completado — NO mejoró |
| 3 | v4 | barrio cleanup (Hot Deck → KNN → desconocido) | en curso |
| 4 | v5 | distancias a centros de referencia (sin API) | pendiente |
| 5 | v6 | log(price) target transform | pendiente |
| 6 | v7 | reducción de dimensionalidad (VarianceThreshold / PCA) | pendiente |

#### Consigna E3 (referencia)

Foco: **ingeniería de atributos** (Clase 7) + **reducción de dimensionalidad** (Clase 8). HP `n_estimators` y `max_depth` del RF son modificables. Consigna completa en `Consigna Entrega Parcial 3.pdf` (pendiente de lectura detallada de sub-ítems).

## Material de clase disponible

### Slides (`diapos_clase/`)

- Clase 01 — Presentación + Qué es la ciencia de datos.
- Clase 02 — Preprocesamiento (tipos de atributos, limpieza, discretización, numerización).
- Clase 04 — Análisis de valores atípicos (IQR, Z-score, Mahalanobis, LOF, IsolationForest).
- Clase 04 — Datos faltantes (MCAR/MAR/MNAR; eliminar, imputar; sustitución por media/mediana/moda; **Hot Deck**, **Cold Deck**; regresión; **MICE**; KNN; marcadores de ausencia).
- **Clase 07 — Ingeniería de atributos** (NUEVO E3): transformaciones (log, sqrt, Box-Cox), discretización, binarización, interacciones, extracción de features de texto/fechas, normalización; `sklearn.preprocessing`.
- **Clase 08 — Reducción de dimensionalidad** (NUEVO E3): `VarianceThreshold`, `SelectKBest` (chi2, f_regression, mutual_info), PCA (`sklearn.decomposition`), TruncatedSVD; análisis de varianza explicada.

### Colabs prácticos (`colabs_clase/`)

- Clase 01 — Qué es la ciencia de datos.
- Clase 02 — Preprocesamiento.
- Clase 04 — Análisis de valores atípicos.
- Clase 04 — Datos faltantes.
- Clase 05 — Práctica de outliers (penguins + IsolationForest).
- Clase 05 — Práctica de datos faltantes (Ames Housing + KNN/MICE).
- **Clase 07 — Ingeniería de atributos** (NUEVO E3): `Clase_07_Ingeniería_de_atributos.ipynb`.
- **Clase 08 — Reducción de dimensionalidad** (NUEVO E3): `Clase_08_Reducción_de_dimensionalidad.ipynb`.
- Colab base E3 del docente: `Colab_Base_para_el_Trabajo_Práctico_(Entrega_3).ipynb` (en `colabs_clase/`).

### Librerías permitidas (vistas en clase hasta E3)

`pandas`, `numpy`, `matplotlib`, `seaborn`, `sqlite3`, `scipy.stats.zscore`, y de `sklearn`:

- `model_selection`, `ensemble.RandomForestRegressor`, `ensemble.IsolationForest`.
- `metrics.root_mean_squared_error`.
- `preprocessing.StandardScaler`, `LabelEncoder`, `OrdinalEncoder`, `KBinsDiscretizer`, `Binarizer`, `PolynomialFeatures`.
- `neighbors.LocalOutlierFactor`.
- `impute.SimpleImputer`, `KNNImputer`, `IterativeImputer` (MICE), `MissingIndicator`.
- **`feature_selection.VarianceThreshold`, `SelectKBest`, `chi2`, `f_regression`, `mutual_info_regression`** (Clase 08).
- **`decomposition.PCA`, `TruncatedSVD`** (Clase 08).

## Estrategia ganadora (sugerida por amigo que ya cursó)

> Tomar con pinzas — usar **sólo** lo que cuadre con el contenido de la clase a la fecha de la entrega.

1. Mergear train y test.
2. Sacar duplicados (buscar por `description` o `url`, **no por `id`**); ordenar y quedarse con el registro más cercano al test. (Hay muchos duplicados por las inmobiliarias pueden subir la misma propiedad varias veces con distintos precios en un lapso relativamente cercano de tiempo, tienen distinto ID, pero en si son la misma propiedad).
3. Del train, traer el `price` e imputarlo en test (= **Hot Deck**, técnica vista en Clase 4 → válido para esta entrega).
4. Guardar el override en un diccionario y aplicarlo al final sobre la predicción del modelo.
5. Tratar outliers.
6. Imputar faltantes (promedios / mediana / KNN).

## Plan de la Entrega 2 

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
- **Si transformás el target, aplicá la inversa antes de subir a
  Kaggle** (regla del profe, generalizable). Las transformaciones a
  **features** (ej. `np.log(m2)`, `np.sqrt(n_features)`) NO necesitan
  inversa — el modelo entrena directamente sobre la feature
  transformada y predice en la escala original del target. Las
  transformaciones al **target** (ej. `np.log(price)`) sí: hay que
  aplicar la inversa antes de armar el CSV. Tabla rápida de inversas:

  | Transformación | Inversa |
  |---|---|
  | `np.log(y)` | `np.exp(y_pred)` |
  | `np.log1p(y)` | `np.expm1(y_pred)` |
  | `np.log10(y)` | `10 ** y_pred` |
  | `np.sqrt(y)` | `y_pred ** 2` |
  | `1 / np.sqrt(y)` | `1 / (y_pred ** 2)` |

  Olvidarse de la inversa manda al CSV números entre 9-15 (los logs
  de precios entre 50 K y 3 M) y el RMSE se va al carajo —
  diagnóstico inmediato: la primera submission después de aplicar
  un target transform tiene RMSE > 100 000 sin sentido. Detalle
  estadístico que vale anotar: cuando entrenás en log y aplicás
  `exp` para volver, estás minimizando RMSE en escala log
  (= MAPE-ish), no RMSE en escala original. Predicis la **mediana**
  del precio, no la media; bajo asunción de normalidad de los
  residuos en log con varianza σ², la corrección por sesgo es
  `exp(y_pred + σ²/2)`. Esto es opcional; primero validar si la
  versión simple (sin corrección) ya mejora vs sin transform.
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
- **Experimentos largos (>30 min) corren con checkpointing resumible.**
  HP sweeps, CV5 multi-seed sobre grids grandes, ablations exhaustivas
  — cualquier corrida donde perder el progreso a mitad de camino
  duela. Patrón de referencia: `entregas/_resumable.py::ResumableRunner`.
  Cada combinación atómica (ej. `(n_estimators, max_depth, fold,
  seed)`) se persiste a un JSONL append-only apenas termina; si el
  proceso muere (Cursor cae, kill -9, internet, lo que sea), la
  próxima corrida lee el JSONL, descarta lo hecho y resume desde la
  combinación pendiente siguiente. Reglas:
  - **Granularidad correcta**: el "combo" debe ser la unidad más
    chica que valga la pena no repetir. Para HP sweep + CV es
    `(hiperparámetros + fold + seed)`, no `(hiperparámetros)`
    completo. Si un fold tarda 5 min y se cae a la hora 4, querés
    perder 5 min, no 4 h.
  - **Key determinística**: `key_fn(combo)` debe dar siempre el mismo
    string para el mismo combo. Si la key cambia entre corridas,
    `pending()` devuelve todo de nuevo y se repite trabajo.
  - **Fallos también se persisten**: una excepción en `run_fn` queda
    grabada con `error: "..."` y NO se reintenta. Si querés
    reintentar, borrá esa línea del JSONL.
  - **El JSONL es la fuente de verdad**: el agregado final
    (`results_df()`, gráficos, decisión de campeón) se hace LEYENDO
    el JSONL, no del estado en RAM de la corrida actual. Así una
    corrida hecha en partes da exactamente el mismo resultado que
    una corrida monolítica.
  - **No usar para CV5 multi-seed que ya cabe en memoria** (ej. v6
    de E2 con 3 seeds × 5 folds en 4 min). Sólo cuando el costo de
    repetir > costo de orquestar.
- **Redondear las predicciones a múltiplos de 1 000 USD antes de
  generar la submission** (regla del profe, generalizable a toda
  entrega). El RF predice continuo (ej. `124 567.89`), pero el
  mercado de propiedades en CABA cotiza en valores redondos.
  Validación empírica sobre los 790 K precios del train
  post-filtros de E1 (USD, venta, 5 K-3 M):

  | Granularidad | % de precios reales que la cumplen |
  |---|---|
  | sin centavos (entero) | **100.00 %** |
  | múltiplos de 10 | 96.13 % |
  | múltiplos de 100 | 94.56 % |
  | **múltiplos de 1 000** | **85.63 %** ← sweet spot |
  | múltiplos de 5 000 | 58.59 % |
  | múltiplos de 10 000 | 37.40 % |
  | múltiplos de 50 000 | 10.47 % |

  Granularidad recomendada: **1 000**. Cubre el 85.6 % del
  comportamiento natural sin introducir sesgo (las granularidades
  más gruesas — 5 K, 10 K — bajan demasiado la cobertura). Si el
  precio real es múltiplo de 1 000 y la predicción es continua, el
  ruido de redondeo esperado es uniforme en `[-500, +500]` USD →
  `σ ≈ 289` USD. Marginal vs el RMSE actual (~93 K, mejora
  esperada ~0.3 %), pero free lunch: cero costo, cero riesgo de
  empeorar. Implementación al final del pipeline, justo antes del
  `to_csv`:

  ```python
  df_ap["price"] = df_ap["price"].clip(lower=1)        # ya estaba
  df_ap["price"] = (df_ap["price"] / 1_000).round() * 1_000  # NUEVO
  df_ap["price"] = df_ap["price"].clip(lower=1_000)    # piso post-redondeo
  df_ap["price"].to_csv(csv_path)
  ```

  Aplica **después** del Hot Deck override (los precios del Hot Deck
  ya vienen del train, ya están redondeados, el redondeo es
  idempotente sobre ellos). Aplica **después** de la inversa del
  target transform si se usó `np.log(price)` u otro
  (`np.exp(y_pred_log)` da continuo, redondeás eso). El piso a
  1 000 USD post-redondeo evita predicciones de `0` USD si el modelo
  predice valores ridículamente bajos.

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

**Entrega 3 — parámetros del informe** *(entrega en curso)*:

- **Consigna**: `Consigna Entrega Parcial 3.pdf` — pendiente de revisar sub-ítems exactos. Una vez revisada, actualizar esta sección con la denotación, encabezados y sub-preguntas.
- **Encabezados base esperados** (a confirmar con la consigna): probablemente `A. Ingeniería de atributos`, `B. Reducción de dimensionalidad`, `C. Modelo (Predicción)`, `D. Entrega`. Ajustar cuando se lea el PDF.
- **Punto que exige comparación con entrega anterior**: comparar contra E2 (mejor Kaggle E2 = 93 151, v4). Tabla **E2 vs E3** con el campeón de cada entrega.
- **Restricción técnica**: HP `n_estimators` y `max_depth` son modificables en E3. Técnicas de clase 07 (FE) y clase 08 (dim reduction) habilitadas. Sin datos externos.
- **Arco narrativo esperado** (guía):
  - Introducción: heredo pipeline E2-v4 (filtros, outliers, imputación, Hot Deck); qué agrego en E3.
  - FE: de los faltantes de m2/dormitorios/baños (hallazgo) → parseo dual (hipótesis) → rescate cuantificado (experimento) → mejora CV5+holdout temporal (resultado). Log-transforms: skew observado → hipótesis de splits → delta en métricas.
  - Dim reduction: qué técnica elegí (VarianceThreshold / PCA / SelectKBest), por qué, resultado.
  - Modelo: tabla E2 vs E3 con Kaggle RMSE.
  - Cierre: próximos pasos → E4 (datos no estructurados, APIs, geográficos).

**Entrega 2 — parámetros del informe** *(referencia histórica)*:

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

| entrega | nombre | RMSE CV5 (mean) | RMSE holdout temporal | RMSE Kaggle |
|---|---|---:|---:|---:|
| entrega_2 | v1 | — | — | 93 167.324 |
| entrega_2 | v4 (campeón E2) | 117 219 | — | 93 151 |
| entrega_3 | v1 | 116 669.79 | 117 235.85 | 93 147 |
| entrega_3 | **v2 (campeón actual)** | **112 776.86** | **111 536.16** | **91 399** |

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
