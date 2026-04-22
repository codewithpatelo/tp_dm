# TP Data Mining — Predicción de precios de propiedades

Trabajo Práctico individual de la materia **Data Mining** de la
**Especialización en Explotación de Datos y Descubrimiento de
Conocimiento** (UBA — Facultad de Ciencias Exactas y Naturales,
Departamento de Computación). El TP está estructurado como una
**competencia de Kaggle**:

> [fcen-dm-2026-prediccion-precio-de-propiedades](https://www.kaggle.com/competitions/fcen-dm-2026-prediccion-precio-de-propiedades)

El objetivo es predecir el **precio (USD)** de publicaciones de
propiedades en Argentina (la mayoría en CABA), evaluado por **RMSE**
(raíz del error cuadrático medio).

Este repositorio documenta mi solución, el proceso iterativo y las
lecciones que fui sacando, con la idea de que sirva como guía didáctica
para alumnos futuros de la materia.

---

## Cómo está organizado el TP

Hay **4 entregas parciales (10 % cada una)** y una **entrega final
(60 %)**. Cada entrega parcial pide:

1. Un CSV con la predicción subido a Kaggle.
2. Un informe corto de **máx. 1 carilla** dirigido a un "supervisor"
   técnico — explica decisiones y resultados, **no código**.
3. (Sólo en la final) la notebook `.ipynb` completa.

| Entrega | Foco temático | Estado |
|---|---|---|
| 1 | Filtros del dataset | aprobada (RMSE 166 979) |
| 2 | Outliers + datos faltantes | aprobada (RMSE **93 151**) |
| 3 | Ingeniería de atributos + reducción de dimensionalidad | en curso |
| 4 | Datos no estructurados + APIs + datos geográficos | pendiente |
| Final | Integración completa | pendiente |

**Restricción central**: cada entrega sólo puede usar técnicas vistas
hasta la clase previa a esa entrega. Nada de XGBoost/LightGBM, nada
de librerías externas no vistas, nada de datos externos. El modelo
"oficial" es `RandomForestRegressor` con `n_estimators=500` y
`max_depth=50` fijos en E1-E2; en E3 se permite tunear esas dos
perillas; en E4 ya se puede cambiar de modelo dentro de lo visto.

### El "robot" del profe

Para cada entrega parcial la cátedra publica una notebook **robot**
con una solución mínima — es el **umbral de aprobación**: hay que
ganarle al robot en Kaggle para aprobar.

| Entrega | Robot | Estado |
|---|---|---|
| 1 | `Robot_1_R2D2.ipynb` | superado |
| 2 | `Robot_2_C3PO.ipynb` | superado ampliamente |
| 3 | _(pendiente de publicación)_ | — |

---

## El problema en una página

- **Datasets**: `entrenamiento.db` (SQLite, ~1.3 M filas, 18 columnas)
  + `a_predecir.csv` (13 471 filas — el test público de Kaggle).
- **Diferencia clave train vs test** — el test ya viene **filtrado**:
  100 % CABA, sólo `venta` + `dolares`, precios en rangos razonables.
  El train es un raw scrape de toda la base con basura (alquileres en
  pesos, propiedades en provincias, precios absurdos…). Filtrar bien
  es la mitad del trabajo de E1.
- **Métrica**: RMSE → penaliza fuerte los errores grandes, así que
  los valores caros y los outliers pesan mucho.
- **CSV de envío**: `id,price` con el precio predicho por fila del test.

---

## Mi solución actual (campeón v4 de Entrega 2 — Kaggle 93 151)

Pipeline de 7 pasos, todos justificados con material de clase:

```
0. Carga (sqlite + csv) — datasets originales NUNCA se sobreescriben

1. Filtros heredados de E1
   - location_1 ∈ {Capital Federal, CABA} OR location_2/3 indica CABA
   - property_type ∈ {casa, casas, ph, departamento, departamentos, cochera}
   - operation_type = "venta", currency_type = "dolares"
   - price ∈ [USD 5 000, USD 3 000 000]
   → 1.3 M filas → 123 102

2. Parsing de `features` (regex sobre texto libre)
   - n_dormitorios, n_banos, m2 numéricos
   - 16 amenities como flags binarios (f_balcon, f_pileta, …)
   - barrio (= location_3), len_descripcion, n_features

3. Tratamiento de outliers (Clase 4-5)
   - Univariados → cell-level winsorize: m2 ∉ [10, 1500] → NaN, etc.
   - Multivariados → IsolationForest(contamination=0.01) sobre
     (price, m2, n_dormitorios, n_banos) escalados con StandardScaler.
     Sólo elimina filas en train (test no se filtra).
   → 122 383 filas

4. Imputación (Clase 4 — Datos Faltantes)
   - Numéricas → SimpleImputer(strategy="median") fit on train
   - Categóricas → SimpleImputer(strategy="most_frequent")
   - Marcador de ausencia: m2_was_na (porque m2 es la más imputada)

5. Hot Deck por descripción normalizada (Clase 4)
   - Normalizar description: lower + sin acentos + sin puntuación + min 20 chars
   - Diccionario {desc_norm → price_mediano} construido SÓLO con train
   - Override sobre el test cuando hay match exacto
   → 1 058 overrides (7.85 % del test)

6. Codificación + modelo
   - pd.factorize para barrio_id y property_type_id
   - RandomForestRegressor(n_estimators=500, max_depth=50, random_state=42)
   - 26 features finales

7. Predicción
   - rf.predict(X_test) + override del Hot Deck dict
   → CSV listo para subir
```

**Validación local** (la consigna empezó pidiendo holdout único, pero
fui evolucionando):

- **v1-v3**: holdout único `train_test_split(test_size=0.2, random_state=42)`.
- **v4-v5**: **CV5** (KFold de 5 folds) — porque el holdout único tenía
  std ≈ 2 K, mayor que las diferencias entre versiones.
- **v6 en adelante**: **CV5 multi-seed + holdout temporal** — porque
  v5 mejoró CV5 pero empeoró Kaggle (distribution shift). El split
  temporal (último 20 % por `publication_date`) lo aproxima.

---

## El proceso de aprendizaje (lecciones de cada experimento)

| Versión | Cambio | RMSE Kaggle | Lección clave |
|---|---|---:|---|
| v1 | Pipeline base (filtro + parsing + outliers + impute + HD) | **93 167** | Tener un baseline reproducible vale más que cualquier feature nueva |
| v2 | Agregué `precio_mediano_barrio` y `precio_mediano_barrio_tipo` | 96 891 ⚠️ | **Target leakage**: usar precio del train para imputar features rompe en producción |
| v3 | Imputé lat/lon por mediana de barrio en lugar de global | 97 423 ⚠️ | Otro leakage encubierto: la "señal extra" venía del target, no de la geografía |
| **v4** | **Hot Deck con descripción NORMALIZADA** (lower/sin acentos/sin puntuación) | **93 151** ✓ | Mejorar la **calidad** del input es más rentable que agregar features dudosas |
| v5 | 8 features nuevas (floor, calidad textual, temporales) | 97 498 ⚠️ | CV5 mejoró −4 877; Kaggle empeoró +4 347 → **distribution shift no detectado** |
| v6 | Diagnóstico: CV5 multi-seed + **holdout temporal** + mini-ablation | (no submit) | Holdout temporal +10 491 sobre CV5 confirma el shift; aisló culpables (`pub_year`/`pub_month` rompen +3 385 puro) |

**v4 sigue siendo el campeón** después de 6 iteraciones. La lección
meta: a veces la mejor entrega es la que NO subís, porque tu
diagnóstico te dijo que el "ganador local" era ruido.

---

## Buenas prácticas que adopté

### Sobre la metodología

1. **Una hipótesis por experimento**. Si una corrida cambia features
   nuevas + nueva imputación + nuevo Hot Deck a la vez, no se puede
   atribuir éxito ni fracaso a ninguna en particular. Cada hipótesis
   se aísla y se prueba aislada (regla del consejo "9 errores típicos
   en Kaggle" filtrada al contexto del TP).

2. **Validación honesta antes que LB público**. El leaderboard público
   de Kaggle es sólo ~30 % del test final; ganar 500 puntos contra esa
   muestra puede ser ruido. La fuente de verdad es la validación local
   (CV5 multi-seed + holdout temporal). Esto evita sobre-ajustar al
   leaderboard.

3. **Doble criterio para auto-submit a Kaggle** (desde v6): una
   corrida sólo se sube si mejora **simultáneamente** CV5 y holdout
   temporal por > 1 500 RMSE en cada uno. Si sólo una métrica mejora,
   estoy frente a un trade-off (probable shift) → no submit.

4. **Procesar train + test en simultáneo, NUNCA mergearlos**. Todo
   estimador con estado (`SimpleImputer`, `StandardScaler`,
   `IsolationForest`, `KNNImputer`, `factorize`, vocabularios de Hot
   Deck) se fittea **sólo con train** y se aplica a ambos. Los pasos
   sin estado (parsing, regex, marcadores) corren con el mismo código
   dos veces, una sobre cada DataFrame.

5. **Outliers ridículos en una celda → impute esa celda, no descartes
   la fila**. Si una propiedad tiene `n_banos = 800`, el resto
   (precio, m², barrio, descripción, amenities) sigue siendo
   informativo. Tirar toda la fila por un valor ridículo desperdicia
   el resto. La eliminación de filas se reserva para outliers
   **multivariados** estructurales (IsolationForest).

### Sobre la reproducibilidad

6. **Convención de 3 notebooks por entrega** (`entregas/entrega_<n>/`):
   - `notebook_inicial.ipynb` — el que provee el profe, congelado al inicio.
   - `notebook_campeon.ipynb` — la versión cuya predicción mejor
     fue a Kaggle (se promueve con `entregas/promote_champion.py`).
   - `Colab_Base_(Entrega_<n>).ipynb` (en raíz) — la versión actual de
     experimentación.
   Si un experimento revienta el notebook, no perdemos el campeón.

7. **Cada corrida deja artefactos auditables**:
   - `solucion-entrega<N>-v<v>.csv` (envío).
   - `solucion-entrega<N>-v<v>.json` (parámetros completos: filtros,
     hiperparámetros, métricas locales, `csv_md5`).
   - Entrada nueva en `leaderboard.md` con descripción + RMSE Kaggle.
   - `entrega_<N>_v<v>.executed.ipynb` (notebook con outputs).

8. **Documentar siempre el porqué, no el qué**. El informe de cada
   entrega es para un supervisor; no se aceptan afirmaciones del
   estilo *"se eligió la mediana porque es más robusta"* sin un
   número o un análisis previo que lo respalde. Cualquier umbral o
   hiperparámetro debe poder responder "¿por qué este valor y no
   otro?" apuntando a evidencia del JSON, del EDA o de un colab de
   clase.

### Sobre la herramienta

9. **Orquestador único**: `entregas/run_entrega.py` ejecuta el
   notebook de la entrega vía `nbformat` + `ExecutePreprocessor`,
   captura outputs, registra el experimento, actualiza
   `leaderboard.md` y dispara el auto-submit a Kaggle si pasa el
   doble criterio. Esto evita que el "registro" se desincronice del
   código real.

10. **Nunca tocar los archivos originales** (`entrenamiento.db`,
    `a_predecir.csv`). Las mutaciones in-memory sobre los DataFrames
    son OK siempre que la lectura se haga al inicio de cada corrida.

---

## Errores que aprendí a evitar (los más caros)

### 1. Target leakage disfrazado de feature engineering (v2/v3)

Imputar `lat/lon` con la **mediana del precio del barrio** parece
inocente — al fin y al cabo, el target (price) no aparece en la
fórmula. Pero el barrio se calculó usando filas del train que sí
tienen su precio observado, y el RF aprende que "barrios donde el
mediano de price es alto" predicen bien. En el test público (~30 %
del test final) eso funciona porque los barrios overlap; en el test
privado se desploma.

**Cómo lo cacé**: el holdout local mejoró pero Kaggle empeoró +3 700.
Eso es **siempre** un síntoma de leakage o shift.

**Cómo lo evito ahora**: cualquier feature que dependa del target
del train se calcula **out-of-fold** o se descarta.

### 2. Sobre-confiar en CV5 con KFold aleatorio (v5)

KFold mezcla años uniformemente; el test de Kaggle viene de un
proceso temporal específico. Cuando v5 agregó `pub_year` y
`pub_month` como features, el RF aprendió "publicaciones de 2025 son
más caras que las de 2022" y el CV5 lo premió (cada fold tenía
ambos años). En Kaggle, donde el test viene de un período distinto,
ese lookup es inútil o contraproducente.

**Cómo lo cacé**: CV5 mejoró −4 877; Kaggle empeoró +4 347.

**Cómo lo evito ahora**: holdout temporal **además** del CV5.
Auto-submit requiere mejorar las dos métricas simultáneamente.

### 3. Confundir "barrio" con "etiqueta basura del barrio"

`location_3` (que usábamos como barrio) tiene **31.85 % del train y
21.67 % del test mal etiquetado** con valores como `"Ciudad Autónoma
de Buenos Aires"`, `"Buenos Aires"`, `"Capital Federal"` — son
"barrios" gigantes con miles de propiedades que en realidad NO tienen
barrio. El RF venía splitteando por esa etiqueta como si fuera
informativa.

**Cómo lo cacé**: análisis EDA específico de `location_2`/`location_3`
en E3 (`entregas/entrega_3/eda_features_v4.md`).

**Cómo lo voy a evitar**: cleanup cascada (regex sobre `description` →
`KNNImputer` sobre `lat`/`lon`) en v6 de E3.

### 4. Tirar filas con outlier en UNA columna (lección del profe)

Antes de v4, una fila con `n_banos = 800` la perdíamos entera. Eso
desperdicia el resto del registro (price, barrio, m², amenities).

**Cómo lo evito ahora**: cell-level winsorize → NaN, después
imputación. La fila vive y aporta el resto de la información.

### 5. Sobreajustar al leaderboard público

El leaderboard público es una muestra; cada decisión basada sólo en
él es "Goodhart's Law" (cuando una métrica se vuelve objetivo, deja de
ser una buena métrica).

**Cómo lo evito ahora**: la decisión de submit la toma la validación
local (CV5 multi-seed + holdout temporal). Kaggle valida o refuta,
pero no decide.

---

## Cómo está organizado el repo

```
tp_dm/
├── README.md                         ← este archivo
├── .claude/CONTEXT.md                ← contexto completo para LLM/lectura humana
├── datasets/                         ← entrenamiento.db, a_predecir.csv (NO incluidos en git)
├── colabs_clase/                     ← notebooks de las clases (material de cátedra)
├── diapos_clase/                     ← slides de las clases
├── Robot_1_R2D2.ipynb                ← solución mínima del profe para E1
├── Robot_2_C3PO.ipynb                ← solución mínima del profe para E2
├── Colab_Base_para_el_TP_(Entrega_2).ipynb  ← notebook actual de experimentación
└── entregas/
    ├── run_entrega.py                ← orquestador: ejecuta + registra + submit
    ├── promote_champion.py           ← copia .executed.ipynb → notebook_campeon.ipynb
    ├── _lib.py                       ← helpers (parse leaderboard, regenerate informe, md→pdf)
    ├── entrega_1/
    ├── entrega_2/
    │   ├── notebook_campeon.ipynb    ← v4 (la que ganó Kaggle 93 151)
    │   ├── leaderboard.md            ← tabla de TODAS las versiones probadas
    │   ├── Entrega_2_informe.md      ← informe de 1 carilla del campeón
    │   ├── solucion-entrega2-v*.csv  ← un CSV por experimento
    │   ├── solucion-entrega2-v*.json ← parámetros completos por experimento
    │   └── entrega_2_v*.executed.ipynb  ← notebook con outputs por experimento
    └── entrega_3/                    ← en curso
        ├── eda_features_v4.py        ← EDA preparatorio (skewness, IG, drift, geografía)
        ├── eda_features_v4.md        ← resultado del EDA
        ├── error_analysis_v4.py      ← análisis de residuals del campeón
        └── error_analysis_v4.md      ← resultado del análisis de errores
```

## Cómo correr una corrida

Necesitás los datasets en `datasets/entrenamiento.db` y
`datasets/a_predecir.csv` (no los subo al repo por tamaño / términos
de la competencia).

```bash
# Ejecutar el notebook de la entrega 2 (versión actual)
python entregas/run_entrega.py --entrega entrega_2 --nombre v7 \
    --desc "Descripción corta del cambio que estoy probando"

# Una vez subido a Kaggle, registrar el RMSE público
python entregas/run_entrega.py --record-kaggle \
    --entrega entrega_2 --nombre v7 --kaggle-rmse 92500.0

# Si v7 pasa el doble criterio CV5+holdout, promoverlo a campeón
python entregas/promote_champion.py --entrega entrega_2 --nombre v7
```

## Stack técnico

- **Python 3.13**
- **pandas / numpy / scipy** — manipulación y stats.
- **scikit-learn** — `RandomForestRegressor`, `IsolationForest`,
  `SimpleImputer`, `KNNImputer`, `IterativeImputer (MICE)`,
  `MissingIndicator`, `StandardScaler`, `LocalOutlierFactor`,
  `model_selection`, `metrics.root_mean_squared_error`.
- **sqlite3** — lectura del train.
- **kaggle CLI** — submissions automatizadas.
- **nbformat + nbclient** — ejecución programática del notebook.

## Referencias

- [Competencia en Kaggle](https://www.kaggle.com/competitions/fcen-dm-2026-prediccion-precio-de-propiedades)
- [Especialización en Explotación de Datos y Descubrimiento de Conocimiento — UBA Exactas](https://datamining.dc.uba.ar/datamining/)
- Material de clase en `colabs_clase/` y `diapos_clase/`.
- Contexto completo del proyecto: `.claude/CONTEXT.md`.

## Disclaimer — uso de herramientas de IA

Este TP fue desarrollado con asistencia de **Cursor IDE** usando los
modelos **Claude Opus 4.7** y **Claude Code** (Anthropic) como agentes
de programación. La IA se usó para:

- Pair-programming en la implementación del pipeline (parsing de
  features, validación cruzada, automatización del orquestador).
- Diseño y revisión crítica del plan de experimentos
  (hipótesis explícitas, detección de leakage, propuesta de
  controles como holdout temporal y multi-seed).
- Generación de los EDAs preparatorios (`eda_features_v4.py`,
  `error_analysis_v4.py`) y de la primera versión del informe.
- Documentación (este README, `CONTEXT.md`, comentarios en código).

Las **decisiones técnicas finales**, la **interpretación de
resultados**, la elección de qué experimentos correr y la
narrativa del informe de cada entrega son mías. La IA fue una
herramienta de aceleración y revisión, no de delegación: cada
sugerencia se contrastó contra el material de clase
(`colabs_clase/`, `diapos_clase/`) y contra el resultado empírico
de la corrida correspondiente en Kaggle.

---

_Repositorio mantenido como diario público de aprendizaje. Si sos
alumno futuro de la materia y algo de esto te sirve, mejor todavía._
