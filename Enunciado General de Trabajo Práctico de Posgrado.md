# **Condiciones Generales de la Competencia Individual 2026**

**Curso**: Data Mining  
**Carrera**: Esp. en Explotación de Datos y Descubrimiento de Conocimiento  
**Versión doc**: 20260311

# **1\. Introducción**

La evaluación es un trabajo práctico individual (análisis de datos) para predecir precios de propiedades, desarrollado como una competencia en la plataforma Kaggle provista por la asignatura.

# **2\. Objetivo General del Trabajo Práctico**

El objetivo es predecir precios de propiedades mediante un análisis exhaustivo de los datos, aplicando las estrategias y métodos aprendidos en clase para superar los umbrales de aprobación establecidos.

# **3\. Formato del Trabajo Práctico**

El trabajo consistirá en una competición en Kaggle. Se proporcionará un conjunto de datos para la predicción de la columna objetivo, y los estudiantes deberán cargar y documentar sus propuestas. Una función de pérdida establecerá un *ranking*: aquellos que obtengan el menor error ascenderán en la clasificación. De esta manera, la competición valorará la aplicación óptima de las técnicas abordadas. Los pormenores del software se detallarán durante la sesión de clase.

# **4\. Estructura del Trabajo Práctico**

El TP se compone de tres entregas parciales y una entrega final, cada una con un enfoque y requisitos específicos. A continuación se describe el cronograma y detalles de cada entrega.

## **4.1. Entregas Parciales**

Las entregas parciales están diseñadas para guiar el desarrollo progresivo del trabajo, asegurando la retroalimentación oportuna y la correcta orientación metodológica. Cada entrega aborda varios aspectos cruciales de la disciplina de minería de datos y cuenta con su propio enunciado detallado y condiciones específicas.

| Entrega Parcial | Fecha Límite C1 | Fecha Límite C2 | Enunciado Específico y Condiciones | Obligatoria |
| :---- | :---- | :---- | :---- | :---- |
| Entrega 1 | 25 mar 2026 12:00 a.m. GMT-3 | 27 mar 2026 12:00 a.m. GMT-3 | [Entrega Parcial 1](https://docs.google.com/document/u/0/d/1P15nWg1SkgLj33f7zeqsFiEbDaLpyzMeI9ZDCiTZAic/edit) | Si (10% de la nota final) |
| Entrega 2 | 15 abr 2026 12:00 a.m. GMT-3 | 17 abr 2026 12:00 a.m. GMT-3 | File | Si (10% de la nota final) |
| Entrega 3 | 13 may 2026 12:00 a.m. GMT-3 | 15 may 2026 12:00 a.m. GMT-3 | File | Si (10% de la nota final) |
| Entrega 4 | 10 jun 2026 12:00 a.m. GMT-3 | 12 jun 2026 12:00 a.m. GMT-3 | File | Si (10% de la nota final) |

### **4.1.1. Condiciones Generales para Entregas Parciales**

* **Formato y Extensión:** Cada enunciado específico detalla el formato de entrega.  
* **Modalidad:** Las entregas se realizan a través de la plataforma antes de la fecha límite establecida.  
* **Evaluación:** Las entregas parciales son obligatorias pero varían el aporte que hacen a la nota final.  
* **Consultas:** Para consultas específicas relacionadas con cada entrega, se utilizará el espacio de la clase práctica en modalidad presencial y/o virtual (se irá indicando semana a semana).

## **4.2. Entrega Final**

Fecha de entrega final: 28 feb 2026 7:00 p.m. GMT-3

La Entrega Final representa la consolidación de todo el trabajo realizado en las etapas parciales. Debe ser un documento notebook Colab completo y coherente que refleje el cumplimiento del objetivo general del TP y cuyo resultado sea el mismo archivo que hay seleccionado para el ranking final.

La entrega final aporta el 60% de la nota final.

## **4.3. Requisitos de la Entrega Final**

1. **Integración:** Debe integrar y corregir los aspectos desarrollados en las Entregas Parciales 1, 2, 3 y 4, considerando la retroalimentación recibida.  
2. **Formato:** Debe ser un documento notebook Colab completo y coherente que refleje el cumplimiento del objetivo general del TP y cuyo resultado sea el mismo archivo que hay seleccionado para el ranking final.  
3. **Contenido:** Debe incluir el código y texto explicando las decisiones tomadas. Considerarlo como un informe documentado, debe estar acorde a una presentación profesional.  
4. **Presentación:** Las primeras posiciones, en clase, el día de la entrega, deberán exponer las decisiones que consideran los llevaron a tener la mejor solución, qué técnicas le aportaron más y cuales no funcionaron o lo hicieron de manera deficiente.

## **4.4. Formato de entrega**

Cada entrega solicita 2 archivos:

1) Un archivo en formato csv con la predicción en la plataforma de la competencia

2) Un archivo tipo informe (pdf, docx, etc...) en el campus de la asignatura

3) Solo en el caso de la entrega final, se le solicitará el notebook colab (en formato ipynb)

### **4.4.1. Entrega de predicción en plataforma**

La plataforma detallará el formato de entrega. Independientemente del script o notebook que se utilice, la plataforma solicitará la subida de un archivo en formato `csv` (valores separados por coma) con 2 columnas: Id de la propiedad y Precio Pronosticado. Ejemplo:

`id,price`  
`1,5000.00`  
`2,3542.25`  
etc...

Será obligatoria la inclusión de la cabecera tal como figura en el ejemplo. Los identificadores de publicación deben coincidir con los provistos en el conjunto de datos de prueba.

### **4.4.2. Informe en campus**

En 3 las entregas parciales se solicitará que suba un informe al campus.  El informe es un resumen ordenado de las decisiones tomadas en el notebook que da como resultado la predicción elegida para la entrega.

Es importante pensar en el informe como algo que se presenta a un Jefe o Supervisor, interesado en el proceso, las decisiones y resultados, pero no interesado en el código en sí. Pueden ponerse resultados o estadísticas siempre que dejen en claro una decisión tomada, sin excederse.

Como orientación, un informe de una entrega no debería exceder una carilla.

### **4.4.3. Entrega de notebook final**

Solo en la Entrega final se solicitará la notebook completa (formato ipynb, exportable desde Google Colab). Se espera que la notebook tenga comentarios y sea ordenada, sin comentarios innecesarios ni obvios.

Se espera que la notebook solicite solo los archivos de la competencia y genere EXACTAMENTE el mismo csv que usted elija como entrega final.

Se aconseja aprovechar la capacidad de las notebooks de escribir bloques en formato Markdown, lo que mejora la organización y legibilidad. Similar a los informes de las entregas parciales, estos comentarios deben ser directos, sencillos y breves, comentando solamente conclusiones o decisiones basadas en los datos analizados. En el caso de las notebooks, se puede intercalar la narración con código que muestre como salida lo que se desea comentar.

# **5\. Criterios de Evaluación**

Para aprobar una entrega (entrega parcial 2 y entrega final), los estudiantes deberán presentar una predicción que supere el mejor resultado obtenido por los docentes en el *leaderboard* público de la competencia para la primera entrega y el *leaderboard* privado para el resto.

* Predicciones superiores a la de los docentes: Las predicciones que superen el umbral establecido por los docentes recibirán una calificación entre 7 y 10 puntos, distribuida de la siguiente manera:  
  * Los primeros cuatro estudiantes: 10 puntos  
  * Los siguientes ocho estudiantes: 9 puntos  
  * Los siguientes dieciséis estudiantes: 8 puntos  
  * El resto: 7 puntos  
* Instancia de Recuperación: Los estudiantes que no superen la predicción de los docentes tendrán una oportunidad de recuperación. Dispondrán de una semana para presentar una nueva entrega. Si en esta instancia logran superar la predicción docente, obtendrán una nota entre 4 y 6 puntos:  
  * Los primeros cuatro estudiantes: 6 puntos  
  * Los siguientes ocho estudiantes: 5 puntos  
  * El resto: 4 puntos  
* Si no logran superar la predicción docente en la instancia de recuperación, la entrega será calificada con cero puntos.

# **6\. Detalles sobre la entrega del código fuente**

En cada entrega, además de la predicción, se deberá adjuntar el código fuente debidamente documentado que genere dicha predicción. El código debe presentarse en formato Jupyter Notebook (.ipynb) y debe ser ejecutable en Google Colab, generando la predicción enviada.

**IMPORTANTE:** Se sancionará con la desaprobación automática a los alumnos en cuyas entregas se detecte copia y además se reportará a la Dirección de la Maestría solicitando sanciones mayores.  
