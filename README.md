 # 🏦 Credit Default Prediction

> Proyecto de ML end-to-end · Clasificación Binaria · Limpieza de datos · Feature Engineering · Regresión Logística · XGBoost + SHAP + LIME 

![Python](https://img.shields.io/badge/Python-3.14-blue) ![Xgboost](https://img.shields.io/badge/XGBoost-1.x-teal) ![SHAP](https://img.shields.io/badge/SHAP-0.44-purple) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange) ![LIME](https://img.shields.io/badge/LIME-0.2.2-orange)

---

## Resumen

Este proyecto desarrolla un sistema integral y listo para producción para la predicción del riesgo de impago crediticio a partir de datos financieros estructurados. En él, se abarca todo el ciclo de vida del modelo, desde el análisis exploratorio de datos y featuring engineering hasta el entrenamiento de modelos de machine learning y el análisis de explicabilidad mediante valores SHAP. Se evaluaron distintos enfoques, como Regresión Logística y XGBoost, abordando de forma específica el fuerte desbalanceo del conjunto de datos (93%-7%) mediante técnicas adecuadas para este tipo de problemática. XGBoost fue finalmente seleccionado por su sólido rendimiento predictivo y su capacidad para afrontar eficazmente este escenario. El modelo final obtenido permite estimar de manera fiable la probabilidad de que un cliente incurra en dificultades financieras en un horizonte de dos años, contribuyendo así a mejorar la toma de decisiones en concesión de crédito. 

## 🎯 Puntos clave  
- Se ha creado un modelo de ML de predicción de riesgo de morosidad con un dataset amplio (100k+ filas), con una variable target severamente desbalanceada (93%-7%).
- En el eda se han detectado algunos patrones interesantes y útiles para una posible toma de decisiones de negocio.
- En el modelado se ha hecho especial énfasis en maximizar la relación entre recall de la variable minoritaria y la precisión global del modelo (F1-score), ajustando hiperparámetros y threshold acorde a este criterio.
- Se han evaluado diferentes métricas como la precisión global del modelo, F1-Score y AUC-PR, tomando las precauciones necesarias a sabiendas del fuerte sesgo que podría generar el desbalance de la variable respuesta.
- Finalmente se ha analizado la importancia de algunas de las variables más críticas del modelado, utilizando LIME y SHAP.


## Pipeline
```
Limpieza de datos > EDA  ›  Feature Engineering  ›  Entrenamiento y evaluación de los modelos  ›  SHAP Analysis
```


## 📊 Dataset
El presente proyecto ha sido desarrollado utilizando el conjunto de datos:  '[Give me some credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)'.
Este conjunto de datos incluye información financiera y de comportamiento de los solicitantes de crédito. Cada fila representa a una persona que solicita un préstamo e incluye atributos como ingresos, deudas, historial de pagos, número de cuentas abiertas y tamaño de la familia. Estos datos permiten analizar el riesgo de incumplimiento y predecir la probabilidad de que un solicitante no pague su deuda.

| Columnas         | Nombre Simplificado   | Descripción                                                          |
| ------------------------------------ | ---------------- | -------------------------------------------------------------------- |
| SeriousDlqin2yrs                     | Moroso           | Variable binaria que indica si la persona no pagó su deuda por más de 90 días (1 = Sí, 0 = No)   |
| RevolvingUtilizationOfUnsecuredLines | Uso de Crédito % | Porcentaje del crédito disponible que se está utilizando actualmente |
| age                                  | Edad             | Edad del prestatario en años                                         |
| NumberOfTime30-59DaysPastDueNotWorse | Retrasos 1 Mes   | Número de veces que el prestatario tuvo un retraso de 1 mes          |
| DebtRatio                            | Deuda vs Ingreso | Deuda mensual y gastos divididos por el ingreso total                |
| MonthlyIncome                        | Ingreso Mensual  | Ingreso mensual bruto del prestatario                                |
| NumberOfOpenCreditLinesAndLoans      | Cuentas Abiertas | Número total de tarjetas de crédito y préstamos activos              |
| NumberOfTimes90DaysLate              | Retrasos 3 Meses | Número de veces que el prestatario tuvo un retraso de 3 o más meses  |
| NumberRealEstateLoansOrLines         | Hipotecas        | Número de préstamos o líneas de crédito inmobiliario                 |
| NumberOfTime60-89DaysPastDueNotWorse | Retrasos 2 Meses | Número de veces que el prestatario tuvo un retraso de 2 meses        |
| NumberOfDependents                   | Tamaño Familiar  | Número de dependientes (hijos, cónyuge u otros)                      |



## Etapas

### 🧹 1. Limpieza de datos 



#### 1.1. Datos Faltantes

- Primeramente hemos comprobado aquellas columnas que presentan datos faltantes:

![Descripción](images/missing.png)

- Como se puede comprobar por el presente gráfico, las variable MontlyIncome y NumberOfDependents son las únicas que presentan missing values.

- En primera instancia trataremos la variable NumberOfDependents, ya que es más intuitiva. Para entenderla veámos la distribución de sus valores:
  
Dependientes | Nº de clientes
-------------|---------------
0            | 86,705
1            | 26,292
2            | 19,501
3            | 9,479
4            | 2,860
5            | 745
6            | 158
7            | 51
8            | 24
9            | 5
10           | 5
13           | 1
20           | 1

- La distribución de la variable muestra que la gran mayoría de los clientes presentan entre 0 y 2 dependientes, concentrando así la mayor parte de las observaciones. Asimismo, se identifican valores atípicos claros (como 10, 13 y 20 dependientes), cuya frecuencia es extremadamente baja y, por tanto, poco representativa del conjunto de datos. En consecuencia, se ha optado por eliminar estos outliers para evitar distorsiones en el análisis. Para la imputación de valores faltantes en el resto de observaciones, se ha utilizado la moda (0), al ser el valor más frecuente y representativo de la distribución.
 
- La variable Montly Income, por otro lado, es más compleja de tratar, esta presenta un 19,77% y una distribución sesgada a la derecha, con algunos outliers extremos.
- Examianamos la distribución de la variable segmentada según la condición de morosidad del cliente, habiendo recortado los outliers más evidentes:

![Descripción](images/Income-Default.png)


- Dado que la distribución de la variable difiere entre individuos en situación de default y aquellos que no lo están, imputar los valores faltantes utilizando una medida global podría introducir sesgos y distorsionar la relación con la variable objetivo. Por ello, se opta por una imputación más robusta basada en la mediana específica de cada grupo, preservando mejor la estructura real de los datos. Adicionalmente, hemos aplicado una transformación logarítmica, lo que nos permite reducir la asimetría y el efecto de valores extremos, favoreciendo una distribución más estable y adecuada para el modelado.
  
```python
# Realizamos primeramente una transformación logarítmica
df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome"])

# Imputamos con mediana por grupo de default 
df["MonthlyIncome_log"] = df.groupby("SeriousDlqin2yrs")["MonthlyIncome_log"]\
                            .transform(lambda x: x.fillna(x.median()))
```



### 📈 2. Análisis exploratorio

Univariate and bivariate analysis of demographics, payment history, credit limits, and bill amounts. Includes:

- Class imbalance diagnosis
- Missing value patterns
- Outlier detection via IQR and visual inspection
- Correlation heatmaps and target-stratified distributions

#### 2.1. Correlaciones
- En esta sección, se examina la matriz de correlación con el objetivo de identificar qué variables presentan mayor asociación con la variable objetivo SeriousDlqin2yrs, así como posibles problemas de multicolinealidad entre features. Este análisis resulta especialmente útil para entender qué señales aportan mayor valor predictivo y para orientar tanto la selección de variables como la construcción de nuevas transformaciones que mejoren el rendimiento y la interpretabilidad del modelo.
  
<p align="center">
  <img src="images/Corr.png" width="600"/>
</p>

-El análisis de correlaciones muestra que la variable objetivo SeriousDlqin2yrs (default) está principalmente asociada con indicadores de comportamiento de pago atrasado, destacando weighted_late_score, TotalPastDue y NumberOfTimes90DaysLate, que presentan las correlaciones positivas más elevadas. Esto confirma que el historial de morosidad reciente es el principal driver del riesgo de incumplimiento. Variables derivadas como HasSeriousDelinquency y los distintos contadores de retrasos (30-59 y 60-89 días) también refuerzan esta señal, evidenciando una estructura coherente entre features relacionadas. Por otro lado, variables como age y CreditHistoryLength muestran correlaciones negativas moderadas, sugiriendo que perfiles más maduros y con mayor historial crediticio tienden a presentar menor probabilidad de default. En contraste, variables financieras clásicas como DebtRatio o MonthlyIncome tienen una relación débil con la variable objetivo, lo que sugiere que, en este dataset, el comportamiento histórico es mucho más predictivo que la capacidad económica declarada. Finalmente, se observa cierta multicolinealidad entre variables derivadas de morosidad, lo cual se tendrá en cuenta en fases posteriores de modelado para evitar redundancias y mejorar la interpretabilidad del modelo.

#### 2.2. Comportamiento de las variable bajo riesgo


<p align="center">
  <img src="images/meandeaf.png" width="800"/>
</p>






#### 2.3. Análisis por grupos de edad

- Este análisis explora la relación entre la edad de los clientes y su comportamiento crediticio, con foco en la probabilidad de default y los distintos niveles de morosidad. A través de la segmentación por grupos etarios, se busca identificar patrones de riesgo que permitan mejorar la capacidad predictiva del modelo de credit risk.
  
<p align="center">
  <img src="images/age_group_analysis.png" width="700"/>
</p>

La gráfica muestra una clara concentración del riesgo en los grupos de edad intermedia, especialmente entre 36 y 55 años, donde se observan las tasas más altas tanto de default como de retrasos en distintos rangos (30–59 y 60–89 días). El grupo de 46–55 años destaca como el segmento con mayor volumen de incumplimientos y morosidad acumulada, lo que sugiere una combinación de mayor exposición crediticia y potenciales tensiones financieras. En contraste, los segmentos más jóvenes (18–25) y mayores (65+) presentan niveles significativamente más bajos de incumplimiento, lo que puede estar asociado a menor acceso al crédito o a comportamientos más conservadores. 

### 🧩 3. Ingeniería de variables (Feature Engineering)

-Esta sección resume las variables derivadas creadas con el objetivo de mejorar la capacidad predictiva del modelo de riesgo de crédito. Las transformaciones se centran en capturar la capacidad de pago, el comportamiento histórico del cliente y su segmentación.



Esta sección resume las variables derivadas creadas con el objetivo de mejorar la capacidad predictiva del modelo de riesgo de crédito.

| Variable                | Tipo        | Descripción                                                                 | Intuición de riesgo                          |
| ----------------------- | ----------- | --------------------------------------------------------------------------- | -------------------------------------------- |
| `income_per_dependent`  | Numérica    | Ingreso mensual dividido por número de dependientes (+1 para evitar división por cero) | Menor valor → mayor carga financiera         |
| `utilization_capped`    | Numérica    | Utilización de crédito acotada entre 0 y 1                                  | Reduce el impacto de valores extremos        |
| `CreditHistoryLength`   | Numérica    | Edad - 18 (aproximación a la antigüedad crediticia)                         | Mayor antigüedad → menor riesgo              |
| `TotalPastDue`          | Numérica    | Número total de retrasos en pagos                                           | Más retrasos → mayor riesgo                  |
| `weighted_late_score`   | Numérica    | Puntuación ponderada de retrasos según gravedad                             | Penaliza más los impagos severos             |
| `HasSeriousDelinquency` | Binaria     | 1 si existe algún retraso >90 días                                          | Fuerte indicador de default                  |
| `high_utilization_flag` | Binaria     | 1 si la utilización de crédito >80%                                         | Alta utilización → mayor riesgo              |
| `AgeGroup`              | Categórica  | Edad agrupada en rangos                                                     | Captura efectos del ciclo de vida            |
| `IncomeGroup`           | Categórica  | Cuartiles de ingreso                                                        | Segmentación socioeconómica                  |
| `DTICategory`           | Categórica  | Categorías del ratio deuda/ingresos (DTI)                                   | Mayor DTI → menor capacidad de pago          |

### 📊 4. Desarrollo de los modelos de ML

Se evaluaron dos modelos para el problema de clasificación:

- Regresión Logística como modelo Baseline.
- XGBoost como modelo más avanzado.
- Se han evaluado otros modelos como LightGBM y RandomForest, sin embargo XGBoost ha arrojado mejores resultados.

---


#### 🔎 4.1 Resultados: Regresión Logística 



| Clase | Precisión | Recall | F1-score | Soporte |
|------|----------|--------|----------|---------|
| 0    | 0.97     | 0.87   | 0.92     | 27,996  |
| 1    | 0.26     | 0.65   | 0.37     | 1,948   |

| Métrica global     | Valor |
|-------------------|------|
| Accuracy          | 0.86 |
| Macro Avg F1      | 0.64 |
| Weighted Avg F1   | 0.88 |
| AUC-PR            | 0.8508 |

---

#### 🔎 4.2 Resultados: XGBoost (Optimizado)

| Clase | Precisión | Recall | F1-score | Soporte |
|------|----------|--------|----------|---------|
| 0    | 0.97     | 0.93   | 0.95     | 27,996  |
| 1    | 0.40     | 0.66   | 0.50     | 1,948   |

| Métrica global     | Valor |
|-------------------|------|
| Accuracy          | 0.91 |
| Macro Avg F1      | 0.72 |
| Weighted Avg F1   | 0.92 |
| AUC-ROC           | 0.8765 |

---

####  4.3 Comparación Directa

| Métrica              | Regresión Logística | XGBoost |
|---------------------|--------------------|--------|
| Accuracy            | 0.86               | 0.91   |
| F1-score (Clase 1)  | 0.37               | 0.50   |
| Recall (Clase 1)    | 0.65               | 0.66   |
| Precisión (Clase 1) | 0.26               | 0.40   |

---

#### 4.4 Optimización de Threshold

Dado el fuerte desbalance de clases, no se utilizó el threshold por defecto (0.5).  
En su lugar, se optimizó el umbral de decisión priorizando:

> **Recall ≥ 0.65 en la clase positiva**

- Al final hemos seleccionado: 

| Modelo              | Threshold óptimo | Criterio |
|---------------------|-----------------|----------|
| Regresión Logística | 0.01            | Maximizar recall ≥ 0.65 |
| XGBoost             | 0.3268          | Maximizar recall ≥ 0.65 |

---

### 🧠 5. Interpretación de los modelos 


- El threshold de la Regresión Logística evidencia sus limitaciones
- XGBoost permite un ajuste más equilibrado y usable en producción

👉 Esto refuerza la elección de XGBoost como modelo final

#### 5.1 Regresión Logística (threshold = 0.01)

- Threshold extremadamente bajo  
- El modelo clasifica casi todo como positivo
- Resultado:
  - ✅ Alto recall (detecta muchos positivos)
  - ❌ Muy baja precisión (muchos falsos positivos)

👉 Indica que el modelo **no separa bien las clases**

---

#### 5.2 XGBoost (threshold = 0.3268)

- Threshold más razonable
- Mantiene recall ≥ 0.65 sin colapsar la precisión

👉 Indica que el modelo:
- Tiene mejor capacidad de discriminación
- Permite un balance más realista entre métricas

---

####  5.3 Implicaciones de negocio

- Reducir el threshold aumenta el recall pero también los falsos positivos
- Aumentarlo mejora precisión pero pierde casos positivos

👉 La elección depende del coste relativo de:
- Falsos negativos (casos no detectados)
- Falsos positivos (alarmas innecesarias)

---



### 🧠 6. Análisis De los resultados

#### 6.1 Desbalance de clases

El dataset presenta un fuerte desbalance:
- Clase 0: ~93%
- Clase 1: ~7%

Esto hace que:
- Accuracy sea una métrica limitada
- Sea clave analizar recall, precisión y F1 en la clase minoritaria

---

#### 6.2 Regresión Logística

- Buen rendimiento en la clase mayoritaria
- Recall aceptable en clase 1 (0.65)
- **Problema principal:** precisión muy baja (0.26)
  - Muchos falsos positivos
- Modelo simple, interpretable, pero limitado para capturar relaciones complejas
- **AUC-PR (Logística): 0.8508** : Métrica adecuada para desbalance

👉 En la práctica: sirve como baseline, pero no es suficiente

---

#### 6.3 XGBoost

- Mejora clara en todas las métricas clave
- **Gran mejora en precisión de la clase 1 (0.26 → 0.40)**
- Recall prácticamente igual (0.66)
- F1-score mucho más equilibrado (0.50)
- **AUC-PR (XGBoost): 0.8765** : Buena separación entre clases

⚠️ 6.5 Limitaciones

- Precisión en clase positiva aún moderada (0.40)
- Persisten falsos positivos
- Dataset desbalanceado sigue siendo un reto

---

#### 🚀 6.4 Conclusión

XGBoost es claramente superior a la Regresión Logística en este problema:

- Mejora significativa en la detección de la clase minoritaria
- Mejor equilibrio entre precisión y recall
- Mayor robustez global

👉 Es el modelo recomendado para producción.

---

### 🔧 7. Importancia de las variables



#### 7.1 Importancia en el gain

<p align="center">
  <img src="images/Features_xgb.png" width="600"/>
</p>

#### 7.2 Lime

<p align="center">
  <img src="images/Lime_xgb.png" width="600"/>
</p>

#### 7.3 Shap

<p align="center">
  <img src="images/Shap_xgb.png" width="600"/>
</p>

### 🔧 8. Próximos pasos

- Ampliación del EDA
- Técnicas de balanceo (SMOTE, undersampling)
- Optimización enfocada en métricas de negocio:
  - Recall: evaluar en términos monetarios si perder positivos es crítico.
  - Precisión: evaluar si incurrir en falsos positivos es costoso.
- Feature engineering adicional
- Ensemble de modelos


---













