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

### 1. Limpieza de datos 



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



### 2. Análisis exploratorio

Univariate and bivariate analysis of demographics, payment history, credit limits, and bill amounts. Includes:

- Class imbalance diagnosis
- Missing value patterns
- Outlier detection via IQR and visual inspection
- Correlation heatmaps and target-stratified distributions


### 🧩 Ingeniería de variables (Feature Engineering)

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



