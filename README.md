 # 🏦 Credit Default Prediction

> Proyecto de ML end-to-end · Clasificación Binaria · Limpieza de datos · Feature Engineering · Regresión Logística · XGBoost + SHAP + LIME 

![Python](https://img.shields.io/badge/Python-3.14-blue) ![Xgboost](https://img.shields.io/badge/XGBoost-1.x-teal) ![SHAP](https://img.shields.io/badge/SHAP-0.44-purple) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange) ![LIME](https://img.shields.io/badge/LIME-0.2.2-orange)

---

## Resumen

Este proyecto desarrolla un sistema integral y listo para producción para la predicción del riesgo de impago crediticio a partir de datos financieros estructurados.En él, se abarca todo el ciclo de vida del modelo, desde el análisis exploratorio de datos y featuring engineering hasta el entrenamiento de modelos de machine learning y el análisis de explicabilidad mediante valores SHAP.Se evaluaron distintos enfoques, como Regresión Logística y XGBoost, abordando de forma específica el fuerte desbalanceo del conjunto de datos (93%-7%) mediante técnicas adecuadas para este tipo de problemática. XGBoost fue finalmente seleccionado por su sólido rendimiento predictivo y su capacidad para afrontar eficazmente este escenario. El modelo final obtenido permite estimar de manera fiable la probabilidad de que un cliente incurra en dificultades financieras en un horizonte de dos años, contribuyendo así a mejorar la toma de decisiones en concesión de crédito. 

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
El presente proyecto ha sido desarrollado utilizando el presente conjunto de datos:  '[Give me some credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)'.
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



## Stages

### 1. Exploratory Data Analysis

Univariate and bivariate analysis of demographics, payment history, credit limits, and bill amounts. Includes:

- Class imbalance diagnosis
- Missing value patterns
- Outlier detection via IQR and visual inspection
- Correlation heatmaps and target-stratified distributions
