# 🏦 Credit Default Prediction

> Proyecto de ML end-to-end · Clasificación Binaria · Limpieza de datos · Feature Engineering · XGBoost + SHAP + LIME 

![Python](https://img.shields.io/badge/Python-3.14-blue) ![Xgboost](https://img.shields.io/badge/XGBoost-1.x-teal) ![SHAP](https://img.shields.io/badge/SHAP-0.44-purple) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange) ![LIME](https://img.shields.io/badge/LIME-0.2.2-orange)

---

## Overview

This project builds a production-ready pipeline for predicting credit default risk from structured financial data. It covers the full lifecycle — from raw data exploration and feature engineering, through gradient boosting model training, to explainability analysis using SHAP values. The goal is to support lending decisions by identifying high-risk applicants while maintaining interpretability for regulators and business stakeholders.

---

## Pipeline
```
Limpieza de datos > EDA  ›  Feature Engineering  ›  Entrenamiento y evaluación de los modelos  ›  SHAP Analysis
```

## Stages

### 1. Exploratory Data Analysis

Univariate and bivariate analysis of demographics, payment history, credit limits, and bill amounts. Includes:

- Class imbalance diagnosis
- Missing value patterns
- Outlier detection via IQR and visual inspection
- Correlation heatmaps and target-stratified distributions


## Summary of the project
Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. 
This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.
The goal of this competition is to build a model that borrowers can use to help make the best financial decisions
This project develops an end-to-end credit risk prediction system using machine learning to identify potential loan defaults. Multiple models including Logistic Regression, Random Forest, and XGBoost were evaluated, with XGBoost selected as the final model based on balanced predictive performance and robustness on imbalanced data. The model achieved strong classification performance while incorporating explainability through SHAP values and fairness evaluation across demographic groups. 
Fairness mitigation techniques improved disparate impact with minimal performance trade-off, demonstrating a production-oriented approach to responsible credit risk modelling.

## 🎯 Main Objectives:
- Build predictive model for loan default classification (100,000+ applications)
- Build predictive model for loan default classification (100,000+ applications)
- Build predictive model for loan default classification (100,000+ applications)
- Build predictive model for loan default classification (100,000+ applications)


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
