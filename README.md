# 🏦 Credit Default Prediction

> End-to-end ML pipeline · Binary classification · LightGBM + SHAP

![Python](https://img.shields.io/badge/Python-3.10-blue) ![LightGBM](https://img.shields.io/badge/LightGBM-4.x-teal) ![SHAP](https://img.shields.io/badge/SHAP-0.44-purple) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)

---

## Overview

This project builds a production-ready pipeline for predicting credit default risk from structured financial data. It covers the full lifecycle — from raw data exploration and feature engineering, through gradient boosting model training, to explainability analysis using SHAP values. The goal is to support lending decisions by identifying high-risk applicants while maintaining interpretability for regulators and business stakeholders.

---

## Pipeline
```
EDA  ›  Feature Engineering  ›  LightGBM Training  ›  SHAP Analysis
```

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
The following project has been developed using the "Give Me Some Credit" dataset from Kaggle.
It contains financial and behavioral attributes of borrowers. Each row represents a person applying for credit.

| Original Column Name                               | Simple Name       | Description                                                                 |
|----------------------------------------------------|-----------------|-----------------------------------------------------------------------------|
| SeriousDlqin2yrs                                   | Defaulted       | Whether the person failed to pay their debt for 90+ days (1 = Yes, 0 = No) |
| RevolvingUtilizationOfUnsecuredLines              | Credit Usage %  | Percentage of available credit currently being used                        |
| age                                                | Age             | Borrower's age in years                                                     |
| NumberOfTime30-59DaysPastDueNotWorse              | 1-Month Lates   | Number of times the borrower was 1 month past due                           |
| DebtRatio                                          | Debt vs Income  | Monthly debt and expenses divided by total income                            |
| MonthlyIncome                                      | Monthly Income  | Borrower's gross monthly income                                             |
| NumberOfOpenCreditLinesAndLoans                   | Open Accounts   | Total number of active credit cards and loans                               |
| NumberOfTimes90DaysLate                            | 3-Month Lates   | Number of times the borrower was 3+ months past due                          |
| NumberRealEstateLoansOrLines                       | Mortgages       | Number of real estate loans or lines                                        |
| NumberOfTime60-89DaysPastDueNotWorse              | 2-Month Lates   | Number of times the borrower was 2 months past due                           |
| NumberOfDependents                                 | Family Size     | Number of dependents (children, spouse, or others)                          |
