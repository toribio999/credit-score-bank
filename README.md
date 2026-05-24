# 🏦 Credit Default Prediction

> End-to-end ML Project · Binary Classification · Data Cleaning · Feature Engineering · Logistic Regression · XGBoost + SHAP + LIME 

![Python](https://img.shields.io/badge/Python-3.14-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-teal)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.44-purple)
![LIME](https://img.shields.io/badge/LIME-0.2.2-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-black)
![NumPy](https://img.shields.io/badge/NumPy-1.x-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-red)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-lightblue)
---

## Summary

This project develops a comprehensive system aimed at predicting credit default risk from structured financial data. The full analytical lifecycle is covered, from exploratory data analysis (EDA) and feature engineering through to training, validation and evaluation of machine learning models, also incorporating explainability techniques via SHAP and LIME.

Several approaches were evaluated, including Logistic Regression, XGBoost and LightGBM, with particular attention paid to the strong class imbalance present in the dataset (93% non-default / 7% default) using strategies suited to this type of problem.

Ultimately, XGBoost was selected as the final model due to its solid predictive performance and its ability to handle this scenario effectively. The model estimates the probability that a customer will experience financial difficulties within a two-year horizon, enabling better decision-making in credit granting and risk management processes.

## 🎯 Key Highlights

- Developed a machine learning model to predict credit default risk using a dataset of over 100,000 records with a highly imbalanced target variable (93% non-default vs. 7% default).
- Designed and implemented a comprehensive data cleaning and preprocessing pipeline to ensure data quality prior to modelling. This included detecting and removing erroneous or inconsistent records, handling outliers and unrealistic values, and applying tailored missing-value treatment strategies based on the characteristics of each variable.
- Conducted an extensive Exploratory Data Analysis (EDA) to understand the dataset and identify the main drivers of default risk. The analysis covered feature distributions, target imbalance, risk profile segmentation, the relationship between delinquency history and default probability, and age-group analysis to uncover behavioural differences across customer cohorts.
- Performed feature engineering to create meaningful variables that better captured customers’ credit behaviour and payment history, improving the model’s predictive performance.
- Focused model optimisation on balancing minority-class recall and precision through hyperparameter tuning and threshold selection. Following business requirements, a minimum recall of 0.65 for the default class was established, and the configuration with the highest minority-class F1-score meeting this constraint was selected.
- Selected XGBoost as the final model, achieving an AUC-PR of 0.91, a particularly relevant metric for highly imbalanced classification problems. Using an optimised decision threshold of 0.3268, the model achieved a recall of 0.66, precision of 0.40, and F1-score of 0.50 for the default class, delivering robust performance despite the challenging class distribution.
- Analysed model interpretability using SHAP and LIME, revealing that variables related to payment delinquency history and accumulated overdue events were the strongest predictors of future default risk.



## 📊 Dataset
This project was developed using the dataset: '[Give me some credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)'.
This dataset includes financial and behavioural information about credit applicants. Each row represents a person applying for a loan and includes attributes such as income, debt, payment history, number of open accounts and family size. This data allows analysis of default risk and prediction of the probability that an applicant will fail to repay their debt.

| Columns                              | Simplified Name    | Description                                                          |
| ------------------------------------ | ------------------ | -------------------------------------------------------------------- |
| SeriousDlqin2yrs                     | Defaulter          | Binary variable indicating whether the person failed to pay their debt for more than 90 days (1 = Yes, 0 = No) |
| RevolvingUtilizationOfUnsecuredLines | Credit Utilisation % | Percentage of available credit currently being used               |
| age                                  | Age                | Age of the borrower in years                                         |
| NumberOfTime30-59DaysPastDueNotWorse | 1-Month Delays     | Number of times the borrower was 1 month late on a payment           |
| DebtRatio                            | Debt vs Income     | Monthly debt and expenses divided by total income                    |
| MonthlyIncome                        | Monthly Income     | Gross monthly income of the borrower                                 |
| NumberOfOpenCreditLinesAndLoans      | Open Accounts      | Total number of active credit cards and loans                        |
| NumberOfTimes90DaysLate              | 3-Month Delays     | Number of times the borrower was 3 or more months late on a payment  |
| NumberRealEstateLoansOrLines         | Mortgages          | Number of real estate loans or lines of credit                       |
| NumberOfTime60-89DaysPastDueNotWorse | 2-Month Delays     | Number of times the borrower was 2 months late on a payment          |
| NumberOfDependents                   | Family Size        | Number of dependants (children, spouse or others)                    |



## 🧹 1. Data Cleaning Process

In this phase, various cleaning tasks were carried out to improve the quality and consistency of the dataset, removing inconsistent records prior to modelling. Additionally, missing values present in the data were also addressed.


### 1.1 Initial Data Cleaning


- **`age` depuration:** observations with implausible ages were removed, specifically values below 18 and above 110.
- **Removal of irrelevant columns:** `Unnamed: 0` was discarded, as it only corresponded to a residual index.
- **Outlier treatment:** anomalous records were detected in the variables `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse` and `NumberOfTime90DaysPastDueNotWorse`, related to payment delinquency history. These cases were removed to avoid distortions in the analysis. Records removed for this last reason represented only 270 observations, so the information loss was minimal relative to the total dataset size.


### 1.2 Missing Data

Missing values were analysed to assess both their distribution and potential impact on model reliability. The inspection revealed that only two variables contained missing observations: MonthlyIncome and NumberOfDependents.

![Description](images/missing_values.png)


Given the limited scope of missingness, variable-specific imputation strategies were applied rather than global deletion. This approach preserves data volume while avoiding the introduction of bias that could arise from removing informative observations.

In particular, MonthlyIncome, a key variable in credit risk assessment, was treated carefully to ensure that imputation did not distort its distribution, while NumberOfDependents was handled using a more straightforward strategy due to its lower predictive sensitivity.

#### 1.2.1 NumberOfDependents

First, the variable `NumberOfDependents` will be addressed, as it is more intuitive. To better understand it, let us look at the distribution of its values:
  
Dependants | No. of clients
-----------|---------------
0          | 86,705
1          | 26,292
2          | 19,501
3          | 9,479
4          | 2,860
5          | 745
6          | 158
7          | 51
8          | 24
9          | 5
10         | 5
13         | 1
20         | 1

The variable's distribution shows that the vast majority of customers have between 0 and 2 dependants, concentrating most observations. Clear outliers are also identified (such as 10, 13 and 20 dependants), whose frequency is extremely low and therefore not representative of the dataset. Consequently, these outliers have been removed to avoid distortions in the analysis. For imputing missing values in the remaining observations, the mode (0) was used, as it is the most frequent and representative value of the distribution.

### 1.2.2 Monthly Income

`MonthlyIncome` requires a more careful treatment due to its relatively high proportion of missing values (19.77%) and its strongly right-skewed distribution, further affected by extreme outliers.

The distribution of this variable, segmented by default status and after trimming extreme values, is shown below:

![Description](images/income_distr.png)

A clear difference in distribution between default and non-default groups can be observed, suggesting that income contains predictive signal with respect to the target variable. For this reason, a global imputation strategy could distort this relationship and introduce bias into the model.

Missing values were therefore imputed using the median computed separately for each default class. This group-wise approach helps preserve the underlying relationship between income and default behaviour.

In addition, a logarithmic transformation (`log1p`) was applied to reduce skewness and mitigate the influence of extreme values, resulting in a more stable and model-friendly distribution.

```python
# Log transformation to reduce skewness
df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome"])

# Group-wise median imputation by target class
df["MonthlyIncome_log"] = df.groupby("SeriousDlqin2yrs")["MonthlyIncome_log"] \
    .transform(lambda x: x.fillna(x.median()))
```





## 📈 2. Exploratory Analysis

The Exploratory Data Analysis (EDA) phase was conducted to gain a comprehensive understanding of the dataset and identify the key factors associated with credit default risk. This stage involved examining the distribution of variables, detecting missing values and outliers, analyzing relationships between features and the target variable, and assessing potential patterns within the data. Given the highly imbalanced nature of the dataset, special attention was paid to understanding the characteristics of defaulted and non-defaulted customers. The insights obtained during this analysis guided feature engineering decisions, data preprocessing strategies, and the subsequent development of predictive machine learning models

### 2.1 Variable Distribution Analysis

In this section, the distribution of numerical variables was analysed with the aim of better understanding the dataset structure, identifying skewness, detecting extreme values and anticipating potential preprocessing needs before modelling. 

#### 2.1.1 Predictor Variable Distributions
This subsection explores the statistical distribution of the predictor variables. 
To facilitate visualisation, some variables were represented with visual trimming at the 99th percentile for a cleaner display.

<img src="images/num_var_distribution.png" style="width: 1000px; height: auto;"/>

**Main observations:**

- Variables such as `MonthlyIncome` and `DebtRatio` show strong positive skewness, with high concentration at low values and a long tail of extreme values.
- `RevolvingUtilizationOfUnsecuredLines` also shows right skew, with a large accumulation near zero and some high-value cases.
- `age` has a relatively stable distribution close to a unimodal shape, concentrating mainly between 35 and 65 years.
- Variables related to delinquency (`NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTimes90DaysLate` and `NumberOfTime60-89DaysPastDueNotWorse`) show strong zero-inflation, indicating that most customers have no recent payment delays.
- Variables such as `NumberRealEstateLoansOrLines`, `NumberOfOpenCreditLinesAndLoans` and `NumberOfDependents` have a discrete nature with clear concentrations at certain integer values.

**Implications for modelling:**

- The presence of skewness and outliers justifies the use of robust scaling techniques or transformations for certain variables.
- The zero-inflation observed in delinquency variables may provide high predictive power as it is closely tied to credit risk.
- This analysis is particularly relevant for linear models such as logistic regression, while tree-based models tend to be more robust to these types of distributions.


#### 2.1.2 Target Variable Distribution

This subsection examines the distribution of the target variable, with particular attention to its class imbalance, which may have important implications for model performance.

<p align="center">
  <img src="images/Target_distribution.png" width="500"/>
</p>

As it is clearly visible, the target variable shows a marked class imbalance, with non-default cases being overwhelmingly dominant over default events. This behaviour is expected in real credit portfolios, where the delinquency rate is typically low. However, this asymmetry may bias predictive model training towards the dominant class, so imbalance-robust metrics and specific techniques such as class weighting or resampling will be used during the modelling phase.

### 2.2 Default Behavior Analysis

Under this section, the analysis focuses on understanding how the likelihood of default changes across different customer groups and financial profiles. The study includes the estimation of the mean default rate together with both parametric (95% normal confidence intervals) and non-parametric (bootstrap confidence intervals) uncertainty estimates, providing a more robust view of risk behavior. In addition, differences between key segments — such as age ranges, income levels, delinquency history, and credit exposure — are explored to identify patterns associated with higher default probability. To complement the visual analysis, statistical techniques including Levene’s test and Cliff’s Delta are applied to evaluate variance differences and effect sizes between groups, helping distinguish statistically meaningful relationships from purely descriptive patterns.

#### 2.2.1 Mean Default Probability and Confidence Interval

The target variable `SeriousDlqin2yrs` is a binary indicator of whether a borrower experienced serious financial delinquency (90+ days past due) within a two-year window. Given its binary nature, standard descriptive statistics such as min, max, and percentiles are uninformative — the metrics below focus on class prevalence and imbalance, which directly influence modelling decisions around sampling strategy, class weighting, and evaluation metrics.

| Metric | Value |
|---|---|
| Mean | 6.60% |
| 95% Confidence Interval | [0.0647, 0.0672] |
| Default cases (1) | 9,878 |
| Non-default cases (0) | 139,839 |
| Class imbalance ratio | 14.2 : 1 |

The estimated mean default rate in the dataset is **0.0660**, indicating that approximately 6.6% of the observed individuals are classified as defaulters. To quantify the uncertainty of this estimate, we compute a 95% confidence interval using the Wilson method, obtaining a range of **[0.0647, 0.0672]**. This provides a precise analytical estimate of the true population default rate.

In addition to the analytical approach, we also estimate confidence intervals using bootstrap resampling. This method repeatedly samples from the dataset with replacement and recalculates the default rate, allowing us to empirically approximate its sampling distribution. The resulting interval does not rely on strong parametric assumptions and is therefore particularly useful for validating the robustness of the analytical estimate.

Together, both approaches provide complementary perspectives: the Wilson interval offers a closed-form statistical estimate, while bootstrap methods provide a data-driven, assumption-light validation of uncertainty.

<p align="center">
  <img src="images/bootstrap_default.png" width="800"/>
</p>

Bootstrap's approach gives us a similar 95% confidence interval of **[0.0647, 0.0674]**. The close agreement between both intervals reinforces the stability of the estimated default rate and suggests that the result is not sensitive to the underlying assumptions of the analytical method.

#### 2.2.2 Feature Distribution by Default Status

This section reveals clear differences between customers with and without payment default. On average, customers who do not default have higher monthly income, a slightly older age and lower debt ratios, while the default group concentrates profiles with lower income capacity and greater financial pressure. These variables show a consistent relationship with credit risk, making them particularly relevant for building predictive models aimed at estimating default probability and improving decision-making in credit granting.

<p align="center">
  <img src="images/feature_distr_default.png" width="800"/>
</p>

Comparison of key financial features between defaulting and non-defaulting borrowers reveals meaningful differences across all variables analyzed. Non-defaulters show a higher average revolving utilization of unsecured lines (~6.1 vs ~4.4), a higher monthly income (~$6,500 vs ~$5,500), and a higher debt ratio (~360 vs ~300), suggesting that defaulters tend to have a weaker overall financial profile despite lower absolute exposure. Age also differs notably: non-defaulters have a higher median age (~52) compared to defaulters (~46), consistent with the findings from the age-group analysis. While distributions are heavily right-skewed and contain extreme outliers across all features, the mean differences are statistically distinguishable, making these variables relevant predictors for credit risk modeling.


#### 2.2.3 Delinquency and Default History

A borrower's history of late payments is often considered one of the most direct signals of future credit risk. The following analysis examines how the frequency of past delinquencies — across three severity buckets — relates to the likelihood of serious default, revealing a clear and consistent escalation in risk with each additional missed payment.

<p align="center">
  <img src="images/delinquency_history.png" width="800"/>
</p>


Past delinquency behavior proves to be one of the strongest indicators of future default risk. Across all three delinquency buckets — 30–59, 60–89, and 90+ days past due — defaulters show a substantially higher proportion of clients with at least one recorded delay compared to non-defaulters. The default probability curves further confirm a steep, monotonic increase with the number of delays: even a single 60–89 day late event raises the default probability to roughly 50%, and borrowers with repeated 90+ day delinquencies face default rates exceeding 65%. These patterns highlight delinquency history as a critical feature that should be prioritized in any predictive credit risk model.


#### 2.2.4 Analysis by Age Group

This analysis explores the relationship between customers' age and their credit behaviour, focusing on default probability and different levels of delinquency. Through segmentation by age groups, the aim is to identify risk patterns that can improve the predictive capacity of the credit risk model.
  
<p align="center">
  <img src="images/credit_behavior_age.png" width="700"/>
</p>

Analysis of default rates and delinquency patterns reveals a clear age-related trend: younger borrowers, particularly the 26–35 cohort, exhibit the highest credit risk across all metrics, including default rate (~11%), and late payment counts at every delinquency bucket (30–59, 60–89, and 90+ days). Risk decreases steadily with age, with the 65+ group showing the lowest default rate (~2.3%) and minimal delinquency counts. 

<p align="center">
  <img src="images/composite_risk_index.png" width="700"/>
</p>

This pattern is further confirmed by the Composite Credit Risk Index, where the 26–35 group scores closest to 1.0, while the 65+ group scores near zero. Notably, the 18–25 segment shows relatively moderate risk compared to 26–35, likely reflecting limited credit access rather than responsible behavior. These findings suggest that age is a meaningful predictor of credit risk and should be weighted accordingly in risk scoring models.



### 2.3 Correlations
- Finally, the correlation matrix is examined with the aim of identifying which variables show the greatest association with the target variable SeriousDlqin2yrs, as well as potential multicollinearity issues between features. This analysis is particularly useful for understanding which signals provide the most predictive value and for guiding both variable selection and the construction of new transformations to improve model performance and interpretability.
  
<p align="center">
  <img src="images/Corr.png" width="600"/>
</p>

**Observations**
- The correlation analysis shows that the target variable SeriousDlqin2yrs (default) is primarily associated with late payment behaviour indicators, with `weighted_late_score`, `TotalPastDue` and `NumberOfTimes90DaysLate` showing the highest positive correlations. This confirms that recent delinquency history is the main driver of default risk.
- Derived variables such as `HasSeriousDelinquency` and the various delay counters (30–59 and 60–89 days) also reinforce this signal, evidencing a coherent structure among related features. On the other hand, variables such as `age` and `CreditHistoryLength` show moderate negative correlations, suggesting that more mature profiles with a longer credit history tend to have a lower probability of default.
- In contrast, classic financial variables such as `DebtRatio` or `MonthlyIncome` have a weak relationship with the target variable, suggesting that in this dataset, historical behaviour is far more predictive than declared economic capacity. Finally, some multicollinearity is observed among delinquency-derived variables, which will be taken into account in later modelling phases to avoid redundancy and improve model interpretability.


## 🧩 3. Feature Engineering

This section summarises the derived variables created with the aim of improving the predictive capacity of the credit risk model. The transformations focus on capturing payment capacity, the customer's historical behaviour and their segmentation.



| Variable                | Type        | Description                                                                 | Risk Intuition                               |
| ----------------------- | ----------- | --------------------------------------------------------------------------- | -------------------------------------------- |
| `income_per_dependent`  | Numerical   | Monthly income divided by number of dependants (+1 to avoid division by zero) | Lower value → greater financial burden     |
| `utilization_capped`    | Numerical   | Credit utilisation capped between 0 and 1                                   | Reduces the impact of extreme values         |
| `CreditHistoryLength`   | Numerical   | Age - 18 (approximation of credit history length)                           | Longer history → lower risk                  |
| `TotalPastDue`          | Numerical   | Total number of payment delays                                              | More delays → higher risk                    |
| `weighted_late_score`   | Numerical   | Weighted score of delays by severity                                        | Penalises severe delinquencies more heavily  |
| `HasSeriousDelinquency` | Binary      | 1 if there is any delay >90 days                                            | Strong indicator of default                  |
| `high_utilization_flag` | Binary      | 1 if credit utilisation >80%                                                | High utilisation → higher risk               |
| `AgeGroup`              | Categorical | Age grouped into ranges                                                     | Captures life-cycle effects                  |
| `IncomeGroup`           | Categorical | Income quartiles                                                            | Socioeconomic segmentation                   |
| `DTICategory`           | Categorical | Debt-to-income ratio (DTI) categories                                       | Higher DTI → lower repayment capacity        |

Basic ratios such as `income_per_dependent` and `utilization_capped` adjust raw figures to reflect real financial pressure — the former accounts for household obligations, the latter clips reporting anomalies to a meaningful `[0, 1]` range. Delinquency is represented at three levels of granularity: a raw count (`TotalPastDue`), a severity-weighted score (`weighted_late_score`) that penalizes 90+ day lates more heavily than shorter delays, and a binary flag (`HasSeriousDelinquency`) that makes the critical threshold explicit for tree-based models. A high utilization flag encodes the industry-recognized >80% risk threshold as a discrete signal. Finally, age, income, and debt-to-income ratio are discretized into ordered categories — grounded in standard lending guidelines — to capture non-linear relationships and improve model interpretability for business stakeholders.

## 📊 4. ML Model Development

This stage focuses on developing predictive models to estimate the probability of customer default. Given the complexity of credit risk prediction and the inherent imbalance in the dataset, the modelling approach prioritises algorithms and strategies capable of effectively capturing minority class behaviour.

A baseline Logistic Regression model was first implemented to establish a reference performance level. Subsequently, a more advanced ensemble-based model (XGBoost) was developed to capture non-linear relationships and feature interactions that linear models may not fully represent.

The objective of this stage is not only to maximise predictive performance, but also to ensure that the final model provides a robust and reliable separation between default and non-default cases, suitable for real-world credit risk applications.


### 4.1 Evaluation Strategy and Threshold Optimization

The dataset exhibits a strong class imbalance (93% non-default vs 7% default), making accuracy an unreliable evaluation metric, as a naïve model predicting only the majority class would still achieve 93% accuracy without any predictive value.

For this reason, model optimisation focused on the F1-score of the positive class (defaults), as it provides a balanced trade-off between precision and recall in imbalanced classification problems.

In addition, a minimum recall constraint of 0.65 was introduced to ensure that at least two-thirds of actual default cases are correctly identified, reflecting the importance of reducing missed risky borrowers in a credit risk context.

Finally, AUC-PR (Area Under the Precision-Recall Curve) was used as a complementary evaluation metric, as it provides a more informative view of model performance under severe class imbalance.


### 4.2 Logistic Regression *(baseline)*

As a first step, as is common in the literature for this type of problem, a logistic regression model was fitted. Due to the class imbalance problem, the model was trained with balanced class weights (`class_weight='balanced'`).

| Metric    | Class 0 (non-default) | Class 1 (default) |
|-----------|-----------------------|-------------------|
| Precision | 0.97                  | 0.26              |
| Recall    | 0.87                  | 0.65              |
| F1-Score  | 0.92                  | 0.37              |
| **AUC-PR** | **0.8508** | **Threshold: 0.5800** |

After evaluating multiple models, the best-performing one achieves an AUC-PR of 0.8508 with an optimal threshold of 0.58. For the minority class (default), the model reaches the maximum F1-Score at the minimum required recall of 65%, thus meeting the objective of detecting a relevant proportion of real cases. However, this result is achieved with a precision of only 0.26, meaning that only 1 in 4 customers classified as default actually is. Consequently, the model generates a high volume of false positives, producing approximately three incorrect alerts for every correct one. Its F1-Score of 0.37 confirms low discriminative capacity, so this model will serve only as a comparative reference against more sophisticated approaches.

---

### 4.3 XGBoost *(final model)*

As a second approach, an XGBoost classifier was developed and optimised through randomized hyperparameter search with cross-validation. Designed to capture complex non-linear relationships in the data, the model substantially outperformed the Logistic Regression baseline across all key evaluation metrics.

| Metric    | Class 0 (non-default) | Class 1 (default) |
|-----------|-----------------------|-------------------|
| Precision | 0.97                  | 0.40              |
| Recall    | 0.93                  | 0.66              |
| F1-Score  | 0.95                  | 0.50              |
| **AUC-PR** | **0.9021** | **Threshold: 0.3268** |

This model required lowering the decision threshold to **0.3268** (well below the default 0.5) to achieve the target recall. This reflects that the model, trained on imbalanced data, tends to assign low probabilities to the minority class, and the classification threshold must be reduced to capture more real defaults.

With this adjustment, the model detects **66% of real defaults** (recall), with an associated precision of 40%: that is, out of every 10 customers classified as default, 6 actually are and 4 are false alarms. This trade-off is common and generally acceptable in credit risk contexts, where the cost of missing a default far exceeds that of investigating a false alarm. The **F1-Score of 0.50** reflects this balance in a highly challenging scenario. 

The most significant improvement lies in precision (+14 p.p.), which translates into an F1-Score 35% higher (0.50 vs. 0.37) and an AUC-PR 5 points greater, while maintaining virtually identical recall. In other words, XGBoost detects the same proportion of real defaults while generating considerably fewer false alarms.
In a credit scoring context, this difference has direct practical implications: a more refined list of at-risk customers reduces operational review costs and avoids unnecessary friction with customers who would not have defaulted. For all these reasons, XGBoost is selected as the final model for the project.


### 4.4 AUC-PR Curves

Both models were evaluated using the Precision-Recall curve, which is more informative
than ROC in the presence of class imbalance, as it focuses on the model's ability to
correctly identify the minority class.

<p align="center">
  <img src="images/AUC-PR_Curve.png" width="600"/>
</p>

The key difference lies in precision throughout
that range: XGBoost maintains consistently higher precision across all thresholds,
with a smoother and slower decay. Logistic Regression degrades more steeply and
erratically from the start, reflecting its weaker ability to rank positive cases
in an imbalanced setting. At any given recall level, XGBoost generates fewer false
positives, making it the more reliable model for risk prioritization.


## 🔧 5. Feature Importance

To ensure transparency of the final model (XGBoost), three complementary interpretability techniques were applied: **Feature Importance (Gain)** for a global view of each variable's predictive power, **LIME** for explanations at the individual instance level, and **SHAP** to understand the directional impact of each feature on predictions.

All three techniques converge on a clear conclusion: the customer's historical payment behaviour is, by far, the most determining factor in predicting default.

---

### 5.1 Gain-Based Importance

The gain importance chart reflects how much each variable contributes to reducing impurity in the model's trees. The two dominant features are `weighted_late_score` and `TotalPastDue`, with noticeably higher importance than the rest — roughly twice that of the third most relevant variable — indicating that the model relies very heavily on the customer's late payment history. Next come `HasSeriousDelinquency` and `NumberOfTimes90DaysLate`, which reinforce the same signal: severe and repeated delays are the most robust predictor of default. At a secondary level of importance are `high_utilization_flag` and `utilization_capped`, indicating that the level of available credit utilisation also provides valuable information, though well below the delinquency variables. The remaining features — income, debt-to-income ratio, number of open lines — contribute marginally in terms of gain.

---

<p align="center">
  <img src="images/Features_xgb.png" width="600"/>
</p>

### 5.2 LIME

The LIME explanation corresponds to a specific instance classified as **Default** and allows us to understand which factors led the model to that particular decision. The variable with the greatest negative weight (pushing towards default) is `MonthlyIncome_missing <= 0`, confirming that the absence of income data is the most determining individual signal in this case. This is followed by `MonthlyIncome > 7403` and `weighted_late_score <= 0`, which may seem counterintuitive — high income pushing towards default — but is explained by the fact that LIME analyses local combinations of conditions: in this specific profile, other risk factors prevail over income level. `TotalPastDue <= 0` and `MonthlyIncome_log > 8.91` also contribute negatively. In the opposite direction, `high_utilization_flag <= 0` and `NumberRealEstateLoansOrLines > 2` act as slight risk-mitigating factors for this particular instance.

---

<p align="center">
  <img src="images/Lime_xgb.png" width="600"/>
</p>

### 5.3 SHAP

####  5.3.1 SHAP Global Analysis

The SHAP analysis complements gain-based importance by adding the **direction** of each variable's effect on the default probability. `weighted_late_score` again leads: high values (in red) are associated with positive SHAP values, pushing the prediction towards default, while low values reduce risk. `TotalPastDue` shows a similar pattern though with less dispersion, and `utilization_capped` also has a positive impact when elevated.

A particularly relevant finding is the behaviour of `MonthlyIncome_missing`: the absence of income data generates strongly positive SHAP values (higher default risk), suggesting that the lack of income information is itself a risk signal that the model has learned to exploit. Conversely, high values of `MonthlyIncome` act as a protective factor, pushing predictions towards non-default. The variable `age` shows a moderate protective effect for older customers, consistent with the credit risk literature.

---


<p align="center">
  <img src="images/Shap_xgb.png" width="600"/>
</p>


#### 5.3.2 SHAP Dependence Plot

SHAP dependence plots provide a detailed view of how individual feature values influence model predictions while also revealing potential interaction effects with other variables. Each point represents an observation, where the x-axis shows the feature value and the y-axis shows its corresponding SHAP value (i.e., the contribution of that feature to the predicted probability of default). Color gradients highlight interactions with additional variables, helping to uncover relationships that may not be visible through traditional feature importance rankings.

<p align="center">
  <img src="images/shap_dependence.png" width="600"/>
</p>

The dependence plots confirm that **payment delinquency and outstanding debt are the strongest drivers of default risk**. The `weighted_late_score` feature exhibits a clear positive and non-linear relationship with its SHAP values: as the severity and frequency of late payments increase, the model assigns a substantially higher default risk, although the marginal impact gradually levels off at extreme values. A similar pattern is observed for `TotalPastDue`, where increasing amounts of overdue debt consistently raise the predicted probability of default before reaching a plateau. In contrast, `MonthlyIncome` shows a much weaker direct effect, with most observations concentrated around low SHAP values and only a limited number of extreme-income outliers. Additionally, the color distributions suggest interaction effects between delinquency-related variables and missing income information, indicating that the model relies more heavily on payment behavior than on income levels when assessing credit risk. Together, these results reinforce the importance of repayment history as the primary determinant of default predictions in the model.

### 5.4 Conclusion

All three interpretability techniques are consistent with each other and point to the same explanatory core: **late payment history and accumulated delinquency are the dominant predictors of default**, followed at a distance by credit utilisation level and the availability of income information. This coherence between global and local methods reinforces confidence in the model and facilitates its potential use in regulated environments where the explainability of credit decisions is a requirement.


## ➡️ 6. Next Steps

- **Expand exploratory data analysis (EDA)** to uncover deeper patterns and potential feature interactions not yet explored.
- **Experiment with advanced class imbalance techniques**, such as SMOTE or undersampling strategies, to further improve minority class detection.
- **Incorporate business-driven optimisation criteria**, translating model performance into financial impact:
  - *Recall:* quantify the cost of missed defaults in monetary terms.
  - *Precision:* assess the operational cost associated with false positives.
- **Explore ensemble approaches**, combining multiple models to improve robustness and predictive performance.













