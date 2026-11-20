<p align="center">
  <img src="zassets/title.png" alt="Bad Debt Prediction" width="900">
</p>

---

<p align="center">
<img src="https://img.shields.io/badge/Python-306998?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-4D77CF?style=flat-square&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Pandas-4F5B93?style=flat-square&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-5B6FA6?style=flat-square&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/DVC-6B5CA5?style=flat-square&logo=dvc&logoColor=white" />
<img src="https://img.shields.io/badge/MLflow-4B78A8?style=flat-square&logo=mlflow&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-3A8F8F?style=flat-square&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/PSI%20%2F%20CSI-6666A8?style=flat-square&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-367FA9?style=flat-square&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/AWS%20SageMaker-B8792E?style=flat-square&logo=amazonaws&logoColor=white" />
<img src="https://img.shields.io/badge/DagsHub-222222?style=flat-square&logo=dagshub&logoColor=white" />
</p>

------

## 1. Problem Statement

Buy Now, Pay Later (BNPL) is a short-term financing model that allows customers to purchase products or services and repay the amount through scheduled instalments. While this provides flexibility to customers, defaults on instalment payments can create **bad debt, financial losses and cash-flow challenges** for BNPL businesses. Therefore, understanding customer creditworthiness and repayment behaviour is important for managing credit risk.

This project focuses on developing a **machine learning-based Bad Debt Prediction system** to identify customers who are more likely to default on their instalment payments. It uses customer details, loan information, credit bureau data and payment history to classify customers as **Good Debt or Bad Debt**, supporting better-informed lending decisions.

The primary objective is to **minimise potential financial losses caused by customer defaults** by identifying as many high-risk customers as possible before lending decisions are made. Accordingly, the project prioritises **high Bad Debt recall** to reduce the number of actual defaulters missed by the model while supporting risk-based customer assessment.

## 2. Dataset Description

The project uses multiple datasets containing approximately **110,000 initial records and 99 features**, covering four major information categories.

| Dataset Category | Description |
|---|---|
| Customer Data | Customer profiles and personal information. |
| Loan Data | Loan details, amounts and repayment-related information. |
| Credit Bureau Data | Credit scores, credit enquiries, account history and outstanding credit exposure. |
| Payment History Data | Previous repayments, defaults and delinquency behaviour. |

These datasets were integrated using customer identifiers to create a unified Credit Risk Master Table containing **91,849 records and 99 features**.

**Target Variable:** `OUTCOME`

- **Good Debt:** Customers classified as non-defaulters.
- **Bad Debt:** Customers classified as defaulters.

---

## 1. Data Profiling and Preparation

I started by exploring approximately **110,000 records with 99 features**, covering customer details, loan information, credit bureau data and payment history. The initial analysis focused on understanding the data quality, including missing values, duplicate records, data types and invalid entries. During this process, **3,644 duplicate records** were identified and removed.

Some features had a considerable amount of missing data, with missing values ranging from **30–40%** in certain columns. Rather than removing all such features or filling missing values blindly, I examined their patterns and retained those that still provided useful information for credit risk assessment.

However, six features — **`WORST_1M`, `WORST_3M`, `WORST_6M`, `WORST_12M`, `WORST_18M` and `WORST_24M`** — had approximately **90% missing values** and offered limited additional information. I decided to remove them from the analysis.

I also excluded **`SCORE_INDICATOR_CR22`** after observing a non-monotonic relationship with default risk, where higher scores were associated with higher default rates. Since this behaviour was difficult to interpret from a credit risk perspective and raised concerns about the feature's reliability, I decided not to include it in the modelling process.

Collection-related variables, including **`AGENCY_AMT` and `MTH_TILL_AGENCY`**, were also reviewed as part of the feature assessment.

After the initial cleaning and feature assessment, the customer, loan, credit bureau and supplementary datasets were merged using customer identifiers. This resulted in **12,611 unmatched customer IDs** being excluded, followed by the removal of another **1,896 unmatched records** during validation.

After these steps, the consolidated **Credit Risk Master Table** contained **91,849 records and 89 features**. This dataset was then used for the subsequent exploratory data analysis and model development.


- **Number of Observations** - A total of 110,000 observations (99 variables) was explored.The following features of the data were analysed in order to understand the quality of the dataset: data types, missing data, duplicate data, and incorrect data.Within the data, 3,644 observations of duplicate data were identified and removed.

- **Missing Data** - It can be seen that within the data there is a large amount of missing data present. Approximately, there are 30-40% of missing data was present in some variables. However, it was considered whether the missing data could be useful to understand credit risk and not immediately removing all variables with missing data.
    
However, six variables — `WORST_1M, WORST_3M, WORST_6M, WORST_12M, WORST_18M and   WORST_24M` --- had approximately 90% missing data. These were removed due to little extra useful information being provided by  the variables. 

After analysis, it was found that the variable `SCORE_INDICATOR_CR22` did not have a logical relationship with the risk of default. As higher scores corresponded to higher default risks, this variable was removed. Collection-related variables, including AGENCY_AMT and MTH_TILL_AGENCY, were also reviewed as part of the feature assessment.  

- **Combining Datasets and Validating**- Datasets relating to customers, loans, credit bureau information, and additional information were combined based on customer identifiers. During combination, 12,611 customer identifiers were not found in all datasets. Therefore, these were removed.Further validation was performed on individual observations. This meant an additional 1,896 observations were removed.

As a result, a single Credit Risk Master Table was created containing 91,849 observations, and 89 variables.


## 2. Data Splitting and Leakage Prevention

Before performing exploratory data analysis and making feature-related decisions, I split the dataset into **75% training data and 25% testing data**.

This was done to prevent data leakage and ensure that information from the test set did not influence exploratory analysis, feature selection or preprocessing decisions. The training set was used for data exploration, feature analysis and model development, while the test set was kept separate for final model evaluation.

This approach helped maintain an independent test dataset and provided a more reliable assessment of how the model performs on unseen data.



## 3. Exploratory Data Analysis (EDA)

I started by examining the distribution of the target variable, `OUTCOME`, which revealed a significant class imbalance. Approximately **92% of customers belonged to Good Debt and 8% to Bad Debt**. I also explored numerical and categorical features using descriptive statistics, histograms, boxplots and frequency distributions to understand the underlying patterns in the data.

I compared the **CR21 and CR22 credit bureau features** across Good Debt and Bad Debt customers using box plots to understand how their distributions and median values differed between the two groups. The comparison showed more noticeable differences in several CR22 features, making them more relevant for further credit risk analysis. Based on these observations, I selected relevant CR22 features for subsequent analysis and model development.

I also examined previous defaults, repayment patterns and delinquency indicators to understand how customer payment behaviour differed between Good Debt and Bad Debt customers.

To understand the relationship between credit exposure and repayment behaviour, I examined scatter plots involving features such as `RISK_SCORE`, `SCORE_CR22`, delinquency counts and payment irregularity indicators against `CREDIT_CARD_AMT_CR22`, `OVERDRAFT_AMT_CR22`, `DEFAULT_AMT` and `DEFAULT_UNPAID_AMT_CR22`.

The plots showed that these **relationships were not always linear**. Some customers had relatively low outstanding or default amounts but still displayed high-risk payment behavior. Similarly, higher credit exposure did not necessarily mean a proportionately higher risk of default. This suggested that default risk was influenced by a combination of credit history, repayment behavior and credit exposure rather than any single feature.

I also examined **outliers using the Interquartile Range (IQR) method**. The highest outlier percentages were found in `CLOSED_ACCT_CR22` (24.11%), `CREDIT_CARD_AMT_CR22` (21.22%) and `Cumulative_Delinquency_Exposure` (17.93%). After reviewing these features, I retained the outliers because many represented genuine customer credit and payment behaviour rather than data errors.

Based on these findings, I assessed the features according to their statistical patterns, observed behaviour and relevance to credit risk. For further analysis, they were organised into four categories: **Credit Risk, Payment Behaviour, Credit Exposure and Customer Profile**.



## 3. Model Development

After splitting the dataset into training, validation and test sets, **SMOTE-Tomek was applied only to the training data** to address class imbalance. The validation and test sets were kept unchanged to ensure a fair evaluation of model performance.

Three machine learning models were developed: **Random Forest, XGBoost and CatBoost**. For each model, the classification threshold was selected using validation data before evaluating the final performance on the test set.

| **Model** | **Bad Debt Recall** | **Threshold** | **ROC-AUC** | **Precision** |
|---|---:|---:|---:|---:|
| **Random Forest** | **81.39%** | **0.26** | **0.7377** | **12.04%** |
| XGBoost | 74.49% | 0.10 | 0.6892 | 11.40% |
| CatBoost | 85.08% | 0.09 | 0.7208 | 11.51% |

#### Selected Model: Random Forest

Random Forest was selected as the final model after comparing the results of all three algorithms.

Compared with XGBoost, Random Forest achieved **higher Bad Debt recall (81.39% against 74.49%)** and a **higher ROC-AUC (0.7377 against 0.6892)**. This indicates that it identified more actual Bad Debt cases and performed better at distinguishing between Good Debt and Bad Debt customers.

CatBoost achieved the **highest Bad Debt recall at 85.08%**. However, Random Forest recorded a slightly higher **ROC-AUC of 0.7377 compared with 0.7208 for CatBoost**. CatBoost also showed a larger gap between training and validation performance, suggesting potential overfitting.

Based on these results, Random Forest was chosen because it offered a **better overall balance between identifying Bad Debt cases, distinguishing between the two classes and generalising to unseen data**.

On the final test set, Random Forest achieved **81.39% Bad Debt recall, 12.04% precision, 20.98% F1-score, 0.7377 ROC-AUC and 0.2620 PR-AUC**, using a classification threshold of 0.26.

The five most influential features identified were `RISK_SCORE`, `SCORE_CR22`, `NO_ADDRESS_AT_MONTHS`, `OPEN_ACCT_CR22` and `CLOSED_ACCT_CR22`.

<div align="center">
  <img src="zassets/RF_Confusion_Matrix.png" alt="Random Forest Final Test Confusion Matrix" width="750">
</div>

<p align="center">
  <em>Figure: Random Forest final test confusion matrix and business interpretation.</em>
</p>

#### MLflow Experiment Tracking and Model Registry

MLflow was used to record and compare experiments conducted with Random Forest, XGBoost and CatBoost. Model parameters, classification thresholds and test performance metrics were logged for each experiment. All three trained models were also registered in the MLflow Model Registry, with individual versions maintained for tracking and future reference.

**Registered Models*

<div align="center">
  <img src="zassets/registered_table.png" alt="MLflow Registered Models" width="100%">
</div>

<p align="center">
  <em>Figure: MLflow Model Registry showing the three registered models and their versions.</em>
</p>

