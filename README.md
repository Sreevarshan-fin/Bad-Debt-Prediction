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

In the current business process, products or services are sometimes sold to customers who later default on payments, leading to **bad debts and financial losses**. These defaults not only impact revenue but also affect **cash flow and risk management**. Presently, there is no predictive system in place to **assess the creditworthiness or payment behavior** of a customer before the sale is made.
To address this issue, the project aims to develop a **predictive machine learning model** that can **identify potential bad debts in advance**, using historical customer and transaction data. By analyzing patterns in past purchases, payment behaviors, credit scores, and other related features, the model will enable the business to make data-driven decisions on whether to approve or reject a sale, thereby **minimizing financial risk** and improving overall profitability

---

## 1. Data Profiling and Preparation -- 94% 

- **Dataset Description** - The data comprises 110,000 observations and 99 variables. Data has been merged from different sources (customer, loan, credit bureau details and payment histories) to create a Credit Risk Master Table.
  
- **Data Quality Assessment** – Examined the initial dataset for data types, missing values, duplicate records and incorrect entries. Identified and removed **3,644 duplicate records**.

- **Missing Values:** Within the dataset, between **30-40%** of missing values were present. However, it was considered whether the missing values could provide useful information with regards to credit risk. Variables with missing values were therefore not immediately removed. However, **8 variables** had approximately **90%** missing values. These variables were removed as little additional useful information was provided by them. Upon analysis it was found that variable ‘ SCORE_INDICATOR_CR22’ did not fit logically with default risk. The higher the score, the higher the risk of default so this variable was removed.
  
- **Dataset Integration and Validation**- Data from the customer, loan, credit bureau and general information was merged according to customer identifier. During combination, **12,611**    customer identifiers were not found in all datasets. Therefore, these were removed. Further validation was performed on individual observations. This meant additional **1,896** observations were removed.

After this process **91,849 observations and 89 variables** remained in a single Credit Risk Master Table.

## 2. Data Splitting and Leakage Prevention

Split into a **test set (25%) and training set (75%)**. This ensured **no data leakage** occurred. The test set was not used at all during data exploration, features selection and preprocessing. The training set was used to perform data exploration, features analysis and model building. However, only the test set was used to evaluate the model's performance.
This means we have an independent test set which allows us to get a more accurate indication of how well the model would work with new da

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

