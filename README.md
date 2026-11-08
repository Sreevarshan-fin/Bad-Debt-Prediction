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
<img src="https://img.shields.io/badge/Evidently%20AI-5B63A8?style=flat-square&logoColor=white" />
<img src="https://img.shields.io/badge/PSI%20%2F%20CSI-6666A8?style=flat-square&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-367FA9?style=flat-square&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/AWS%20SageMaker-B8792E?style=flat-square&logo=amazonaws&logoColor=white" />
<img src="https://img.shields.io/badge/AWS%20ECS-B8792E?style=flat-square&logo=amazonaws&logoColor=white" />
<img src="https://img.shields.io/badge/AWS%20EC2-B8792E?style=flat-square&logo=amazonaws&logoColor=white" />
<img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white" />
<img src="https://img.shields.io/badge/GitHub-24292F?style=flat-square&logo=github&logoColor=white" />
<img src="https://img.shields.io/badge/DagsHub-222222?style=flat-square&logo=dagshub&logoColor=white" />
</p>

------

## 📌 Problem Statement

In a **Buy Now, Pay Later (BNPL)** business, approving a customer is not the end of the transaction—it is the beginning of a **credit risk**. When customers fail to repay their installments, businesses face **bad debt, revenue loss, and cash-flow pressure**.

Without a predictive system, identifying customers who are likely to default **before credit approval** is difficult. This creates a need for a data-driven approach that can assess customer credit behavior and identify potential risks in advance.

This project addresses the challenge by developing a **Machine Learning–based Bad Debt Prediction system** that identifies **high-risk customers before approval**, helping businesses make more informed lending decisions and strengthen **credit-risk management**.

---

# 1. Data Profiling & Preparation

**Dataset Understanding:** Started by exploring approximately **100,000 records with 99 features**, covering customer details, loan information, credit bureau data and payment history.

**Data Quality Assessment:** Checked the dataset for missing values, duplicates, incorrect data types and invalid records. Removed **3,644 duplicate records (3.64%)** during the initial cleaning process.

**Missing Value Investigation:** Found **30–40% missing values** in some features. Investigated their patterns and retained them where they represented meaningful business scenarios or the availability of customer information.

**Domain-Level Investigation:** Reviewed the features in the context of lending and credit risk to understand what they represented and how customer details, loan characteristics and repayment behaviour could relate to default risk.

**Data Integration:** Combined the customer, loan, credit bureau and additional information datasets using customer identifiers. Found **12,611 customer IDs** in the Additional Information dataset without matching credit-behaviour records. Excluded these records because the required behavioural information was unavailable.

**Data Reconciliation:** During subsequent processing, identified and removed another **4,507 duplicate records**.

**Final Dataset:** The resulting Credit Risk Master Table contained **91,849 records and 97 features**, which were used for further analysis and model development.

# 2. Exploratory Data Analysis (EDA)

**Target Variable Analysis:** Analysed the `OUTCOME` variable and found an imbalance between the two classes, with approximately **92% Good Debt and 8% Bad Debt**.

**Feature Distribution Analysis:** Explored numerical and categorical features using descriptive statistics, histograms, boxplots and frequency distributions to understand their patterns and variations.

**Credit Risk Analysis:** Compared CR21 and CR22 feature distributions to understand how credit-related indicators differed and how they were associated with default behaviour.

**Payment Behaviour Analysis:** Examined repayment patterns, previous defaults and delinquency indicators to identify differences between Good Debt and Bad Debt customers.

**Zero-Value Analysis:** Looked into features with a high proportion of zero values, such as `DEFAULT_AMT` (**95.27%**) and `DEFAULTS_12M` (**99.24%**), to understand whether these values reflected actual customer behaviour.

**Skewness Analysis:** Identified highly skewed financial features, particularly `PERSONAL_LOAN_AMT_CR22` (**110.71**) and `OVERDRAFT_AMT_CR22` (**103.22**).

**Outlier Analysis:** Used the Interquartile Range (IQR) method to examine potential outliers. Some features showed relatively high outlier percentages:
- `CLOSED_ACCT_CR22`: **24.11%**
- `CREDIT_CARD_AMT_CR22`: **21.22%**
- `Cumulative_Delinquency_Exposure`: **17.93%**

Reviewed these observations in their business context and retained genuine high-value records rather than removing them solely because they were statistical outliers.

**Feature Selection:** Selected relevant features by considering statistical findings, lending domain knowledge, business relevance and their relationship with default risk. Grouped the selected features into four areas: Credit Risk, Payment Behaviour, Credit Exposure and Customer Profile.

**Outcome:** Completed data profiling and exploratory analysis to understand the dataset, identify important patterns and prepare it for class-imbalance handling and credit-risk model development.
