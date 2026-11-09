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



### Problem Statement

In a **Buy Now, Pay Later (BNPL) business**, assessing the likelihood of a customer defaulting on instalment payments is important for managing credit risk. Customer defaults can lead to bad debt, financial losses and cash-flow pressure.

This project focuses on developing a **Machine Learning-based Bad Debt Prediction system** to identify customers with a higher likelihood of default by analysing customer details, loan information, credit bureau data and payment history. The objective is to classify customers into Good Debt and Bad Debt categories and support more informed lending decisions.

### 1. Data Profiling & Preparation

Explored approximately **110,000 records across 99 features**, covering customer details, loan information, credit bureau attributes and payment history. Assessed data quality by checking missing values, duplicates, data types and invalid entries. Removed **3,644 duplicate records** during the initial cleaning process.

Identified **30–40% missingness** in selected features. Examined missing-value patterns and retained meaningful missing values where appropriate. Also reviewed lending-related features to understand their characteristics and relevance to default prediction.

Integrated customer, loan, credit bureau and additional information datasets using customer identifiers. During integration, excluded **12,611 unmatched customer IDs** and a further **1,896 unmatched records** during validation. The final Credit Risk Master Table contained **91,849 records and 99 features**, which was used for further analysis and model development.

### 2. Exploratory Data Analysis (EDA)

Analysed the `OUTCOME` variable and identified class imbalance, with approximately **92% Good Debt and 8% Bad Debt**. Explored numerical and categorical features using descriptive statistics, histograms, boxplots and frequency distributions to understand their patterns and distributions.

Compared **CR21 and CR22** credit bureau features across Good Debt and Bad Debt customers. Observed clearer median separation in selected CR22 features. Also examined repayment patterns, previous defaults and delinquency indicators to identify differences in payment behaviour between the two groups.

Performed outlier analysis using the **Interquartile Range (IQR)** method. Higher outlier percentages were observed in `CLOSED_ACCT_CR22` (**24.11%**), `CREDIT_CARD_AMT_CR22` (**21.22%**) and `Cumulative_Delinquency_Exposure` (**17.93%**). Retained genuine high-value observations rather than removing them solely based on statistical outlier detection.

Selected relevant features based on statistical findings, feature behaviour and their relevance to default prediction. Grouped the selected features into four categories: **Credit Risk, Payment Behaviour, Credit Exposure and Customer Profile**.
