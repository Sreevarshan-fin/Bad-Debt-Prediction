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

### 3. Model Development

The target variable showed class imbalance, with more Good Debt customers than Bad Debt customers. Applied **SMOTE-Tomek** after splitting the dataset into training, validation and test sets, using resampling only on the training data.

Trained and evaluated **Random Forest, XGBoost and CatBoost** models, with threshold optimisation using validation data. Bad Debt recall achieved was **81.39% for Random Forest (0.26), 74.49% for XGBoost (0.10) and 85% for CatBoost (0.09)**.

Model diagnostics showed no clear overfitting or underfitting in Random Forest and XGBoost, while CatBoost showed possible overfitting (training-to-validation ROC-AUC gap of **0.0601**). XGBoost's feature importance was also less aligned with the business context.

Tracked model parameters and evaluation metrics using **MLflow** and registered the selected model in the Model Registry.

Selected **Random Forest** for its overall classification performance and business relevance. It achieved **81.39% Bad Debt recall, 0.7377 ROC-AUC and 0.2620 PR-AUC**. Its feature importance results highlighted relevant credit-risk indicators, including `RISK_SCORE`, `SCORE_CR22`, `NO_ADDRESS_AT_MONTHS`, `OPEN_ACCT_CR22` and `CLOSED_ACCT_CR22`. Although CatBoost achieved higher recall, Random Forest provided a suitable balance between identifying potential bad-debt customers and generalisation, making it the preferred model for further development

--------------------------

# Bad Debt Prediction for a Buy Now, Pay Later (BNPL) Business

## Problem Statement

In a Buy Now, Pay Later business, understanding how likely a customer is to default on instalment payments is central to managing credit risk. Defaults create bad debt, financial losses and cash-flow pressure.

This project set out to build a machine learning-based Bad Debt Prediction system that flags customers with a higher likelihood of default. It draws on customer details, loan information, credit bureau data and payment history to classify customers as Good Debt or Bad Debt, helping the business make better-informed lending decisions.

## 1. Data Profiling and Preparation

I began by exploring roughly 110,000 records across 99 features, covering customer details, loan information, credit bureau attributes and payment history. Data quality was checked for missing values, duplicates, data types and invalid entries. The initial cleaning removed 3,644 duplicate records.

Some features showed 30–40% missingness. I examined the patterns behind these gaps and kept them where they carried meaning, rather than filling them in by default. Lending-related features were also reviewed to understand their characteristics and their relevance to default prediction.

The customer, loan, credit bureau and supplementary datasets were then merged using customer identifiers. This excluded 12,611 unmatched customer IDs, and a further 1,896 unmatched records were removed during validation. The resulting Credit Risk Master Table held 91,849 records and 99 features, and formed the basis for all later analysis and modelling.

## 2. Exploratory Data Analysis (EDA)

The OUTCOME variable showed clear class imbalance, with about 92% of customers classed as Good Debt and 8% as Bad Debt. Numerical and categorical features were examined through descriptive statistics, histograms, boxplots and frequency distributions to understand their patterns.

CR21 and CR22 credit bureau features were compared across the two customer groups, and several CR22 features showed a clearer difference in medians. Repayment patterns, previous defaults and delinquency indicators were also reviewed to see how payment behaviour differed between Good Debt and Bad Debt customers.

Outliers were assessed using the Interquartile Range (IQR) method. The highest outlier percentages appeared in CLOSED_ACCT_CR22 (24.11%), CREDIT_CARD_AMT_CR22 (21.22%) and Cumulative_Delinquency_Exposure (17.93%). Because many of these were genuine high-value observations, I retained them rather than removing them on statistical grounds alone.

Features were selected based on statistical findings, observed behaviour and relevance to default prediction. They were grouped into four categories: Credit Risk, Payment Behaviour, Credit Exposure and Customer Profile.

## 3. Model Development

To address the class imbalance, SMOTE-Tomek was applied after the dataset had been split into training, validation and test sets. Resampling was performed on the training data only, so the validation and test sets remained representative of real conditions.

Three models were trained and evaluated: Random Forest, XGBoost and CatBoost. Decision thresholds were tuned using the validation data. The Bad Debt recall for each model was:

- **Random Forest:** 81.39% (threshold 0.26)
- **XGBoost:** 74.49% (threshold 0.10)
- **CatBoost:** 85% (threshold 0.09)

Diagnostics showed no clear signs of overfitting or underfitting in Random Forest or XGBoost. CatBoost, however, showed possible overfitting, with a training-to-validation ROC-AUC gap of 0.0601. The feature importance results from XGBoost were also less consistent with the business context.

Model parameters and evaluation metrics were tracked with MLflow, and the selected model was registered in the Model Registry.

Random Forest was chosen for its overall classification performance and business relevance. It delivered a Bad Debt recall of 81.39%, a ROC-AUC of 0.7377 and a PR-AUC of 0.2620. Its feature importance results pointed to sensible credit-risk indicators, including RISK_SCORE, SCORE_CR22, NO_ADDRESS_AT_MONTHS, OPEN_ACCT_CR22 and CLOSED_ACCT_CR22.

Although CatBoost achieved higher recall, Random Forest offered a better balance between identifying potential bad-debt customers and generalising to new data. This made it the preferred model for further development.

