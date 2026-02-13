# Credit Risk & Approval Decision Model

## 📌 Project Overview

This project develops a predictive model to assess the likelihood of credit card approval based on applicant financial characteristics.

The objective is to simulate a credit risk decision-support tool that assists financial institutions in evaluating application risk, improving consistency, and supporting data-driven approval decisions.

This case study demonstrates structured data preprocessing, model development, validation, and performance evaluation within a credit risk context.

---

## 📊 Problem Context

Financial institutions must balance:

- Customer acquisition growth
- Risk exposure
- Capital efficiency
- Regulatory compliance

An automated approval support model can:

- Improve decision consistency
- Reduce manual review time
- Highlight high-risk applications
- Support data-backed approval policies

---

## 🧠 Dataset Overview

The dataset includes applicant features such as:

- Income
- Employment status
- Credit history
- Debt indicators
- Other financial variables

The target variable indicates whether an application was approved.

---

## 🔎 Methodology

### 1️⃣ Data Cleaning & Preparation

- Handled missing values using logical imputation strategies
- Encoded categorical variables
- Standardised numeric features where required
- Checked class balance of approval outcomes

---

### 2️⃣ Exploratory Data Analysis

- Analysed distribution of applicant characteristics
- Identified correlations between financial indicators and approval likelihood
- Assessed potential multicollinearity

---

### 3️⃣ Model Development

Tested multiple supervised learning algorithms:

- Logistic Regression
- K-Nearest Neighbours
- Ridge / Lasso Regression
- Other baseline classifiers

Used structured pipelines to:

- Automate preprocessing
- Prevent data leakage
- Ensure reproducibility

---

### 4️⃣ Model Evaluation

Models were evaluated using:

- Cross-validation
- Accuracy
- R² (where applicable)
- Confusion matrix interpretation
- Precision / Recall considerations (risk trade-off context)

Hyperparameters were tuned using:

- GridSearchCV
- RandomizedSearchCV

---

## 📈 Key Insights

- Logistic Regression provided interpretable results aligned with credit risk modelling practices.
- Feature importance analysis highlighted strong influence from income stability and prior credit history.
- Cross-validation reduced overfitting risk and improved generalisation reliability.
- Trade-offs between false approvals and false rejections were considered to reflect real-world lending risk balance.

---

## 🛠 Tools & Technologies

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Pipeline  
- GridSearchCV  
- RandomizedSearchCV  
- Matplotlib  

---

## 📁 Repository Structure
├── credit_data.csv

├── modelling_notebook.ipynb

├── preprocessing.py

├── modelling.py

└── README.md


---

## 💼 Business Relevance

Predictive approval models are widely used in:

- Credit card lending
- Mortgage underwriting
- Personal loan assessment
- Fraud detection
- Risk scoring systems

This project demonstrates capability in:

- Building structured predictive models
- Applying validation frameworks
- Managing model risk considerations
- Interpreting outputs in a commercial context

---

## 🚀 Skills Demonstrated

- Data preprocessing & transformation  
- Supervised learning model development  
- Hyperparameter tuning  
- Cross-validation techniques  
- Risk-oriented evaluation  
- Structured documentation  
