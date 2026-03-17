# Fraudsters Detection Project  

## Description  
This project is a take‑home assignment inspired by the recruitment process at **Revolut**.  
The goal is to explore fictional banking data, identify fraudulent behavior, and build a machine learning model capable of predicting fraudsters.  

The work is implemented in a **Jupyter Notebook**, complemented by a **Streamlit application** for visualization and prediction.  

---

##  Data Files  
- **users.csv** : information about users  
- **transactions.csv** : details of user transactions  
- **countries.csv** : CSV file mapping countries information  
- **currency_details.csv** : CSV file mapping currency details

---

##  Project Workflow  

### 1. Data Exploration  
- Distribution analysis of user and transaction variables  
- Missing values and inconsistencies check  
- Preliminary observations on differences between fraudsters and non‑fraudsters  

### 2. Initial Data Analysis  
- Transactional patterns (amounts, frequencies, currencies)  
- User profile comparisons (age, country, KYC status)  
- Identification of potential fraud signals  

### 3. Feature Engineering  
New features created to enrich the model:  
- **Transaction frequency** : number of transactions per user  
- **Average transaction amount** : mean transaction value per user  
- **Currency diversity** : number of distinct currencies used  
- **Country risk indicator** : based on fraud distribution across countries  
- **KYC status features** : binary indicators derived from KYC information  

Each feature is motivated by business intuition (e.g., unusual transaction frequency or multi‑currency usage may indicate suspicious behavior).  

### 4. Model Building  
Three classification models were trained and compared:  
1. **Logistic Regression** : simple, interpretable baseline  
2. **Random Forest** : ensemble method robust to non‑linear relationships  
3. **Gradient Boosting** : powerful boosting algorithm for complex patterns  

Evaluation metrics: **Accuracy, Precision, Recall, F1‑score, ROC‑AUC**.  

### 5. Streamlit Application  
An interactive app was developed with three pages:  
- **Project description** : overview of the dataset and methodology  
- **Prediction page** : interface to test the model on new users  
- **Visualization page** : exploratory and comparative charts  

---

## Installation & Execution  

### Requirements  
- Python 3.8+  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`  


