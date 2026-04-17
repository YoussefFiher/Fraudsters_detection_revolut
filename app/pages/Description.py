import streamlit as st
import pandas as pd

st.title("Project Description")

st.markdown("""
This project aims to detect fraudulent users on the Revolut platform using a complete end‑to‑end machine learning pipeline, including:

- Exploratory data analysis  
- Advanced feature engineering (35 engineered features)  
- Undersampling to address class imbalance  
- Training multiple models (Logistic Regression, Random Forest, Gradient Boosting)  
- Threshold optimization to maximize recall  
- Saving trained models and optimal thresholds  
- Deployment through a Streamlit application  

The main objective is to capture as many fraudsters as possible while keeping false positives under control.
""")

st.markdown("""

""")

st.divider()

st.subheader("Data description")
st.markdown("""
The file **users.csv** contains data on a subset of fictional banking users. The abbreviation 'KYC' stands for 'Know Your Customer' - a process of identifying and verifying the client's identity when opening an account and periodically over time. The variable IS_FRAUDSTER from this dataset is your target variable. The file transactions.csv contains details of fictional transactions of these users.
The files countries.csv and currency_details.csv are dictionaries that provide explanations of abbreviations used in columns COUNTRY of the users dataset and CURRENCY of the transactions dataset respectively. These dictionaries may be useful but you don't need to use them when solving this task.
""") 

st.divider()

st.subheader("download dataset")

# Charger les datasets
df_users = pd.read_csv("datasets/users.csv")
df_transactions = pd.read_csv("datasets/transactions.csv")
df_countries = pd.read_csv("datasets/countries.csv")
df_currency = pd.read_csv("datasets/currency_details.csv")

# Convertir en CSV
csv_users = df_users.to_csv(index=False).encode("utf-8")
csv_transactions = df_transactions.to_csv(index=False).encode("utf-8")
csv_countries = df_countries.to_csv(index=False).encode("utf-8")
csv_currency = df_currency.to_csv(index=False).encode("utf-8")

# Créer 4 colonnes pour afficher les boutons côte à côte
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.download_button(
        label="Download users.csv",
        data=csv_users,
        file_name="users.csv",
        mime="text/csv"
    )

with col2:
    st.download_button(
        label="Download transactions.csv",
        data=csv_transactions,
        file_name="transactions.csv",
        mime="text/csv"
    )

with col3:
    st.download_button(
        label="Download countries.csv",
        data=csv_countries,
        file_name="countries.csv",
        mime="text/csv"
    )

with col4:
    st.download_button(
        label="Download currency_details.csv",
        data=csv_currency,
        file_name="currency_details.csv",
        mime="text/csv",
    )





