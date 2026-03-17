
from matplotlib.pyplot import xlabel, ylabel
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np 
import plotly.graph_objects as go

st.title("Data visualisation")

users_df = pd.read_csv("../datasets/users.csv")
transaction_df = pd.read_csv("../datasets/transactions.csv")

fraud_counts = users_df["IS_FRAUDSTER"].value_counts()
labels = ["Not Fraudster", "Fraudster"]

fig = px.pie(
    values=fraud_counts.values,
    names=labels,
    title="Fraud vs Non‑Fraud Distribution",
    color=labels,
    color_discrete_map={"Not Fraudster": "#69B687", "Fraudster": "#F44336"}
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
We observe a **strong class imbalance**:  
Most users are *not fraudsters*, while only a very small portion are labeled as *fraudsters*.  
This imbalance is typical in fraud detection and requires techniques like **undersampling** and **threshold tuning**.
""") 

st.divider()

kyc_percent = pd.crosstab(users_df["KYC"],
                        users_df["IS_FRAUDSTER"],
                        normalize="index"
                        )*100

kyc_fig = px.bar(kyc_percent , title= "Fraud distibution by kyc_status",barmode="group")
kyc_fig.update_layout( xaxis_title="KYC Status", yaxis_title="Percentage (%)", legend_title="is fraudster" )

st.plotly_chart(kyc_fig, use_container_width= True)

st.markdown(
            """
            #### what is KYC : 
            KYC (Know Your Customer) is a regulatory process used by financial institutions to verify the identity of their users. It typically includes checking government IDs, verifying addresses, and validating personal information. 
            A strong KYC process helps prevent fraud, money laundering, and identity theft. 
            #### interpretation of the chart: 
            PENDING KYC shows the highest fraud rate (~22%)**. This suggests that fraudsters may attempt to perform transactions **before their identity verification is completed**, taking advantage of the temporary verification gap. - **NONE KYC shows almost 0  fraud**. Accounts with no KYC usually have **restricted functionality**, meaning they cannot perform high‑risk or high‑value transactions. This naturally limits fraud attempts. - **PASSED and FAILED KYC show moderate fraud levels**. Fraudsters may still attempt to pass KYC using stolen or synthetic identities, while failed KYC users may try to exploit limited features before being fully blocked. Overall, the chart highlights how **KYC status is strongly correlated with fraud risk**, and why monitoring users during the verification process is critical. 
             """)

st.divider()

# Identify fraudsters
fraudster_users = set(users_df.loc[users_df["IS_FRAUDSTER"] == True, "ID"])
transaction_df["is_fraudster_beh"] = transaction_df["USER_ID"].isin(fraudster_users)

# Extract amounts
fraud_amount_dist = np.log10(transaction_df.loc[transaction_df["is_fraudster_beh"], "AMOUNT_USD"] + 1)
normal_amount_dist = np.log10(transaction_df.loc[~transaction_df["is_fraudster_beh"], "AMOUNT_USD"] + 1)

# Create figure
fig = go.Figure()

# Fraudster distribution
fig.add_histogram(
    x=fraud_amount_dist,
    name="Fraudster",
    opacity=0.6,
    marker_color="#F44336"
)

# Legit distribution
fig.add_histogram(
    x=normal_amount_dist,
    name="Legit",
    opacity=0.6,
    marker_color="#4CAF50"
)

# Layout
fig.update_layout(
    title="Transaction Amount Distribution (log10 scale)",
    xaxis_title="log10(Transaction Amount + 1)",
    yaxis_title="Count",
    barmode="overlay"   # superposé (ou "group" pour côte à côte)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(""" 

Fraudsters (red) show a right-skewed distribution toward higher amounts (log10 values 4-6)
Non-fraudsters (green) concentrate around lower amounts (log10 values 2-3)
This translates to fraudsters averaging 30,000 compared to 500 for non-fraudsters

""") 

st.divider()

st.subheader("Fraud Distribution Visualizations")
st.write("Select a category to display its fraud distribution chart.")

# --- TYPE ---
type_fraud = transaction_df.groupby(["TYPE","is_fraudster_beh"]).size().unstack().fillna(0)
type_fraud = type_fraud.rename(columns={False: "Legit", True: "Fraudster"})
type_fraud_perc = type_fraud.div(type_fraud.sum(axis=1), axis=0) * 100

type_fraud_fig = px.bar(
    type_fraud_perc,
    barmode="group",
    title="Fraud Distribution by Transaction Type",
    color_discrete_map={"Legit": "#4CAF50", "Fraudster": "#F44336"}
)
type_fraud_fig.update_layout(
    xaxis_title="Transaction Type",
    yaxis_title="Percentage (%)",
    legend_title="User Type",
    title_x=0.5
)

# --- STATE ---
state_fraud = transaction_df.groupby(["STATE","is_fraudster_beh"]).size().unstack().fillna(0)
state_fraud = state_fraud.rename(columns={False: "Legit", True: "Fraudster"})
state_fraud_perc = state_fraud.div(state_fraud.sum(axis=1), axis=0) * 100

state_fraud_fig = px.bar(
    state_fraud_perc,
    barmode="group",
    title="Fraud Distribution by Transaction State",
    color_discrete_map={"Legit": "#4CAF50", "Fraudster": "#F44336"}
)
state_fraud_fig.update_layout(
    xaxis_title="State",
    yaxis_title="Percentage (%)",
    legend_title="User Type",
    title_x=0.5
)

# --- ENTRY METHOD ---
entry_method_grp = transaction_df.groupby(["ENTRY_METHOD","is_fraudster_beh"]).size().unstack().fillna(0)
entry_method_grp = entry_method_grp.rename(columns={False: "Legit", True: "Fraudster"})
entry_method_grp_tab = entry_method_grp.div(entry_method_grp.sum(axis=1), axis=0) * 100

entry_method_fig = px.bar(
    entry_method_grp_tab,
    barmode="group",
    title="Fraud Distribution by Entry Method",
    color_discrete_map={"Legit": "#4CAF50", "Fraudster": "#F44336"}
)
entry_method_fig.update_layout(
    xaxis_title="Entry Method",
    yaxis_title="Percentage (%)",
    legend_title="User Type",
    title_x=0.5
)

# --- BUTTONS ---
col1, col2, col3 = st.columns(3)

with col1:
    show_type = st.button("Transaction Type")

with col2:
    show_state = st.button("State")

with col3:
    show_entry = st.button("Entry Method")

# --- DISPLAY CHART BASED ON BUTTON ---
if show_type:
    st.plotly_chart(type_fraud_fig, use_container_width=True)
    st.markdown("""
                
                
        - BANK_TRANSFER: Dominated by fraudsters (7.70%) - enables quick cash-out
        - ATM: Higher fraud proportion (~4.66%) - immediate cash withdrawal
        - P2P & CARD_PAYMENT: Dominated by non-fraudsters - normal everyday usage
        - TOPUP: Mixed usage, slightly higher fraud proportion
        - These behavioral differences will be valuable features for our fraud detection model. 
""")
    

elif show_state:
    st.plotly_chart(state_fraud_fig, use_container_width=True)
    st.markdown("""
    
    The distribution shows that declined transactions have the highest fraud presence, with fraudsters representing roughly 4.6% of all declined operations. This suggests that fraudulent users often trigger security rules or insufficient‑fund checks that lead to declines.
Cancelled and failed transactions follow, each with fraud rates around 3%, indicating that fraudsters may attempt actions that get interrupted or blocked during processing.
In contrast, completed and reverted transactions show very low fraud rates, meaning that most successful or reversed operations are performed by legitimate users.

""")

elif show_entry:
    st.plotly_chart(entry_method_fig, use_container_width=True)
    st.markdown("""
                

    - manu (Manual Entry): Highest fraud rate (~3.58%) - bypasses physical card verification
    - misc (Miscellaneous): Second highest (~2,44%) - often used for online
""")

st.divider()


st.markdown("""
These visualisations highlight only a small subset of the behavioural patterns present in the data, but they already show how fraudsters differ from legitimate users. A machine learning model can learn these patterns at scale and combine them to make accurate predictions. For example, a user with a pending KYC status who performs mostly bank transfers may exhibit a risk profile similar to known fraudsters, increasing the likelihood that the model classifies them as suspicious. These are just a few features among many; additional graphs could explore relationships involving attributes such as email verification, device information, transaction timing, and more. Together, these signals help the model build a comprehensive understanding of fraudulent behaviour.""")


