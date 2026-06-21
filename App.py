import streamlit as st

st.title("Smart Loan Approval System")

income = st.number_input("Applicant Income")
loan_amount = st.number_input("Loan Amount")

if st.button("Predict"):
    st.success("Loan Approved")
