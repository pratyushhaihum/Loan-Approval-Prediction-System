import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .danger-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏦 Smart Loan Approval Prediction System")
st.markdown("---")

# Check if model files exist
model_exists = os.path.exists('loan_model.pkl') and os.path.exists('label_encoders.pkl') and os.path.exists('feature_names.pkl')

if not model_exists:
    st.error("⚠️ Model files not found! Please run Project.py first to train the model.")
    st.stop()

# Load trained model and encoders
@st.cache_resource
def load_model():
    with open('loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, encoders, feature_names

model, label_encoders, feature_names = load_model()

# Sidebar information
with st.sidebar:
    st.markdown("### 📋 Application Guide")
    st.markdown("""
    This system predicts loan approval decisions based on:
    - **Financial Information**: Income, Loan Amount, Loan Term
    - **Personal Details**: Gender, Marital Status, Education
    - **Employment Status**: Employment type, Credit History
    - **Property Location**: Urban, Semi-urban, or Rural
    
    **How to use:**
    1. Fill in the applicant details
    2. Click 'Predict Loan Status'
    3. View the prediction result
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Model Used**: Logistic Regression  
    **Dataset**: Cleaned Loan Dataset  
    **Features**: 10 numerical and categorical features  
    **Status**: ✅ Active and Ready for Predictions
    """)

# Main content - Two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("💼 Applicant Information")
    
    # Financial Information
    st.markdown("#### 💰 Financial Information")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        applicant_income = st.number_input(
            "Applicant Income (₹)",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=1000,
            help="Monthly salary or annual income"
        )
    
    with col_f2:
        coapplicant_income = st.number_input(
            "Co-applicant Income (₹)",
            min_value=0,
            max_value=1000000,
            value=15000,
            step=1000,
            help="Co-applicant's monthly or annual income"
        )
    
    col_f3, col_f4 = st.columns(2)
    with col_f3:
        loan_amount = st.number_input(
            "Loan Amount Required (₹)",
            min_value=0,
            max_value=1000000,
            value=200000,
            step=10000,
            help="Total loan amount requested"
        )
    
    with col_f4:
        loan_term = st.number_input(
            "Loan Term (months)",
            min_value=12,
            max_value=600,
            value=360,
            step=12,
            help="Duration of loan repayment"
        )
    
    # Credit History
    st.markdown("#### 📊 Credit & Employment")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        credit_history = st.selectbox(
            "Credit History",
            options=[1, 0],
            format_func=lambda x: "Good" if x == 1 else "Poor",
            help="Good = 1, Poor = 0"
        )
    
    with col_c2:
        self_employed = st.selectbox(
            "Employment Type",
            options=["No", "Yes"],
            help="Are you self-employed?"
        )

with col2:
    st.subheader("👤 Personal Details")
    
    # Personal Information
    st.markdown("#### 👥 Demographics")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Applicant's gender"
        )
    
    with col_p2:
        married = st.selectbox(
            "Marital Status",
            options=["Yes", "No"],
            help="Are you married?"
        )
    
    # Education and Property
    st.markdown("#### 🎓 Education & Property")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        education = st.selectbox(
            "Education Level",
            options=["Graduate", "Not Graduate"],
            help="Highest level of education"
        )
    
    with col_e2:
        property_area = st.selectbox(
            "Property Area",
            options=["Urban", "Semiurban", "Rural"],
            help="Type of property area"
        )
    
    # Dependents (placeholder)
    st.markdown("#### 🔍 Summary")
    total_income = applicant_income + coapplicant_income
    st.metric(label="Total Income", value=f"₹{total_income:,.0f}")
    st.metric(label="Loan Amount", value=f"₹{loan_amount:,.0f}")
    st.metric(label="Loan-to-Income Ratio", value=f"{(loan_amount/max(total_income, 1)*100):.1f}%")

# Prediction
st.markdown("---")

if st.button("🎯 Predict Loan Status", key="predict_button", use_container_width=True):
    try:
        # Prepare data for prediction
        input_data = pd.DataFrame({
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Gender': [label_encoders['Gender'].transform([gender])[0]],
            'Married': [label_encoders['Married'].transform([married])[0]],
            'Education': [label_encoders['Education'].transform([education])[0]],
            'Self_Employed': [label_encoders['Self_Employed'].transform([self_employed])[0]],
            'Property_Area': [label_encoders['Property_Area'].transform([property_area])[0]]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Get prediction result
        result = label_encoders['Loan_Status'].inverse_transform([prediction])[0]
        confidence = max(prediction_proba) * 100
        
        # Display results
        st.markdown("---")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if result == "Approved":
                st.markdown(f"""
                <div class='success-box'>
                    <h3>✅ LOAN APPROVED</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='danger-box'>
                    <h3>❌ LOAN REJECTED</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_res2:
            st.markdown("### 📈 Prediction Details")
            st.write(f"**Prediction**: {result}")
            st.write(f"**Confidence Score**: {confidence:.2f}%")
            st.write(f"**Model Used**: Logistic Regression")
        
        # Application Summary
        st.markdown("---")
        st.markdown("### 📋 Application Summary")
        summary_data = {
            'Feature': [
                'Applicant Income',
                'Co-applicant Income',
                'Total Income',
                'Loan Amount',
                'Loan Term',
                'Credit History',
                'Gender',
                'Marital Status',
                'Education',
                'Employment',
                'Property Area'
            ],
            'Value': [
                f"₹{applicant_income:,}",
                f"₹{coapplicant_income:,}",
                f"₹{total_income:,}",
                f"₹{loan_amount:,}",
                f"{loan_term} months",
                "Good" if credit_history == 1 else "Poor",
                gender,
                married,
                education,
                "Self-Employed" if self_employed == "Yes" else "Salaried",
                property_area
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p><small>🔒 This system is for demonstration purposes. Always consult with financial advisors for actual loan decisions.</small></p>
    <p><small>© 2024 Loan Approval Prediction System | Powered by Machine Learning</small></p>
</div>
""", unsafe_allow_html=True)
