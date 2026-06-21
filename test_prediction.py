#!/usr/bin/env python3
"""
Test script to verify the loan prediction system works correctly
"""

import pickle
import pandas as pd
import os

print("=" * 70)
print("LOAN PREDICTION SYSTEM - TESTING")
print("=" * 70)

# Check if all required files exist
print("\n[1] Checking required files...")
required_files = ['loan_model.pkl', 'label_encoders.pkl', 'feature_names.pkl', 'cleaned_loan_dataset.csv']
all_files_exist = True

for file in required_files:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")
    if not exists:
        all_files_exist = False

if not all_files_exist:
    print("\n✗ ERROR: Some required files are missing!")
    exit(1)

print("\n✓ All required files found!")

# Load model and encoders
print("\n[2] Loading model and encoders...")
try:
    with open('loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("  ✓ Model loaded")
    
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    print("  ✓ Label encoders loaded")
    
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("  ✓ Feature names loaded")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    exit(1)

# Test predictions with multiple scenarios
print("\n[3] Testing predictions...")

test_cases = [
    {
        'name': 'High Income, Good Credit',
        'applicant_income': 80000,
        'coapplicant_income': 20000,
        'loan_amount': 300000,
        'loan_term': 360,
        'credit_history': 1,
        'gender': 'Male',
        'married': 'Yes',
        'education': 'Graduate',
        'self_employed': 'No',
        'property_area': 'Urban'
    },
    {
        'name': 'Low Income, Poor Credit',
        'applicant_income': 30000,
        'coapplicant_income': 0,
        'loan_amount': 150000,
        'loan_term': 180,
        'credit_history': 0,
        'gender': 'Female',
        'married': 'No',
        'education': 'Not Graduate',
        'self_employed': 'Yes',
        'property_area': 'Rural'
    },
    {
        'name': 'Medium Income, Good Credit',
        'applicant_income': 50000,
        'coapplicant_income': 15000,
        'loan_amount': 200000,
        'loan_term': 360,
        'credit_history': 1,
        'gender': 'Male',
        'married': 'Yes',
        'education': 'Graduate',
        'self_employed': 'No',
        'property_area': 'Semiurban'
    }
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n  Test Case {i}: {test_case['name']}")
    
    try:
        # Prepare input
        input_data = pd.DataFrame({
            'ApplicantIncome': [test_case['applicant_income']],
            'CoapplicantIncome': [test_case['coapplicant_income']],
            'LoanAmount': [test_case['loan_amount']],
            'Loan_Amount_Term': [test_case['loan_term']],
            'Credit_History': [test_case['credit_history']],
            'Gender': [encoders['Gender'].transform([test_case['gender']])[0]],
            'Married': [encoders['Married'].transform([test_case['married']])[0]],
            'Education': [encoders['Education'].transform([test_case['education']])[0]],
            'Self_Employed': [encoders['Self_Employed'].transform([test_case['self_employed']])[0]],
            'Property_Area': [encoders['Property_Area'].transform([test_case['property_area']])[0]]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        result = encoders['Loan_Status'].inverse_transform([prediction])[0]
        confidence = max(proba) * 100
        
        print(f"    Result: {result}")
        print(f"    Confidence: {confidence:.2f}%")
        print(f"    ✓ Prediction successful")
        
    except Exception as e:
        print(f"    ✗ Error: {e}")

print("\n" + "=" * 70)
print("✓ TESTING COMPLETE!")
print("=" * 70)
print("\nNext Steps:")
print("1. Run the Streamlit app: streamlit run App.py")
print("2. Or test with: python test_prediction.py")
print("=" * 70)
