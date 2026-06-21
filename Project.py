# Loan Approval Prediction System
# Complete Model Training Script

# Importing Required Libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("LOAN APPROVAL PREDICTION SYSTEM - MODEL TRAINING")
print("=" * 60)

# =============================================
# 1. LOAD DATASET
# =============================================
print("\n[1] Loading Dataset...")
data = pd.read_csv("cleaned_loan_dataset.csv")

print(f"Dataset shape: {data.shape}")
print(f"\nFirst 5 rows:")
print(data.head())
print(f"\nDataset Info:")
print(data.info())

# =============================================
# 2. DATA PREPROCESSING
# =============================================
print("\n[2] Data Preprocessing...")

# Check for missing values
print(f"Missing values:\n{data.isnull().sum()}")

# Handling Missing Values
data = data.dropna()

# Store label encoders for categorical variables
label_encoders = {}

categorical_columns = [
    'Gender',
    'Married',
    'Education',
    'Self_Employed',
    'Property_Area'
]

print("\nEncoding categorical variables...")
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
    print(f"  {column}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Encode target variable
le_target = LabelEncoder()
data['Loan_Status'] = le_target.fit_transform(data['Loan_Status'])
label_encoders['Loan_Status'] = le_target
print(f"  Loan_Status: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# =============================================
# 3. FEATURE SELECTION
# =============================================
print("\n[3] Feature Selection...")

X = data.drop(['Loan_Status'], axis=1)
y = data['Loan_Status']

print(f"Features: {list(X.columns)}")
print(f"Target variable shape: {y.shape}")

# =============================================
# 4. TRAIN-TEST SPLIT
# =============================================
print("\n[4] Train-Test Split (80-20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# =============================================
# 5. MODEL TRAINING
# =============================================
print("\n[5] Training Logistic Regression Model...")

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

print("✓ Model trained successfully!")

# =============================================
# 6. MODEL EVALUATION
# =============================================
print("\n[6] Model Evaluation...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy Score: {accuracy:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=le_target.classes_)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# =============================================
# 7. SAVE MODEL AND ENCODERS
# =============================================
print("\n[7] Saving Model and Encoders...")

# Save the trained model
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved as 'loan_model.pkl'")

# Save the label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ Label encoders saved as 'label_encoders.pkl'")

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("✓ Feature names saved as 'feature_names.pkl'")

# =============================================
# 8. SAMPLE PREDICTION
# =============================================
print("\n[8] Sample Prediction...")

# Create sample with proper encoding
try:
    sample_dict = {
        'ApplicantIncome': [50000],
        'CoapplicantIncome': [15000],
        'LoanAmount': [200],
        'Loan_Amount_Term': [360],
        'Credit_History': [1],
        'Gender': [label_encoders['Gender'].transform(['Male'])[0]],
        'Married': [label_encoders['Married'].transform(['Yes'])[0]],
        'Education': [label_encoders['Education'].transform(['Graduate'])[0]],
        'Self_Employed': [label_encoders['Self_Employed'].transform(['No'])[0]],
        'Property_Area': [label_encoders['Property_Area'].transform(['Urban'])[0]]
    }

    sample_data = pd.DataFrame(sample_dict)
    prediction = model.predict(sample_data)

    print(f"\nSample Input:")
    print(f"  Applicant Income: 50,000")
    print(f"  Co-applicant Income: 15,000")
    print(f"  Loan Amount: 200")
    print(f"  Loan Term: 360 months")
    print(f"  Credit History: Yes")
    print(f"  Gender: Male")
    print(f"  Married: Yes")
    print(f"  Education: Graduate")
    print(f"  Self Employed: No")
    print(f"  Property Area: Urban")

    result = le_target.inverse_transform(prediction)[0]
    print(f"\n🎯 Prediction: {result}")
except Exception as e:
    print(f"Sample prediction error: {e}")

print("\n" + "=" * 60)
print("✓ MODEL TRAINING COMPLETE!")
print("=" * 60)
