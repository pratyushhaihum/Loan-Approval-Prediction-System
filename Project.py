# Loan Approval Prediction System

# Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
data = pd.read_csv("loan_data.csv")

# Display First 5 Rows
print(data.head())

# -----------------------------
# Data Preprocessing
# -----------------------------

# Handling Missing Values
data.fillna(method='ffill', inplace=True)

# Encoding Categorical Variables
label_encoder = LabelEncoder()

categorical_columns = [
    'Gender',
    'Married',
    'Education',
    'Self_Employed',
    'Property_Area',
    'Loan_Status'
]

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# -----------------------------
# Feature Selection
# -----------------------------

X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = data['Loan_Status']

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model Training
# -----------------------------

model = LogisticRegression()

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Model Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Sample Prediction
# -----------------------------

sample_data = np.array([
    1,      # Gender
    1,      # Married
    0,      # Dependents
    1,      # Education
    0,      # Self Employed
    5000,   # ApplicantIncome
    2000,   # CoapplicantIncome
    150,    # LoanAmount
    360,    # Loan_Amount_Term
    1,      # Credit_History
    2       # Property_Area
]).reshape(1, -1)

prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("\nLoan Approved")
else:
    print("\nLoan Rejected")
