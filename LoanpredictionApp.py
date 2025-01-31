import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Title of the web app
st.title("Loan Status Prediction with Random Forest")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Read the dataset
    loan_data = pd.read_csv(uploaded_file)
    st.write("Dataset Overview:")
    st.dataframe(loan_data.head())  # Show the first few rows of the dataset
    
    # Check for missing values
    st.write("Missing Values in the Dataset:")
    st.write(loan_data.isnull().sum())

    # Get summary statistics
    st.write("Summary Statistics:")
    st.write(loan_data.describe())

    # Visualize the distribution of Loan Status
    st.write("Distribution of Loan Status:")
    sns.countplot(x='Loan_Status', data=loan_data)
    st.pyplot(plt)

    # Handle missing values
    loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].median(), inplace=True)
    loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].median(), inplace=True)
    loan_data['Credit_History'].fillna(loan_data['Credit_History'].median(), inplace=True)

    # Convert categorical variables using LabelEncoder
    le = LabelEncoder()
    loan_data['Gender'] = le.fit_transform(loan_data['Gender'].astype(str))
    loan_data['Married'] = le.fit_transform(loan_data['Married'].astype(str))
    loan_data['Education'] = le.fit_transform(loan_data['Education'])
    loan_data['Self_Employed'] = le.fit_transform(loan_data['Self_Employed'].astype(str))
    loan_data['Property_Area'] = le.fit_transform(loan_data['Property_Area'])
    loan_data['Loan_Status'] = le.fit_transform(loan_data['Loan_Status'])

    # Replace '3+' with 3 in the 'Dependents' column
    loan_data['Dependents'] = loan_data['Dependents'].replace('3+', 3)
    loan_data['Dependents'].fillna(0, inplace=True)
    loan_data['Dependents'] = loan_data['Dependents'].astype(int)

    # Define features (X) and target variable (y)
    X = loan_data.drop(columns=['Loan_Status', 'Loan_ID'])  # Features
    y = loan_data['Loan_Status']  # Target

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict the test set results
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)

