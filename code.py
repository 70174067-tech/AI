import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ==========================================
# TASK 1: DATA LOADING & UNDERSTANDING
# ==========================================
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("--- Dataset Overview ---")
print(f"Shape: {df.shape}")
print("\nData Types and Missing Values:")
print(df.info())

print("\nTarget Variable Distribution ('Churn'):")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True) * 100)

# ==========================================
# TASK 2: DATA PREPROCESSING
# ==========================================
# Convert TotalCharges from object to numeric (handling empty string spaces ' ')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')

# Drop missing values (11 rows where tenure=0)
df_clean = df.dropna(subset=['TotalCharges']).copy()

# Drop identifier column
df_clean = df_clean.drop(columns=['customerID'])

print(f"\nCleaned Dataset Shape: {df_clean.shape}")
print(f"Duplicates remaining: {df_clean.duplicated().sum()}")

# ==========================================
# TASK 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(data=df_clean, x='Churn', palette='Set2', ax=axes[0, 0])
axes[0, 0].set_title('Overall Churn Distribution')

sns.countplot(data=df_clean, x='Contract', hue='Churn', palette='Set1', ax=axes[0, 1])
axes[0, 1].set_title('Churn by Contract Type')

sns.countplot(data=df_clean, x='InternetService', hue='Churn', palette='Set1', ax=axes[1, 0])
axes[1, 0].set_title('Churn by Internet Service Type')

sns.kdeplot(data=df_clean, x='tenure', hue='Churn', fill=True, common_norm=False, ax=axes[1, 1])
axes[1, 1].set_title('Tenure Distribution by Churn Status')

plt.tight_layout()
plt.savefig('churn_eda_summary.png')
plt.show()

# ==========================================
# TASK 4: FEATURE ENGINEERING & ENCODING
# ==========================================
y = df_clean['Churn'].map({'Yes': 1, 'No': 0})
X = df_clean.drop(columns=['Churn'])

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [c for c in X.columns if c not in num_cols]

# One-hot encoding categorical variables
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# ==========================================
# TASK 5: TRAINING & TESTING SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# Train Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

# ==========================================
# TASK 6: MODEL EVALUATION
# ==========================================
results = []
for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    })

eval_df = pd.DataFrame(results)
print("\n=== Model Performance Comparison ===")
print(eval_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']])

for res in results:
    print(f"\n--- {res['Model']} Confusion Matrix ---")
    print(res['Confusion Matrix'])

# ==========================================
# TASK 7: SAMPLE PREDICTION FUNCTION
# ==========================================
def predict_customer_churn(customer_dict, model, scaler, feature_columns):
    df_cust = pd.DataFrame([customer_dict])
    df_cust['TotalCharges'] = pd.to_numeric(df_cust['TotalCharges'], errors='coerce').fillna(0)
    
    cat_c = [c for c in df_cust.columns if c not in num_cols]
    df_cust_enc = pd.get_dummies(df_cust, columns=cat_c, drop_first=True)
    
    for col in feature_columns:
        if col not in df_cust_enc.columns:
            df_cust_enc[col] = 0
    df_cust_enc = df_cust_enc[feature_columns]
    
    df_cust_enc[num_cols] = scaler.transform(df_cust_enc[num_cols])
    
    pred_class = model.predict(df_cust_enc)[0]
    pred_prob = model.predict_proba(df_cust_enc)[0][1]
    
    return ("Churn (Yes)" if pred_class == 1 else "No Churn (No)"), pred_prob

# Sample High-Risk Customer
sample_customer_high_risk = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
    'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
    'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 75.50, 'TotalCharges': 151.00
}

pred_status, pred_prob = predict_customer_churn(
    sample_customer_high_risk, trained_models['Logistic Regression'], scaler, X_train.columns.tolist()
)

print(f"\nPrediction for Sample Customer: {pred_status} with Probability: {pred_prob:.2%}")