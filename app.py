import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
st.set_page_config(page_title="Stroke Prediction App", layout="wide")
st.title("🧠 Stroke Prediction Dashboard")
st.markdown("This app uses multiple ML models (LogReg, SVM, KNN, RF, GB) trained with **SMOTE** to predict stroke risk.")
# ===============================
# Load Saved Models
# ===============================
@st.cache_resource
def load_models():
    models = {}
    models['Logistic Regression'] = pickle.load(open("logistic_regression_smote.pkl", "rb"))
    models['SVM'] = pickle.load(open("svm_smote.pkl", "rb"))
    models['KNN'] = pickle.load(open("knn_smote.pkl", "rb"))
    models['Random Forest'] = pickle.load(open("random_forest_smote.pkl", "rb"))
    models['Gradient Boosting'] = pickle.load(open("gradient_boosting_smote.pkl", "rb"))
    # Uncomment if you saved voting
    # models['Voting Classifier'] = pickle.load(open("vc_model.pkl", "rb"))
    return models

models = load_models()

# ===============================
# Page Config
# ===============================


# ===============================
# Sidebar Input
# ===============================
st.sidebar.header("Patient Information")
def user_input():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 0, 100, 50)
    hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 40, 300, 100)
    bmi = st.sidebar.slider("BMI", 10, 60, 25)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    
    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# ===============================
# Preprocessing
# ===============================
# Drop gender "Other"
if "gender" in input_df.columns and (input_df["gender"] == "Other").any():
    input_df = input_df[input_df["gender"] != "Other"]

# One-hot encoding (must match training preprocessing)
input_processed = pd.get_dummies(input_df)

# Align with training feature names
# Load feature names from training
feature_names = pickle.load(open("feature_names.pkl", "rb"))  # Save this during training
for col in feature_names:
    if col not in input_processed:
        input_processed[col] = 0
input_processed = input_processed[feature_names]  # reorder

st.subheader("📝 Patient Data")
st.write(input_df)

# ===============================
# Predictions
# ===============================
st.subheader("🔮 Model Predictions")
results = {}

for name, model in models.items():
    try:
        y_pred = model.predict(input_processed)[0]
        y_proba = model.predict_proba(input_processed)[0][1]
        results[name] = {"Prediction": y_pred, "Probability": y_proba}
    except Exception as e:
        results[name] = {"Error": str(e)}

results_df = pd.DataFrame(results).T
st.dataframe(results_df)

# ===============================
# ROC & Precision-Recall Curves (static example with training/test)
# ===============================
st.subheader("📊 Model Performance (Test Data)")
st.markdown("Below are sample plots from training/test evaluation.")

# Load test data results if saved
try:
    X_test = pickle.load(open("x_test.pkl", "rb"))
    y_test = pickle.load(open("y_test.pkl", "rb"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_title("ROC Curves")
    axes[0].legend()

    # Precision-Recall
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            axes[1].plot(rec, prec, label=name)
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend()

    st.pyplot(fig)
except:
    st.warning("Test data not available for ROC/PR plots. Save `x_test.pkl` and `y_test.pkl` to show them.")

