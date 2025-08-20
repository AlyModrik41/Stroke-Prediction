import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =============================
# Load models
# =============================
model_files = {
    "Logistic Regression": "saved_models/logistic_regression_smote.pkl",
    "SVM": "saved_models/svm_smote.pkl",
    "KNN": "saved_models/knn_smote.pkl",
    "Random Forest": "saved_models/random_forest_smote.pkl",
    "Gradient Boosting": "saved_models/gradient_boosting_smote.pkl"
}
models = {name: joblib.load(path) for name, path in model_files.items()}

# =============================
# Load datasets (for full evaluation)
# =============================
x_linear = pd.read_csv('linear_models_df.csv')
x_tree = pd.read_csv('tree_models_df.csv')

# Remove any unwanted index columns
x_linear = x_linear.loc[:, ~x_linear.columns.str.contains('^Unnamed')]
x_tree = x_tree.loc[:, ~x_tree.columns.str.contains('^Unnamed')]

# =============================
# Streamlit UI
# =============================
st.title("Stroke Prediction App")
st.write("Choose a model, predict stroke probability for a single patient, or evaluate the whole dataset.")

# =============================
# Model selection
# =============================
model_choice = st.selectbox(
    "Select a model (Logistic Regression recommended)",
    options=list(models.keys()),
    index=0
)
model = models[model_choice]

# =============================
# Feature Input Section
# =============================
st.subheader("Enter Patient Features")

# Numeric features
age_hyper = st.number_input("Age", 0, 120, 30)
avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0,1])
heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0,1])

# Categorical features
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

# =============================
# Convert inputs to dataframe
# =============================
input_dict = {
    "age_hyper": [age_hyper],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "gender": [gender],
    "work_type": [work_type],
    "smoking_status": [smoking_status],
    "marital_status": [marital_status],
    "residence_type": [residence_type]
}
df_input = pd.DataFrame(input_dict)

# =============================
# One-hot encode categorical variables
# =============================
categorical_cols = ["gender", "work_type", "smoking_status", "marital_status", "residence_type"]
df_input = pd.get_dummies(df_input, columns=categorical_cols)

# =============================
# Match model training columns
# =============================
if model_choice in ["Logistic Regression", "SVM", "KNN"]:
    training_cols = x_linear.columns
else:
    training_cols = x_tree.columns

# Add missing columns
for col in training_cols:
    if col not in df_input.columns:
        df_input[col] = 0

# Drop extra columns not in training
df_input = df_input[training_cols]

# Convert to numpy array
input_data = df_input.values

# =============================
# Single patient prediction
# =============================
if st.button("Predict Single Patient Stroke Probability"):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[:, 1][0]
    else:
        try:
            prob = model.decision_function(input_data)
            prob = (prob - prob.min()) / (prob.max() - prob.min())
            prob = prob[0]
        except:
            prob = model.predict(input_data)[0]
    st.success(f"🩺 Predicted probability of having a stroke: {prob*100:.2f}%")

# =============================
# Full dataset evaluation
# =============================
st.markdown("---")
st.subheader("Evaluate Whole Dataset")

if st.button("Evaluate Whole Dataset"):
    # Pick dataset
    if model_choice in ["Logistic Regression", "SVM", "KNN"]:
        X = x_linear.copy()
    else:
        X = x_tree.copy()

    # Remove unwanted columns
    X = X.loc[:, training_cols]

    # Compute probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        try:
            probs = model.decision_function(X)
            probs = (probs - probs.min()) / (probs.max() - probs.min())
        except:
            probs = model.predict(X)

    stroke_count = np.sum(probs >= 0.5)  # threshold = 0.5
    total_count = len(X)
    st.success(f"Predicted Stroke Cases: {stroke_count}/{total_count}")
    st.info(f"Percentage of patients predicted to have a stroke: {stroke_count/total_count*100:.2f}%")
