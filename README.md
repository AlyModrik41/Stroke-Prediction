# 🧠 Stroke Prediction using Machine Learning

Predicting the likelihood of a stroke based on patient health data.  
This project applies machine learning models to a healthcare dataset, addressing **class imbalance**, performing **feature engineering**, and deploying a **Streamlit web app** for interactive predictions.

---

## 📌 Project Overview
- Conducted **Exploratory Data Analysis (EDA)** with heatmaps & visualizations.  
- Preprocessed data with:
  - Encoding categorical variables  
  - Scaling numerical features (for linear models)  
  - Handling missing values in `smoking_status`  
- Dealt with **class imbalance** using **SMOTE (Synthetic Minority Oversampling Technique)**.  
- Trained multiple ML models:
  - Logistic Regression (with L1/L2 penalties)  
  - Decision Tree  
  - Random Forest  
  - Gradient Boosting  
  - Support Vector Machine (SVM)  
- Applied **GridSearchCV** for hyperparameter tuning.  
- Built a **Streamlit app** for live stroke risk prediction.

---

## ⚙️ Tech Stack
- **Languages:** Python (Pandas, NumPy, Matplotlib, Seaborn)  
- **ML Frameworks:** scikit-learn, imbalanced-learn (SMOTE)  
- **App:** Streamlit  
- **Other:** Joblib (model saving), GridSearchCV (hyperparameter tuning)  

---

## 📊 Results

| Model                | Precision (Stroke=1) | Recall (Stroke=1) | F1-score (Stroke=1) | Accuracy |
|-----------------------|----------------------|-------------------|----------------------|----------|
| Logistic Regression   | 0.16                 | 0.48              | 0.24                 | 0.87     |
| Decision Tree         | 0.12                 | 0.26              | 0.16                 | 0.89     |
| Random Forest         | 0.09                 | 0.10              | 0.09                 | 0.92     |

> ⚠️ The dataset is **highly imbalanced** (stroke cases are rare).  
> Future work may include advanced techniques (XGBoost, cost-sensitive learning, or deep learning).

---

## 🚀 Streamlit App

The app allows users to input health parameters and receive stroke predictions.

### Run Locally:
```bash
git clone https://github.com/yourusername/stroke-prediction.git
cd stroke-prediction
pip install -r requirements.txt
streamlit run app/app.py
