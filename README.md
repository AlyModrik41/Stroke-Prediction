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

## 📊 Data Visualization

Some insights from the dataset:

- **Correlation Heatmap** → shows relation between features (e.g., age, hypertension, heart disease).
-   <img width="1214" height="700" alt="download" src="https://github.com/user-attachments/assets/58d4e193-b28d-4e1f-abfc-f0451e7fcdc8" />
<img width="1214" height="700" alt="download" src="https://github.com/user-attachments/assets/9966fe57-0db9-45f1-93fe-4b6574e58384" />
- **Distribution Plots** → comparison between stroke and non-stroke patients.
- <img width="1227" height="603" alt="download" src="https://github.com/user-attachments/assets/5cc64496-d845-4062-bab1-20c7e1cd3eae" />
<img width="1104" height="360" alt="newplot" src="https://github.com/user-attachments/assets/2309b0f7-723b-445a-b035-349d882213e0" />
- **Categorical Analysis** → smoking status, work type, gender distribution.
- <img width="580" height="464" alt="download" src="https://github.com/user-attachments/assets/2c267457-85c4-474c-8414-711ad6c18a97" />
<img width="580" height="444" alt="download" src="https://github.com/user-attachments/assets/9c544de0-de49-4c70-9b16-e9e9b10c91b1" />

- <img width="580" height="499" alt="download" src="https://github.com/user-attachments/assets/174b62f8-8462-414c-907a-58be9ce43dbe" />
<img width="580" height="511" alt="download" src="https://github.com/user-attachments/assets/a2fdc701-702d-4225-835f-ac3292634fa8" />

## 📊 Random Forest Metrics: 
<img width="536" height="468" alt="download" src="https://github.com/user-attachments/assets/0732b447-6395-47aa-8785-919275f754c0" />
## 📊 Random Forest Feature Importance:
<img width="778" height="526" alt="download" src="https://github.com/user-attachments/assets/2f19c03e-d029-43d5-96e3-b00e748da137" />
## **Confusion Matrices** -> Showing the True Positives, True Negatives, False Positives and False Negatives.
<img width="1470" height="998" alt="download" src="https://github.com/user-attachments/assets/ffb500b1-ac22-4e22-ae34-bc5480b69040" />

---
## 📊 Results

| Model                | Precision (Stroke=1) | Recall (Stroke=1) | F1-score (Stroke=1) | Accuracy |
|-----------------------|----------------------|-------------------|----------------------|----------|
| Logistic Regression   | 0.16                 | 0.48              | 0.24                 | 0.87     |
| Decision Tree         | 0.12                 | 0.26              | 0.16                 | 0.89     |
| Random Forest         | 0.10                 | 0.71              | 0.17                 | 0.70     |

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
