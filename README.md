# 🏦 Loan Default Prediction using Decision Tree and Random Forest

This project focuses on predicting whether a loan applicant will fully repay the loan or default, using machine learning models such as **Decision Tree** and **Random Forest**.

---

## 📌 Objective

The main goal is to:

- Use historical loan data to predict defaults.
- Build a classification model to distinguish between safe borrowers and risky ones.
- Evaluate and compare the performance of Decision Tree vs Random Forest classifiers.

---

## 📂 Dataset Overview

The dataset is based on loan applications and includes features like:

- **fico**: Credit score
- **int.rate**: Interest rate of the loan
- **installment**: Monthly payment
- **purpose**: Reason for loan
- **log.annual.inc**: Log of annual income
- **dti**: Debt-to-income ratio
- **credit.policy**: Whether the applicant met the credit underwriting criteria
- **not.fully.paid**: Target label (1 = loan defaulted, 0 = loan fully paid)

---

## 📊 Workflow Summary

1. **Data Loading & Cleaning**: Handled missing values and verified data consistency.
2. **Exploratory Data Analysis (EDA)**:
   - Compared FICO scores by credit policy.
   - Analyzed loan purpose vs repayment status.
3. **Feature Engineering**:
   - Converted categorical columns into numerical using one-hot encoding.
4. **Model Training**:
   - Split data into training and test sets (70:30).
   - Trained both Decision Tree and Random Forest classifiers.
5. **Model Evaluation**:
   - Used accuracy, precision, recall, F1-score, and confusion matrix to evaluate performance.
   - Compared models on how well they identify loan defaulters.

---

## 🧠 Why Decision Tree and Random Forest?

- **Decision Tree**:
  - Easy to interpret.
  - Captures simple decision paths.
  - Used here as a baseline model.

- **Random Forest**:
  - An ensemble of decision trees.
  - Improves accuracy and reduces overfitting.
  - Used here to improve generalization and boost predictive performance.

---

## ✅ Results Summary

- **Decision Tree** showed balanced performance but struggled to identify many defaults.
- **Random Forest** improved overall accuracy but had lower recall for defaulters (bias toward predicting "fully paid").

---

## 🔮 Future Improvements

- Handle class imbalance using SMOTE or class weights.
- Apply GridSearchCV for hyperparameter tuning.
- Experiment with boosting methods (e.g., XGBoost, LightGBM).
- Deploy the model via Flask or Streamlit.

---

## 💻 Tools Used

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

---


