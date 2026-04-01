# Credit Card Default Prediction (ML)

## Collaborators
- Rachit Thacker  
- Shivani Aggarwal

---

## Project Overview
This project focuses on predicting whether a credit card client will default on their payment in the following month. The dataset contains financial and demographic information for 30,000 credit card clients in Taiwan between April and September 2005.

The goal is to build a machine learning model that can accurately identify high-risk clients while minimizing false alarms.

---

## Dataset Description
The dataset includes:
- **Demographic features**: Age, gender, education, marital status  
- **Financial features**: Credit limit (`LIMIT_BAL`)  
- **Repayment history**: `PAY_0` to `PAY_6`  
- **Bill amounts**: `BILL_AMT1` to `BILL_AMT6`  
- **Payment amounts**: `PAY_AMT1` to `PAY_AMT6`  

Target variable:
- `default.payment.next.month` (1 = default, 0 = no default)

---

## Initial Insights
- Repayment history (e.g., `PAY_0`, `PAY_6`) is the strongest predictor of default risk.
- The dataset is **imbalanced** (~78% non-default, ~22% default).
- Financial behavior (payments vs bills) is more predictive than demographic variables.
- Potential **multicollinearity** exists among monthly bill and payment features.

---

## Workflow

### 1. Data Preparation
- No missing values detected
- Cleaned categorical inconsistencies (e.g., EDUCATION values)
- Train-test split (70/30)

### 2. Exploratory Data Analysis
- Identified class imbalance
- Found:
  - Lower credit limits → higher default risk
  - Lower payment ratios → higher default probability
  - Slight correlation with younger age

### 3. Feature Engineering
- Payment-to-bill ratios (`PAY_RATIO_i`)
- Average repayment delay (`AVG_DELAY`)
- Bill trend over time (`BILL_TREND`)

### 4. Modeling
Models tested:
- Dummy Classifier (baseline)
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### 5. Evaluation Metric
- **Precision** (to minimize false positives and avoid penalizing reliable customers)

---

## Results Summary

| Model | Validation Precision | Test Precision | Key Insight |
|------|--------------------|--------------|------------|
| Logistic Regression | 0.7265 | 0.6916 | Best overall performance |
| Decision Tree (tuned) | 0.6963 | 0.704498 | Strong after pruning |
| Gradient Boosting | 0.6761 | 0.705467 | Good but slower |
| Random Forest | 0.6436 | 0.999039 | High variance |
---

## Key Findings
- **PAY_0 (recent repayment status)** is the most important feature
- Simpler models outperform complex ones
- Strong regularization improves generalization
- Behavioral features dominate over demographics

---

## Final Model
- **Model**: Logistic Regression  
- **Hyperparameter**: C = 0.001  
- **Test Precision**: **0.6916**

---

## Model Interpretation
Using SHAP values:
- Recent payment behavior strongly influences predictions
- Lower credit limits increase risk
- Consistent repayment reduces predicted default probability

---

## Future Improvements
- Try XGBoost / LightGBM
- Add external economic data
- Use advanced encoding (target encoding, embeddings)
- Address class imbalance with resampling techniques

---

## Key Takeaways
- Simpler models can outperform complex ones
- Feature quality matters more than model complexity
- Choosing the right evaluation metric is critical
- Bias-variance tradeoff is central to model performance

