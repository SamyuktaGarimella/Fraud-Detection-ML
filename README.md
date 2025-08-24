# Fraud Detection with Machine Learning

This repository contains an end-to-end machine learning project on **credit card fraud detection** using anonymised transaction data. 

## Project Overview
- **Goal**: Predict fraudulent transactions while minimising false positives and false negatives.  
- **Dataset**: An anonymized credit card transactions dataset with fraud labels (1 = Fraud, 0 = Legitimate).  
- **Approach**:  
  1. Exploratory Data Analysis (EDA)  
  2. Missing value handling & feature selection  
  3. Visualisation of fraud vs non-fraud trends  
  4. Model building with Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost  
  5. Stacking Ensemble for improved performance  
  6. Explainability with **SHAP values**  
- **Evaluation Metrics**: ROC-AUC, Precision-Recall AUC, F1-score, Precision, Recall. (Since the data is highly imbalanced, PR AUC is emphasized.)

---

## Key Insights
- Fraud cases account for **<1%** of all transactions → confirms **severe imbalance**.  
- Transaction amounts are **skewed** → log-transform helps normalization.  
- Several feature groups with very high missing values (>80%) are dropped.  
- Fraud rate varies by product type, confirming **product-related fraud risk patterns**.  

---

## Models & Results
All models were trained/tested using a stratified split.  

| Model                 | ROC-AUC    | PR-AUC     | Notes                             |
| --------------------- | ---------- | ---------- | --------------------------------- |
| Logistic Regression   | 0.8620     | 0.3731     | Strong linear baseline            |
| Random Forest         | 0.8837     | 0.5011     | Handles non-linearity, slower     |
| CatBoost              | 0.9319     | 0.6667     | Handles categorical features well |
| XGBoost               | 0.9337     | 0.6913     | Great on tabular data             |
| LightGBM              | 0.9301     | 0.6936     | Efficient, comparable to XGBoost  |
| **Stacking Ensemble** | **0.9373** | **0.7069** | Best overall                      |

**Best Model** → Stacking Ensemble (ROC-AUC: 0.93, PR-AUC: 0.70)

---

## Visualizations
- Fraud vs Non-Fraud class imbalance  
- Distribution of transaction amounts (with log-transform)  
- Correlation heatmap of numeric features  
- Fraud rates across products  
- Confusion matrix & Precision-Recall curve of best model  
- SHAP feature importance plot  

---

## Threshold Tuning  

All models were evaluated using the **default decision threshold of 0.5**.  

For the **top two models (Stacking Ensemble and LightGBM)**, threshold tuning was also demonstrated.  
This showed how **precision and recall values change** when the threshold is adjusted, allowing the cutoff to be tailored to balance false positives and false negatives depending on the task.  


---

