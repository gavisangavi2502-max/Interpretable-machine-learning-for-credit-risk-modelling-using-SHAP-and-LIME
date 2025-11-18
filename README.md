# Interpretable Machine Learning for Credit Risk Assessment using SHAP and LIME

## Project Overview
This project develops and interprets a predictive model for **binary credit default classification** using a synthetic financial dataset.  
The focus is not only on achieving high predictive accuracy but also on **understanding the model’s decision-making process** using advanced interpretability techniques.

We employ **XGBoost**, a tree-based ensemble method, and apply **SHAP (SHapley Additive Explanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** to extract actionable insights for both global and local interpretability.

---

## Methodology
1. **Dataset Creation**  
   - Synthetic dataset generated with 20 features and 1000 samples.  
   - Target variable: `default` (1 = default, 0 = no default).

2. **Model Training**  
   - Algorithm: XGBoost Classifier  
   - Hyperparameters: `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`  
   - Train-test split: 80/20

3. **Evaluation Metrics**  
   - AUC (Area Under ROC Curve)  
   - F1 Score  
   - F-Beta Score (β=0.5)  
   - Confusion Matrix  
   - Credit Fairness Index (CFI)

---

##  Results

###  Model Performance
```json
{
  "Model": "XGBoost",
  "Parameters": {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1
  },
  "Performance": {
    "AUC": 0.89,
    "F1": 0.76,
    "F-Beta": 0.72,
    "CFI": 0.81
  }
}

Global Interpretability (SHAP)
Top 5 influential features: feature_3, feature_7, feature_12, feature_1, feature_15

SHAP summary plot saved as: shap_summary_plot.png

Local Interpretability (LIME vs SHAP)
Three loan applications analyzed:

Case	SHAP Insight	LIME Insight	Agreement
High Risk	feature_3 strongly increases risk	Same feature highlighted	
Low Risk	feature_12 reduces risk	feature_12 and feature_1 reduce risk	
Borderline	Mixed influence from feature_7	Emphasized feature_15 more	

Economic Summary (F-test)
File: economic_summary.csv

Statistically significant features: feature_3, feature_7, feature_12

Interpretation: These features align with SHAP insights, suggesting strong economic relevance in predicting defaults.

Visual Outputs
roc_curve.png → ROC curve showing model discrimination ability.

confusion_matrix.png → Confusion matrix showing classification results.

shap_summary_plot.png → Global SHAP feature importance.

lime_explanation_high_risk.png, lime_explanation_low_risk.png, lime_explanation_borderline.png → Local explanations.

Executive Summary
The XGBoost model achieved strong predictive performance (AUC = 0.89, F1 = 0.76). SHAP analysis revealed that feature_3 and feature_7 are globally most influential. Local explanations using SHAP and LIME agreed on high-risk and low-risk cases but diverged on borderline cases, highlighting the complementary nature of both methods. The F-test economic summary confirmed the statistical importance of the top features, strengthening confidence in the interpretability results. This project demonstrates how interpretable machine learning can support risk committees in making informed lending decisions.

Deliverables
credit_risk_data.csv → Synthetic dataset

model_training.ipynb → Full code implementation

metrics.json → Model parameters and performance metrics

shap_summary_plot.png → Global interpretability chart

economic_summary.csv → F-test based economic summary

roc_curve.png, confusion_matrix.png → Model evaluation visuals

lime_explanation_*.png → Local interpretability visuals

README.md → Documentation and executive summary

Conclusion
This project balances predictive performance with interpretability, ensuring that credit risk decisions are transparent, explainable, and actionable for non-technical stakeholders.


---
