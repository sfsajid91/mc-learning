# Machine Learning in Healthcare: A Concise Analysis

## Introduction

This report presents two machine learning applications in healthcare: disease prediction from symptoms (classification) and medical insurance cost prediction (regression).

## Dataset Overview

- **Disease Prediction Dataset**: 132 binary symptom features, 41 disease categories, 4,920 training samples
- **Insurance Dataset**: Features include age, sex, BMI, children, smoking status, and region for 1,338 beneficiaries

## Classification Task: Disease Prediction

**Data Analysis**:
The dataset showed varied distribution of diseases with unique symptom patterns.

![Disease Distribution](charts/task1/disease_distribution.png)
_Distribution of top 15 diseases_

**Symptom-Disease Relationships**:
Heatmap analysis revealed clear patterns of symptom co-occurrence for specific diseases.

![Symptom-Disease Relationship](charts/task1/symptom_disease_heatmap.png)
_Relationships between symptoms and diseases_

**Model Comparison**:
Three algorithms were evaluated: Random Forest, SVM, and KNN. Random Forest performed best and was selected for fine-tuning.

![Model Comparison](charts/task1/model_comparison.png)
_Performance comparison across metrics_

**Feature Importance**:
The tuned Random Forest identified key symptoms crucial for accurate disease prediction.

![Tuned Feature Importance](charts/task1/tuned_feature_importance.png)
_Top 20 most important symptoms_

## Regression Task: Insurance Cost Prediction

**Data Analysis**:
Initial exploration revealed right-skewed distribution of insurance charges and strong correlations between features.

![Charges Distribution](charts/task2/charges_distribution.png)
_Distribution of insurance charges_

**Feature Relationships**:
Analysis showed strong correlation between smoking status and charges, with positive correlations for age and BMI.

![Numerical Features vs Charges](charts/task2/numerical_features_vs_charges.png)
_Relationships between numerical features and charges_

**Model Comparison**:
Three regression algorithms were evaluated: Linear Regression, Random Forest, and Gradient Boosting, with Gradient Boosting performing best.

![Model Comparison by R²](charts/task2/model_comparison_R².png)
_Model comparison by R² score_

**Feature Importance**:
The tuned Gradient Boosting model revealed smoking status as the dominant factor affecting insurance costs.

![Feature Importance](charts/task2/tuned_gradient_boosting_feature_importance.png)
_Feature importance in the tuned model_

## Key Insights

**Classification Task**:

- Random Forest achieved exceptional accuracy, demonstrating that symptom patterns strongly predict diseases
- Feature importance analysis provides valuable diagnostic indicators for healthcare professionals

**Regression Task**:

- Gradient Boosting achieved R² of 0.882 after tuning
- Smoking dramatically impacts insurance costs, followed by BMI and age
- Results provide actionable insights for insurance pricing and risk assessment

## Conclusion

Tree-based models (Random Forest and Gradient Boosting) excelled in both healthcare applications due to their ability to capture complex non-linear relationships. The classification task demonstrated how symptom patterns can accurately predict diseases, while the regression task quantified lifestyle factors' impact on healthcare costs. These insights benefit healthcare providers, insurance companies, and patients by enabling more informed decision-making in diagnosis and cost prediction.
