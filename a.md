# Machine Learning in Healthcare: Classification and Regression Analysis

## Introduction
This analysis applies machine learning to healthcare datasets: disease prediction from symptoms (classification) and medical insurance cost prediction (regression). Both tasks demonstrate the effectiveness of various algorithms, particularly tree-based models, in solving healthcare problems.

## Dataset Overview
**Disease Prediction**: 132 binary symptom features across 41 disease categories with 4,920 training samples.
**Insurance Costs**: 1,338 records with features like age, sex, BMI, children, smoking status, and region, predicting medical charges.

## Classification Task: Disease Prediction

The analysis began with examining symptom distribution and their relationships to diseases:

![Symptom-Disease Relationship](charts/task1/symptom_disease_heatmap.png)
_Figure 1: Relationships between top symptoms and diseases_

Three classification models were evaluated—Random Forest, SVM, and KNN:

![Model Comparison](charts/task1/model_comparison.png)
_Figure 2: Classification model performance comparison_

Random Forest emerged as the top performer, achieving near-perfect accuracy after hyperparameter optimization. Feature importance analysis identified the most predictive symptoms:

![Tuned Feature Importance](charts/task1/tuned_feature_importance.png)
_Figure 3: Top 20 most important symptoms for disease prediction_

## Regression Task: Insurance Cost Prediction

Initial data exploration revealed the distribution of charges and relationships with features:

![Numerical Features vs Charges](charts/task2/numerical_features_vs_charges.png)
_Figure 4: Relationships between numerical features and insurance charges_

![Categorical Features vs Charges](charts/task2/categorical_features_vs_charges.png)
_Figure 5: Impact of categorical features on insurance charges_

Three regression models were implemented—Linear Regression (R²: 0.784), Random Forest (R²: 0.866), and Gradient Boosting (R²: 0.879):

![Model Comparison by R²](charts/task2/model_comparison_R².png)
_Figure 6: Regression model comparison by R² score_

Gradient Boosting performed best and was further tuned to achieve R² of 0.882:

![Tuned Gradient Boosting](charts/task2/tuned_gradient_boosting_actual_vs_predicted.png)
_Figure 7: Actual vs Predicted charges for tuned Gradient Boosting_

Feature importance analysis revealed smoking status as the dominant factor in insurance costs:

![Feature Importance](charts/task2/tuned_gradient_boosting_feature_importance.png)
_Figure 8: Feature importance in insurance cost prediction_

## Key Insights

**Classification Task**: 
- Symptom patterns highly predictive of specific diseases
- Random Forest achieved excellent accuracy, demonstrating potential for diagnostic support
- Feature importance analysis identified crucial symptoms for different conditions

**Regression Task**:
- Lifestyle factors, especially smoking, dramatically impact insurance costs
- Age and BMI show strong positive correlations with charges
- Regional variations exist but have less impact than personal factors

## Conclusion

Tree-based models (Random Forest and Gradient Boosting) demonstrated superior performance in both healthcare tasks. For disease prediction, accurate symptom-based classification could support early diagnosis and treatment planning. For insurance costs, quantifying the impact of lifestyle factors provides actionable insights for risk assessment and personal health decisions.

These applications show machine learning's potential to transform healthcare by enabling more accurate diagnoses and fairer insurance pricing based on quantifiable risk factors. The models' success suggests healthcare data's complex non-linear relationships are best captured by algorithms that can handle mixed data types and complex interactions.