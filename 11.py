# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create directory for charts if it doesn't exist
os.makedirs('charts/task1', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Loading Data, Data Pre-processing, EDA
print("Step 1: Loading Data, Data Pre-processing, EDA")
print("-" * 50)

# Load data
train_data = pd.read_csv('Training.csv')
test_data = pd.read_csv('Testing.csv')

# Fix the data structure issue
train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]

# Convert all columns except 'prognosis' to numeric
for col in train_data.columns:
    if col != 'prognosis':
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce').fillna(0).astype(int)

for col in test_data.columns:
    if col != 'prognosis':
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce').fillna(0).astype(int)

# Check the dimensions
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

# Check for missing values
print("\nMissing values in training data:", train_data.isnull().sum().sum())
print("Missing values in testing data:", test_data.isnull().sum().sum())

# Display basic information about the data
print("\nTraining data columns:", len(train_data.columns))
print("First few columns:", list(train_data.columns[:10]))
print("Last column (target):", train_data.columns[-1])

# Look at the first few rows
print("\nFirst 5 rows of training data (first 5 columns and target):")
print(train_data.iloc[:5, list(range(5)) + [-1]])

# Examine the distribution of target variable
disease_counts = train_data['prognosis'].value_counts()
print("\nNumber of unique diseases:", len(disease_counts))
print("Top 10 most common diseases:")
print(disease_counts.head(10))

# Visualize disease distribution
plt.figure(figsize=(12, 6))
plt.bar(disease_counts.index[:15], disease_counts.values[:15])
plt.xticks(rotation=90)
plt.title('Distribution of Top 15 Diseases in Training Data')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('charts/task1/disease_distribution.png')
plt.close()

# Explore symptom frequency
symptom_columns = train_data.columns[:-1]  # All columns except 'prognosis'
symptom_counts = train_data[symptom_columns].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
top_symptoms = symptom_counts.head(15)
sns.barplot(x=top_symptoms.index, y=top_symptoms.values)
plt.xticks(rotation=90)
plt.title('Top 15 Most Common Symptoms')
plt.xlabel('Symptom')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('charts/task1/symptom_frequency.png')
plt.close()

# Explore relationships between symptoms and diseases
# Create a heatmap of the top 10 symptoms and top 10 diseases
top_diseases = disease_counts.index[:10]
top_symptoms_names = symptom_counts.index[:10]

# Create a subset for the heatmap
heatmap_data = pd.DataFrame()
for disease in top_diseases:
    disease_records = train_data[train_data['prognosis'] == disease]
    disease_symptom_counts = disease_records[top_symptoms_names].sum()
    heatmap_data[disease] = disease_symptom_counts

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='g')
plt.title('Relationship between Top Symptoms and Top Diseases')
plt.xlabel('Disease')
plt.ylabel('Symptom')
plt.tight_layout()
plt.savefig('charts/task1/symptom_disease_heatmap.png')
plt.close()

# Step 2: Feature Engineering, Creating Train, and Test Datasets
print("\nStep 2: Feature Engineering, Creating Train, and Test Datasets")
print("-" * 50)


# Separate features and target
X_train = train_data.drop('prognosis', axis=1)
y_train = train_data['prognosis']
X_test = test_data.drop('prognosis', axis=1)
y_test = test_data['prognosis']

# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Get the mapping for later reference
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nNumber of disease classes:", len(label_mapping))
print("First 5 disease to label mappings:")
for i, (disease, label) in enumerate(list(label_mapping.items())[:5]):
    print(f"{disease}: {label}")

# Split training data into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

print("\nSplit data dimensions:")
print(f"X_train_split: {X_train_split.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

# Feature importance analysis
print("\nCalculating feature importance...")
rf_for_importance = RandomForestClassifier(n_estimators=100, random_state=42)
rf_for_importance.fit(X_train, y_train_encoded)

# Get feature importances
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_for_importance.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 20 important features
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
plt.title('Top 20 Most Important Symptoms')
plt.tight_layout()
plt.savefig('charts/task1/feature_importance.png')
plt.close()

print("\nTop 10 most important symptoms:")
print(feature_importances.head(10))

# Step 3: Apply at least 2 algorithms for classification
print("\nStep 3: Apply at least 2 algorithms for classification")
print("-" * 50)


# Function to evaluate and display metrics
def evaluate_model(name, model, X_train, X_val, y_train, y_val):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_val_pred = model.predict(X_val)

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average='weighted')
    recall = recall_score(y_val, y_val_pred, average='weighted')
    f1 = f1_score(y_val, y_val_pred, average='weighted')

    # Print metrics
    print(f"\nEvaluation metrics for {name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Create confusion matrix for top classes
    cm = confusion_matrix(y_val, y_val_pred)

    # For visualization, we'll use a simplified confusion matrix with top classes
    # to avoid overcrowding the plot
    top_classes_indices = np.argsort(np.bincount(y_val))[-10:]  # Top 10 most frequent classes
    cm_subset = cm[np.ix_(top_classes_indices, top_classes_indices)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label_encoder.classes_[i] for i in top_classes_indices],
                yticklabels=[label_encoder.classes_[i] for i in top_classes_indices])
    plt.title(f'Confusion Matrix for {name} (Top 10 Classes)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'charts/task1/{name.replace(" ", "_").lower()}_confusion_matrix.png')
    plt.close()

    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Algorithm 1: Random Forest
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_results = evaluate_model("Random Forest", rf_model, X_train_split, X_val, y_train_split, y_val)

# Algorithm 2: Support Vector Machine
print("\nTraining Support Vector Machine...")
svm_model = SVC(kernel='linear', C=1, random_state=42)
svm_results = evaluate_model("SVM", svm_model, X_train_split, X_val, y_train_split, y_val)

# Algorithm 3: K-Nearest Neighbors
print("\nTraining K-Nearest Neighbors...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_results = evaluate_model("KNN", knn_model, X_train_split, X_val, y_train_split, y_val)

# Step 4: Generate at least 2 Evaluation Metrics on each algorithm
print("\nStep 4: Generate at least 2 Evaluation Metrics on each algorithm")
print("-" * 50)

# Create a DataFrame with results
results = [rf_results, svm_results, knn_results]
model_names = ["Random Forest", "SVM", "KNN"]
metrics_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': [result['accuracy'] for result in results],
    'Precision': [result['precision'] for result in results],
    'Recall': [result['recall'] for result in results],
    'F1 Score': [result['f1'] for result in results]
})

print("\nModel comparison:")
print(metrics_df)

# Step 5: Comparing the results
print("\nStep 5: Comparing the results")
print("-" * 50)

# Visualize model comparison
plt.figure(figsize=(12, 8))
metrics_df_melted = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Score')
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df_melted)
plt.title('Model Comparison by Multiple Metrics')
plt.ylim(0, 1)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('charts/task1/model_comparison.png')
plt.close()

# Identify the best model based on accuracy
best_model_idx = metrics_df['Accuracy'].idxmax()
best_model_name = metrics_df.loc[best_model_idx, 'Model']
best_model = results[best_model_idx]['model']
print(f"\nThe best performing model is {best_model_name} with accuracy {metrics_df.loc[best_model_idx, 'Accuracy']:.4f}")

# Test the best model on the test set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
test_precision = precision_score(y_test_encoded, y_test_pred, average='weighted')
test_recall = recall_score(y_test_encoded, y_test_pred, average='weighted')
test_f1 = f1_score(y_test_encoded, y_test_pred, average='weighted')

print(f"\nTest metrics with {best_model_name}:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")

print("\nClassification Report on Test Data:")
print(classification_report(y_test_encoded, y_test_pred))

# Step 6: Fine Tune the best algorithm
print("\nStep 6: Fine Tune the best algorithm")
print("-" * 50)

# Define parameter grids for each model
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
}

# Get the parameter grid for the best model
best_param_grid = param_grids[best_model_name]

# Create the base model
if best_model_name == 'Random Forest':
    base_model = RandomForestClassifier(random_state=42)
elif best_model_name == 'SVM':
    base_model = SVC(random_state=42)
else:  # KNN
    base_model = KNeighborsClassifier()

print(f"Fine-tuning {best_model_name} with GridSearchCV...")
# Perform grid search
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=best_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train_encoded)

# Get best parameters and model
best_params = grid_search.best_params_
best_tuned_model = grid_search.best_estimator_

print(f"\nBest parameters for {best_model_name}:")
print(best_params)

# Evaluate tuned model on test set
y_test_pred_tuned = best_tuned_model.predict(X_test)
tuned_test_accuracy = accuracy_score(y_test_encoded, y_test_pred_tuned)
tuned_test_precision = precision_score(y_test_encoded, y_test_pred_tuned, average='weighted')
tuned_test_recall = recall_score(y_test_encoded, y_test_pred_tuned, average='weighted')
tuned_test_f1 = f1_score(y_test_encoded, y_test_pred_tuned, average='weighted')

print(f"\nTest metrics with tuned {best_model_name}:")
print(f"Accuracy: {tuned_test_accuracy:.4f}")
print(f"Precision: {tuned_test_precision:.4f}")
print(f"Recall: {tuned_test_recall:.4f}")
print(f"F1 Score: {tuned_test_f1:.4f}")

print("\nTuned model - Classification Report on Test Data:")
print(classification_report(y_test_encoded, y_test_pred_tuned))

# Compare original vs tuned model
print(f"\nImprovement after tuning:")
print(f"Accuracy: {(tuned_test_accuracy - test_accuracy) * 100:.2f}%")
print(f"Precision: {(tuned_test_precision - test_precision) * 100:.2f}%")
print(f"Recall: {(tuned_test_recall - test_recall) * 100:.2f}%")
print(f"F1 Score: {(tuned_test_f1 - test_f1) * 100:.2f}%")

# Create comparison chart for before and after tuning
tuning_comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Before Tuning': [test_accuracy, test_precision, test_recall, test_f1],
    'After Tuning': [tuned_test_accuracy, tuned_test_precision, tuned_test_recall, tuned_test_f1]
})

plt.figure(figsize=(10, 6))
tuning_comparison_melted = pd.melt(tuning_comparison, id_vars=['Metric'], var_name='Model', value_name='Score')
sns.barplot(x='Metric', y='Score', hue='Model', data=tuning_comparison_melted)
plt.title(f'Performance Comparison Before and After Tuning ({best_model_name})')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('charts/task1/tuning_comparison.png')
plt.close()

# Final model visualization (for Random Forest)
if best_model_name == 'Random Forest':
    # Plot feature importance from the tuned model
    feature_importances_tuned = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_tuned_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances_tuned.head(20))
    plt.title('Top 20 Most Important Symptoms (Tuned Random Forest)')
    plt.tight_layout()
    plt.savefig('charts/task1/tuned_feature_importance.png')
    plt.close()

    print("\nTop 10 most important symptoms (after tuning):")
    print(feature_importances_tuned.head(10))

# Save the best tuned model
joblib.dump(best_tuned_model, 'best_disease_prediction_model.pkl')
print("\nBest tuned model saved as 'best_disease_prediction_model.pkl'")

print("\nTask 1 (Classification) complete!")