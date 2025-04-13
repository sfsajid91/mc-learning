# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Create directory for charts if it doesn't exist
os.makedirs('charts/task2', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Loading Data, Data Pre-processing, EDA
print("Step 1: Loading Data, Data Pre-processing, EDA")
print("-" * 50)

# Load the dataset
data = pd.read_csv('insurance.csv')

# Check the dimensions
print("Dataset shape:", data.shape)

# Display basic information
print("\nDataset info:")
print(data.info())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Display basic statistics
print("\nBasic statistics:")
print(data.describe())

# Look at the first few rows
print("\nFirst 5 rows:")
print(data.head())

# Explore the distribution of the target variable (charges)
plt.figure(figsize=(10, 6))
sns.histplot(data['charges'], kde=True)
plt.title('Distribution of Insurance Charges')
plt.xlabel('Charges ($)')
plt.ylabel('Frequency')
plt.savefig('charts/task2/charges_distribution.png')
plt.close()

# Explore relationships between numerical features and target
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.scatterplot(x='age', y='charges', data=data, hue='smoker')
plt.title('Age vs Charges')

plt.subplot(2, 2, 2)
sns.scatterplot(x='bmi', y='charges', data=data, hue='smoker')
plt.title('BMI vs Charges')

plt.subplot(2, 2, 3)
sns.boxplot(x='children', y='charges', data=data)
plt.title('Children vs Charges')

plt.subplot(2, 2, 4)
sns.boxplot(x='sex', y='charges', data=data)
plt.title('Sex vs Charges')
plt.tight_layout()
plt.savefig('charts/task2/numerical_features_vs_charges.png')
plt.close()

# Explore categorical features
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.boxplot(x='smoker', y='charges', data=data)
plt.title('Smoker vs Charges')

plt.subplot(1, 3, 2)
sns.boxplot(x='region', y='charges', data=data)
plt.title('Region vs Charges')

plt.subplot(1, 3, 3)
sns.boxplot(x='sex', y='charges', data=data)
plt.title('Sex vs Charges')
plt.tight_layout()
plt.savefig('charts/task2/categorical_features_vs_charges.png')
plt.close()

# Correlation matrix for numerical features
numerical_data = data.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10, 8))
correlation_matrix = numerical_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('charts/task2/correlation_matrix.png')
plt.close()

# Step 2: Feature Engineering, Creating Train and Test Datasets
print("\nStep 2: Feature Engineering, Creating Train and Test Datasets")
print("-" * 50)

# Separate features and target
X = data.drop('charges', axis=1)
y = data['charges']

# Identify categorical and numerical columns
categorical_cols = ['sex', 'smoker', 'region']
numerical_cols = ['age', 'bmi', 'children']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Step 3: Apply at least 2 algorithms for regression
print("\nStep 3: Apply at least 2 algorithms for regression")
print("-" * 50)

# Create pipelines for different models
models = {
    'Linear Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]),

    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),

    'Gradient Boosting': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
}

# Function to evaluate models
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Print metrics
    print(f"\nEvaluation metrics for {name}:")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Testing MAE: {test_mae:.2f}")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.title(f'Actual vs Predicted Charges ({name})')
    plt.savefig(f'charts/task2/{name.replace(" ", "_").lower()}_actual_vs_predicted.png')
    plt.close()

    # Return metrics for comparison
    return {
        'Model': name,
        'MAE': test_mae,
        'MSE': test_mse,
        'RMSE': test_rmse,
        'R²': test_r2
    }

# Step 4: Generate at least 2 Evaluation Metrics on each algorithm
print("\nStep 4: Generate at least 2 Evaluation Metrics on each algorithm")
print("-" * 50)

# Evaluate all models
results = []
for name, model in models.items():
    result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    results.append(result)

# Step 5: Comparing the results
print("\nStep 5: Comparing the results")
print("-" * 50)

# Create a DataFrame with results
results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df)

# Visualize model comparison
metrics = ['MAE', 'MSE', 'RMSE', 'R²']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric, data=results_df)
    plt.title(f'Model Comparison by {metric}')
    plt.ylabel(metric)
    plt.savefig(f'charts/task2/model_comparison_{metric}.png')
    plt.close()

# Find the best model based on R²
best_model_idx = results_df['R²'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
print(f"\nThe best performing model is {best_model_name} with R² = {results_df.loc[best_model_idx, 'R²']:.4f}")

# Step 6: Fine Tune the best algorithm
print("\nStep 6: Fine Tune the best algorithm")
print("-" * 50)

# Define parameter grids for each model
param_grids = {
    'Linear Regression': {},  # Linear Regression doesn't have hyperparameters to tune

    'Random Forest': {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    },

    'Gradient Boosting': {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7],
        'regressor__subsample': [0.8, 0.9, 1.0]
    }
}

# Get the parameter grid for the best model
best_param_grid = param_grids.get(best_model_name, {})

# If the best model has parameters to tune
if best_param_grid:
    print(f"Fine-tuning {best_model_name}...")

    # Create a grid search
    grid_search = GridSearchCV(
        estimator=models[best_model_name],
        param_grid=best_param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"\nBest parameters for {best_model_name}:")
    print(best_params)

    # Get the best model
    best_tuned_model = grid_search.best_estimator_

    # Evaluate the tuned model
    y_train_pred_tuned = best_tuned_model.predict(X_train)
    y_test_pred_tuned = best_tuned_model.predict(X_test)

    # Calculate metrics for the tuned model
    tuned_test_mae = mean_absolute_error(y_test, y_test_pred_tuned)
    tuned_test_mse = mean_squared_error(y_test, y_test_pred_tuned)
    tuned_test_rmse = np.sqrt(tuned_test_mse)
    tuned_test_r2 = r2_score(y_test, y_test_pred_tuned)

    # Print metrics for the tuned model
    print(f"\nEvaluation metrics for tuned {best_model_name}:")
    print(f"Testing MAE: {tuned_test_mae:.2f}")
    print(f"Testing MSE: {tuned_test_mse:.2f}")
    print(f"Testing RMSE: {tuned_test_rmse:.2f}")
    print(f"Testing R²: {tuned_test_r2:.4f}")

    # Compare original vs tuned model
    original_metrics = results[best_model_idx]
    improvement_r2 = (tuned_test_r2 - original_metrics['R²']) * 100

    print(f"\nImprovement in R² after tuning: {improvement_r2:.2f}%")

    # Plot actual vs predicted values for the tuned model
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred_tuned, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.title(f'Actual vs Predicted Charges (Tuned {best_model_name})')
    plt.savefig(f'charts/task2/tuned_{best_model_name.replace(" ", "_").lower()}_actual_vs_predicted.png')
    plt.close()

    # Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        # Get feature names after preprocessing
        preprocessor = best_tuned_model.named_steps['preprocessor']
        regressor = best_tuned_model.named_steps['regressor']

        # Get feature names after one-hot encoding
        cat_features = preprocessor.transformers_[1][1].get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(cat_features)

        # Get feature importances
        importances = regressor.feature_importances_

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Plot feature importances
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(f'Feature Importances (Tuned {best_model_name})')
        plt.tight_layout()
        plt.savefig(f'charts/task2/tuned_{best_model_name.replace(" ", "_").lower()}_feature_importance.png')
        plt.close()

        print("\nTop 10 most important features:")
        print(feature_importance_df.head(10))
else:
    print(f"{best_model_name} doesn't have hyperparameters to tune.")

print("\nTask 2 (Regression) complete!")
