import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import os
import joblib

print("Starting AUM percentage change prediction model training...")

# === File path ===
file_path = r"joined_fund_data2.xlsx"

# === Load the data ===
print("Loading Excel data...")
try:
    df = pd.read_excel(file_path)
    print(f"Data loaded successfully.")
    print(f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    print(f"Failed to load Excel file: {e}")
    exit(1)

# === Show available columns ===
print("\nAvailable columns in the dataset:")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

# === Calculate AUM percentage change ===
try:
    df['Beginning Net Assets'] = pd.to_numeric(df['Beginning Net Assets'], errors='coerce')
    df['Ending Net Assets'] = pd.to_numeric(df['Ending Net Assets'], errors='coerce')

    df['AUM_Pct_Change'] = ((df['Ending Net Assets'] - df['Beginning Net Assets']) / df['Beginning Net Assets']) * 100

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['AUM_Pct_Change'])

    # ✅ Aggressive capping
    df['AUM_Pct_Change'] = df['AUM_Pct_Change'].clip(-50, 50)

    print(f"\nAUM Percentage Change summary statistics:")
    print(f"Mean: {df['AUM_Pct_Change'].mean():.2f}%")
    print(f"Min: {df['AUM_Pct_Change'].min():.2f}%")
    print(f"Max: {df['AUM_Pct_Change'].max():.2f}%")
    print(f"Median: {df['AUM_Pct_Change'].median():.2f}%")
    print(f"Number of rows after preprocessing: {df.shape[0]}")

except Exception as e:
    print(f"Error calculating AUM percentage change: {e}")
    exit(1)

# === Prepare features and target ===
columns_to_drop = ['AUM_Pct_Change', 'Ending Net Assets', 'Net Change AUM']
X = df.drop(columns_to_drop, axis=1)
y = df['AUM_Pct_Change']

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Found {len(categorical_cols)} categorical columns and {len(numerical_cols)} numerical columns.")

# === Preprocessing pipelines ===
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# === Evaluation function ===
def evaluate_regression_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    explained_var = explained_variance_score(y_true, y_pred)

    # Clean MAPE: skip near-zero actuals
    y_true_array = np.array(y_true)
    mask = np.abs(y_true_array) > 1
    mape = np.mean(np.abs((y_true_array[mask] - y_pred[mask]) / (np.abs(y_true_array[mask]) + 1e-8))) * 100

    residuals = y_true - y_pred

    print(f"\n{model_name} Performance Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"Explained Variance Score: {explained_var:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}%")
    print(f"Median Absolute Error: {median_ae:.4f}%")
    print(f"Root Mean Squared Error: {rmse:.4f}%")
    print(f"Clean MAPE: {mape:.4f}%")
    print(f"Residuals Mean: {residuals.mean():.4f}%")
    print(f"Residuals Std Dev: {residuals.std():.4f}%")

    return {
        'r2': r2,
        'explained_variance': explained_var,
        'mae': mae,
        'median_ae': median_ae,
        'rmse': rmse,
        'mape': mape,
        'residuals_mean': residuals.mean(),
        'residuals_std': residuals.std()
    }

# === Model training + CV ===
print("\n--- Training Random Forest Regressor Model with Cross-Validation ---")
try:
    rf_regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf_regressor)
    ])

    print("Performing 5-fold cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring='r2')

    print("\nCross-Validation Results:")
    print(f"R² Scores: {cv_scores}")
    print(f"Mean R²: {cv_scores.mean():.4f}")
    print(f"Std Dev of R²: {cv_scores.std():.4f}")

    print("\nTraining final model on full training set...")
    rf_pipeline.fit(X_train, y_train)
    rf_predictions = rf_pipeline.predict(X_test)

    rf_metrics = evaluate_regression_model(y_test, rf_predictions, "Random Forest")

    joblib.dump(rf_pipeline, 'AUMChangeModel.joblib')
    print("Random Forest model saved to: AUMChangeModel.joblib")

except Exception as e:
    print(f"Error training model: {e}")

# === Example prediction ===
print("\n--- Example Prediction for a Random Fund ---")
try:
    random_index = np.random.randint(0, len(df))
    fund_example = df.iloc[[random_index]].copy()
    actual = fund_example['AUM_Pct_Change'].values[0]
    fund_input = fund_example.drop(columns_to_drop, axis=1)

    model = joblib.load('AUMChangeModel.joblib')
    predicted = model.predict(fund_input)[0]

    print(f"Actual AUM % Change: {actual:.2f}%")
    print(f"Predicted AUM % Change: {predicted:.2f}%")
    print(f"Absolute Error: {abs(actual - predicted):.2f}%")
    print(f"Relative Error: {100 * abs(actual - predicted) / (abs(actual) + 1e-8):.2f}%")

except Exception as e:
    print(f"Error in example prediction: {e}")

print("\n✅ Script completed.")
