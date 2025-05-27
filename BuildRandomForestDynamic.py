try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.impute import SimpleImputer
    import os
    print("Base packages loaded successfully!")
    
    # Try to import matplotlib for visualizations, but continue if not available
    try:
        import matplotlib.pyplot as plt
        plotting_available = True
        print("Matplotlib loaded successfully!")
    except ImportError:
        plotting_available = False
        print("Matplotlib not available. Will skip visualizations.")
        
except ImportError as e:
    print(f"Error importing base packages: {e}")
    print("Please run: pip install pandas numpy scikit-learn")
    exit(1)

try:
    import joblib
    print("Joblib loaded successfully!")
except ImportError:
    print("Joblib not found. Using pickle instead.")
    import pickle as joblib

# Try to import PyTorch, but continue with only scikit-learn models if it fails
torch_available = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    torch_available = True
    print("PyTorch loaded successfully!")
except ImportError:
    print("PyTorch not available. Will skip neural network model.")
    print("To install PyTorch, please visit: https://pytorch.org/get-started/locally/")

# File path
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

# Print column names to debug
print("\nAvailable columns in the dataset:")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

# Print summary of the numerical columns
print("\nNumerical columns summary:")
numerical_summary = df.describe().T
print(numerical_summary[['mean', 'min', 'max']])

# Use 'Net Cash Flows' as the target for regression
try:
    # Create a copy of the target value before preprocessing
    y_actual = df['Net Cash Flows'].copy()
    
    print(f"\nNet Cash Flows summary statistics:")
    print(f"Mean: ${y_actual.mean():,.2f}")
    print(f"Min: ${y_actual.min():,.2f}")
    print(f"Max: ${y_actual.max():,.2f}")
    print(f"Median: ${y_actual.median():,.2f}")
    
    # Show histogram of Net Cash Flows if plotting is available
    if plotting_available:
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(y_actual, bins=50)
            plt.title('Distribution of Net Cash Flows')
            plt.xlabel('Net Cash Flows ($)')
            plt.ylabel('Frequency')
            plt.savefig('net_cash_flows_distribution.png')
            plt.close()
            print("Saved distribution plot to net_cash_flows_distribution.png")
        except Exception as e:
            print(f"Warning: Could not create distribution plot: {e}")
            
except KeyError:
    print("Error: 'Net Cash Flows' column not found in the dataset.")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Separate features and target for regression
X = df.drop('Net Cash Flows', axis=1)
y = df['Net Cash Flows']  # Using actual cash flow values

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Found {len(categorical_cols)} categorical columns and {len(numerical_cols)} numerical columns.")

# Create preprocessing pipelines
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Function to evaluate and print regression metrics
def evaluate_regression_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"Mean Squared Error: ${mse:,.2f}")
    print(f"Root Mean Squared Error: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Create visualizations only if plotting is available
    if plotting_available:
        try:
            # Scatter plot of actual vs predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([-1e9, 1e9], [-1e9, 1e9], 'r--')  # Diagonal line for perfect predictions
            plt.title(f'{model_name}: Actual vs Predicted Cash Flows')
            plt.xlabel('Actual Cash Flows ($)')
            plt.ylabel('Predicted Cash Flows ($)')
            plt.savefig(f'{model_name.lower().replace(" ", "_")}_prediction_scatter.png')
            plt.close()
            print(f"Saved scatter plot to {model_name.lower().replace(' ', '_')}_prediction_scatter.png")
            
            # Calculate prediction errors
            errors = y_true - y_pred
            
            # Plot histogram of errors
            plt.figure(figsize=(10, 6))
            plt.hist(errors, bins=50)
            plt.title(f'{model_name}: Prediction Errors')
            plt.xlabel('Error ($)')
            plt.ylabel('Frequency')
            plt.savefig(f'{model_name.lower().replace(" ", "_")}_error_hist.png')
            plt.close()
            print(f"Saved error histogram to {model_name.lower().replace(' ', '_')}_error_hist.png")
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
    
    # Return metrics dictionary
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

metrics = []
models = []

# Random Forest Regressor Model
print("\n--- Training Random Forest Regressor Model ---")
try:
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    rf_pipeline.fit(X_train, y_train)
    rf_predictions = rf_pipeline.predict(X_test)
    rf_metrics = evaluate_regression_model(y_test, rf_predictions, "Random Forest")
    
    # Save Random Forest model
    try:
        joblib.dump(rf_pipeline, 'CashFlowPredictionModel.joblib')
        print("Random Forest regression model saved to: CashFlowPredictionModel.joblib")
    except Exception as e:
        print(f"Warning: Could not save Random Forest model: {e}")
    
    # Feature importance analysis if plotting is available
    if plotting_available and hasattr(rf_pipeline['regressor'], 'feature_importances_'):
        try:
            # Get feature names after preprocessing (this is approximate since OneHotEncoder creates new features)
            feature_names = numerical_cols.copy()
            # Add categorical feature names (this is simplified)
            feature_names.extend(categorical_cols)
            
            importances = rf_pipeline['regressor'].feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20 features
            
            plt.figure(figsize=(12, 8))
            plt.title('Top 20 Feature Importances')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices])
            plt.xlabel('Relative Importance')
            plt.savefig('feature_importance.png')
            plt.close()
            print("Saved feature importance plot to feature_importance.png")
        except Exception as e:
            print(f"Warning: Could not create feature importance plot: {e}")
    
    metrics.append(rf_metrics)
    models.append("Random Forest")
except Exception as e:
    print(f"Error training Random Forest regression model: {e}")

# PyTorch Neural Network (only if PyTorch is available)
if torch_available:
    print("\n--- Training PyTorch Neural Network Regressor ---")
    try:
        # Preprocess data for PyTorch
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Scale the target values for better neural network training
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed)
        X_test_tensor = torch.FloatTensor(X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed)
        y_train_tensor = torch.FloatTensor(y_train_scaled)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Define the neural network for regression
        input_size = X_train_tensor.shape[1]
        
        class RegressionNN(nn.Module):
            def __init__(self, input_size):
                super(RegressionNN, self).__init__()
                self.layer1 = nn.Linear(input_size, 128)
                self.layer2 = nn.Linear(128, 64)
                self.layer3 = nn.Linear(64, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = self.relu(self.layer1(x))
                x = self.dropout(x)
                x = self.relu(self.layer2(x))
                x = self.layer3(x)  # No activation for regression
                return x
        
        # Initialize model, loss function, and optimizer
        model = RegressionNN(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the neural network
        num_epochs = 20
        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Evaluate the neural network
        model.eval()
        with torch.no_grad():
            # Get predictions
            nn_predictions_scaled = model(X_test_tensor).squeeze().numpy()
            # Inverse transform to get original scale
            nn_predictions = y_scaler.inverse_transform(nn_predictions_scaled.reshape(-1, 1)).flatten()
        
        nn_metrics = evaluate_regression_model(y_test, nn_predictions, "Neural Network")
        
        # Save PyTorch model
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': y_scaler
            }, 'pytorch_regression_model.pt')
            print("PyTorch Neural Network regression model saved to: pytorch_regression_model.pt")
        except Exception as e:
            print(f"Warning: Could not save PyTorch model: {e}")
        
        metrics.append(nn_metrics)
        models.append("Neural Network")
    except Exception as e:
        print(f"Error training Neural Network regressor: {e}")
else:
    print("\nSkipping Neural Network training since PyTorch is not available.")

# Compare models
if metrics:
    print("\n--- Model Comparison ---")
    print("R² Scores:")
    for model_name, model_metrics in zip(models, metrics):
        print(f"{model_name}: {model_metrics['r2']:.4f}")
    
    if len(metrics) > 1:
        best_model_idx = np.argmax([m['r2'] for m in metrics])
        print(f"\nBest model based on R² score: {models[best_model_idx]}")
    
    print("Models trained and saved successfully!")
else:
    print("\nNo models were successfully trained. Please check the errors above.")

# Function to predict cash flow for a specific fund
def predict_cash_flow_for_fund(model_path, fund_data):
    """
    Predict cash flow for a specific fund using the trained model
    
    Parameters:
    model_path (str): Path to the saved model
    fund_data (pandas.DataFrame): Data for the fund to predict
    
    Returns:
    float: Predicted cash flow
    """
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Make prediction
        prediction = model.predict(fund_data)
        
        return prediction[0]
    except Exception as e:
        print(f"Error predicting cash flow: {e}")
        return None

# Example of how to use the model to predict for a specific fund
print("\n--- Example Prediction for a Specific Fund ---")
try:
    # Select a random fund from the dataset as an example
    random_index = np.random.randint(0, len(df))
    fund_example = df.iloc[[random_index]].copy()
    
    # Store the actual cash flow and remove it from the feature set
    actual_cash_flow = fund_example['Net Cash Flows'].values[0]
    fund_example_X = fund_example.drop('Net Cash Flows', axis=1)
    
    # Show information about the fund
    if 'Fund ID' in fund_example.columns:
        fund_id = fund_example['Fund ID'].values[0]
        print(f"Selected Fund ID: {fund_id}")
    elif 'Fund Name_x' in fund_example.columns:
        fund_name = fund_example['Fund Name_x'].values[0]
        print(f"Selected Fund: {fund_name}")
    else:
        print("Selected a random fund from the dataset")
    
    # Load the model
    model_path = 'CashFlowPredictionModel.joblib'
    
    # Predict cash flow
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        predicted_cash_flow = model.predict(fund_example_X)[0]
        
        print(f"Actual Cash Flow: ${actual_cash_flow:,.2f}")
        print(f"Predicted Cash Flow: ${predicted_cash_flow:,.2f}")
        print(f"Absolute Error: ${abs(actual_cash_flow - predicted_cash_flow):,.2f}")
        print(f"Relative Error: {100 * abs(actual_cash_flow - predicted_cash_flow) / (abs(actual_cash_flow) + 1e-10):.2f}%")
    else:
        print(f"Model file {model_path} not found. Run the full script first to train and save the model.")
except Exception as e:
    print(f"Error in example prediction: {e}")

print("\nScript completed. You can now use the trained model to predict cash flows for specific funds.")
