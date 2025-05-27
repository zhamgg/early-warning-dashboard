try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.impute import SimpleImputer
    import os
    print("Base packages loaded successfully!")
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

# File path - Updated to use the new data file
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

# Create target variable: 1 if Net Cash Flows < -1,000,000, else 0
try:
    df['target'] = (df['Net Cash Flows'] < -1000000).astype(int)
    print(f"Target distribution: {df['target'].value_counts()}")
except KeyError:
    print("Error: 'Net Cash Flows' column not found in the dataset.")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Function to evaluate and print metrics
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

metrics = []
models = []

# Random Forest Model
print("\n--- Training Random Forest Model ---")
try:
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    rf_pipeline.fit(X_train, y_train)
    rf_predictions = rf_pipeline.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_predictions, "Random Forest")
    
    # Save Random Forest model with the new name
    try:
        joblib.dump(rf_pipeline, 'EarlyWarningWeights.joblib')
        print("Random Forest model saved to: EarlyWarningWeights.joblib")
    except Exception as e:
        print(f"Warning: Could not save Random Forest model: {e}")
    
    metrics.append(rf_metrics)
    models.append("Random Forest")
except Exception as e:
    print(f"Error training Random Forest model: {e}")

# 3. PyTorch Neural Network (only if PyTorch is available)
if torch_available:
    print("\n--- Training PyTorch Neural Network ---")
    try:
        # Preprocess data for PyTorch
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed)
        X_test_tensor = torch.FloatTensor(X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed)
        y_train_tensor = torch.FloatTensor(y_train.values)
        y_test_tensor = torch.FloatTensor(y_test.values)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor.unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Define the neural network
        input_size = X_train_tensor.shape[1]
        
        class NeuralNetwork(nn.Module):
            def __init__(self, input_size):
                super(NeuralNetwork, self).__init__()
                self.layer1 = nn.Linear(input_size, 128)
                self.layer2 = nn.Linear(128, 64)
                self.layer3 = nn.Linear(64, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = self.relu(self.layer1(x))
                x = self.dropout(x)
                x = self.relu(self.layer2(x))
                x = self.sigmoid(self.layer3(x))
                return x
        
        # Initialize model, loss function, and optimizer
        model = NeuralNetwork(input_size)
        criterion = nn.BCELoss()
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
            y_pred_list = []
            for X_batch, _ in test_loader:
                y_pred = model(X_batch)
                y_pred_list.append(y_pred)
            
            y_pred_tensor = torch.cat(y_pred_list)
            nn_predictions = (y_pred_tensor > 0.5).int().squeeze().numpy()
        
        nn_metrics = evaluate_model(y_test, nn_predictions, "Neural Network")
        
        # Save PyTorch model
        try:
            torch.save(model.state_dict(), 'pytorch_model.pt')
            print("PyTorch Neural Network model saved to: pytorch_model.pt")
        except Exception as e:
            print(f"Warning: Could not save PyTorch model: {e}")
        
        metrics.append(nn_metrics)
        models.append("Neural Network")
    except Exception as e:
        print(f"Error training Neural Network: {e}")
else:
    print("\nSkipping Neural Network training since PyTorch is not available.")

# Compare models
if metrics:
    print("\n--- Model Comparison ---")
    print("F1 Scores:")
    for model_name, model_metrics in zip(models, metrics):
        print(f"{model_name}: {model_metrics['f1']:.4f}")
    
    if len(metrics) > 1:
        best_model_idx = np.argmax([m['f1'] for m in metrics])
        print(f"\nBest model based on F1 score: {models[best_model_idx]}")
    
    print("Models trained and saved successfully!")
else:
    print("\nNo models were successfully trained. Please check the errors above.")
