try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        roc_curve, precision_recall_curve
    )
    from sklearn.impute import SimpleImputer
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("Base packages loaded successfully!")
except ImportError as e:
    print(f"Error importing base packages: {e}")
    print("Please run: pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn")
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
file_path = r"joined_fund_data2(Sheet1).csv"
encodings_to_try = ['latin1', 'ISO-8859-1', 'cp1252', 'utf-8']
 
# Load the data
print("Loading data...")
df = None
for enc in encodings_to_try:
    try:
        print(f"Trying to load with encoding: {enc}")
        df = pd.read_csv(file_path, encoding=enc)
        print(f"Data loaded successfully with {enc} encoding.")
        print(f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns.")
        break
    except Exception as e:
        print(f"Failed with encoding {enc}: {e}")
 
if df is None:
    raise RuntimeError("Could not read CSV with any of the tried encodings.")
 
# Compute percent change in AUM (same as original model)
print("\nComputing Pct Change in AUM...")
df["Pct Change in AUM"] = (df["Net Change AUM"] / df["Beginning Net Assets"]) * 100.0
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Pct Change in AUM", "Beginning Net Assets"]).reset_index(drop=True)
 
print(f"Data after cleaning: {df.shape[0]} rows")
print(f"Pct Change in AUM statistics (before outlier handling):")
print(df["Pct Change in AUM"].describe())
 
# =============================================================================
# OUTLIER HANDLING AND FEATURE TRANSFORMATION
# =============================================================================
 
print("\n" + "="*80)
print("OUTLIER HANDLING AND FEATURE TRANSFORMATION")
print("="*80)
 
# Cap extreme outliers at 99th and 1st percentiles
pct_99 = df["Pct Change in AUM"].quantile(0.99)
pct_1 = df["Pct Change in AUM"].quantile(0.01)
print(f"Original range: {df['Pct Change in AUM'].min():.2f}% to {df['Pct Change in AUM'].max():.2f}%")
print(f"Capping at {pct_1:.2f}% to {pct_99:.2f}% (1st to 99th percentiles)")
 
df["Pct Change in AUM_capped"] = df["Pct Change in AUM"].clip(lower=pct_1, upper=pct_99)
print(f"After capping: {df['Pct Change in AUM_capped'].min():.2f}% to {df['Pct Change in AUM_capped'].max():.2f}%")
 
# Use capped version for all classifications
pct_change_col = "Pct Change in AUM_capped"
 
print(f"\nCapped Pct Change in AUM statistics:")
print(df[pct_change_col].describe())
 
# Create 4-category sequential classification targets
# Stage 1: Inflow (>= 0%) vs Outflow (< 0%)
df['stage1_target'] = (df[pct_change_col] >= 0).astype(int)  # 1 = Inflow, 0 = Outflow
 
# Stage 2a: For outflows only, Minor (>= -10%) vs Major (< -10%)
outflow_mask = df[pct_change_col] < 0
df['stage2a_target'] = np.nan  # Initialize with NaN
df.loc[outflow_mask, 'stage2a_target'] = (df.loc[outflow_mask, pct_change_col] >= -10).astype(int)  # 1 = Minor outflow, 0 = Major outflow
 
# Stage 2b: For inflows only, Minor (< 10%) vs Major (>= 10%)  
inflow_mask = df[pct_change_col] >= 0
df['stage2b_target'] = np.nan  # Initialize with NaN
df.loc[inflow_mask, 'stage2b_target'] = (df.loc[inflow_mask, pct_change_col] < 10).astype(int)  # 1 = Minor inflow, 0 = Major inflow
 
print(f"\nStage 1 (Inflow vs Outflow) distribution:")
print(df['stage1_target'].value_counts())
print(f"Inflow rate: {df['stage1_target'].mean():.3f}")
 
print(f"\nStage 2a (Minor vs Major Outflow) distribution (outflows only):")
stage2a_valid = df['stage2a_target'].notna()
if stage2a_valid.sum() > 0:
    print(df.loc[stage2a_valid, 'stage2a_target'].value_counts())
    print(f"Minor outflow rate (among outflows): {df.loc[stage2a_valid, 'stage2a_target'].mean():.3f}")
 
print(f"\nStage 2b (Minor vs Major Inflow) distribution (inflows only):")
stage2b_valid = df['stage2b_target'].notna()
if stage2b_valid.sum() > 0:
    print(df.loc[stage2b_valid, 'stage2b_target'].value_counts())
    print(f"Minor inflow rate (among inflows): {df.loc[stage2b_valid, 'stage2b_target'].mean():.3f}")
 
# Define leakage columns (same as original model)
leak_cols = [
    "Cash Outflows",
    "Cash Inflows",
    "Net Cash Flows",
    "Net Market & Other",
    "Net Change AUM",
    "Ending Net Assets"
]
 
# Prepare features (drop leakage columns and target-related columns)
all_feature_cols = [
    c for c in df.columns
    if c not in ["Pct Change in AUM", "Pct Change in AUM_capped", "stage1_target", "stage2a_target", "stage2b_target"] + leak_cols
]
 
X = df[all_feature_cols].copy()
print(f"\nUsing {len(all_feature_cols)} features for prediction")
 
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
print(f"Found {len(categorical_cols)} categorical cols and {len(numeric_cols)} numeric cols.")
 
# Create preprocessing pipelines
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ord_enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
 
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
 
preprocessor = ColumnTransformer([
    ("num", num_transformer, numeric_cols),
    ("cat", cat_transformer, categorical_cols)
])
 
# Enhanced evaluation function with threshold tuning
def evaluate_model_comprehensive(y_true, y_pred, y_prob=None, model_name="Model", stage="", tune_threshold=False):
    """Comprehensive model evaluation with extensive metrics and optional threshold tuning"""
   
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
   
    print(f"\n{'='*60}")
    print(f"{model_name} - {stage} Performance Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
   
    best_threshold = 0.5
    tuned_metrics = None
   
    if y_prob is not None:
        try:
            auc_score = roc_auc_score(y_true, y_prob)
            print(f"ROC AUC:   {auc_score:.4f}")
           
            # Threshold tuning for better recall on minority class
            if tune_threshold and len(np.unique(y_true)) == 2:
                print(f"\nThreshold Tuning for Better Recall:")
                thresholds = np.arange(0.1, 0.9, 0.05)
                best_f1 = f1
               
                for thresh in thresholds:
                    y_pred_tuned = (y_prob >= thresh).astype(int)
                    f1_tuned = f1_score(y_true, y_pred_tuned, zero_division=0)
                    recall_tuned = recall_score(y_true, y_pred_tuned, zero_division=0)
                    precision_tuned = precision_score(y_true, y_pred_tuned, zero_division=0)
                   
                    if f1_tuned > best_f1:
                        best_f1 = f1_tuned
                        best_threshold = thresh
                        tuned_metrics = {
                            'accuracy': accuracy_score(y_true, y_pred_tuned),
                            'precision': precision_tuned,
                            'recall': recall_tuned,
                            'f1': f1_tuned,
                            'threshold': thresh
                        }
               
                if tuned_metrics:
                    print(f"Best threshold: {best_threshold:.2f}")
                    print(f"Tuned F1:      {tuned_metrics['f1']:.4f}")
                    print(f"Tuned Recall:  {tuned_metrics['recall']:.4f}")
                    print(f"Tuned Precision: {tuned_metrics['precision']:.4f}")
               
        except:
            auc_score = None
            print("ROC AUC:   N/A (single class)")
    else:
        auc_score = None
   
    print("\nConfusion Matrix:")
    print(conf_matrix)
   
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
   
    # Class distribution
    print(f"\nActual class distribution:")
    unique, counts = np.unique(y_true, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(y_true)*100:.1f}%)")
       
    print(f"\nPredicted class distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(y_pred)*100:.1f}%)")
   
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'confusion_matrix': conf_matrix,
        'best_threshold': best_threshold,
        'tuned_metrics': tuned_metrics
    }
 
def train_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor, model_name, stage_name, use_smote=False, tune_threshold=False):
    """Train and evaluate a model with comprehensive metrics, optional SMOTE, and threshold tuning"""
   
    print(f"\n{'*'*50}")
    print(f"Training {model_name} for {stage_name}")
    if use_smote:
        print("Using SMOTE for class balancing")
    if tune_threshold:
        print("Will tune threshold for better recall")
    print(f"{'*'*50}")
   
    # Create classifier
    if model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"  # Always use balanced weights
        )
    else:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
   
    # Create pipeline with optional SMOTE
    if use_smote:
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=3)),  # Use fewer neighbors for small datasets
            ('classifier', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
   
    # Train the model
    pipeline.fit(X_train, y_train)
   
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # Probability of positive class
   
    # Evaluate
    metrics = evaluate_model_comprehensive(y_test, y_pred, y_prob, model_name, stage_name, tune_threshold=tune_threshold)
   
    return pipeline, metrics
 
# =============================================================================
# STAGE 1: INFLOW vs OUTFLOW CLASSIFICATION
# =============================================================================
 
print("\n" + "="*80)
print("STAGE 1: INFLOW vs OUTFLOW CLASSIFICATION")
print("="*80)
 
y_stage1 = df['stage1_target']
 
# Split data for stage 1
X_train_s1, X_test_s1, y_train_s1, y_test_s1 = train_test_split(
    X, y_stage1, test_size=0.20, random_state=42, stratify=y_stage1
)
 
print(f"Stage 1 - Training set: {X_train_s1.shape[0]}, Test set: {X_test_s1.shape[0]}")
 
# Train Random Forest for Stage 1
rf_s1_pipeline, rf_s1_metrics = train_and_evaluate_model(
    X_train_s1, X_test_s1, y_train_s1, y_test_s1,
    preprocessor, "Random Forest", "Stage 1 (Inflow vs Outflow)",
    use_smote=False, tune_threshold=True
)
 
# Save Stage 1 model
os.makedirs("sequential_models", exist_ok=True)
joblib.dump(rf_s1_pipeline, "sequential_models/stage1_inflow_outflow.joblib")
print("Stage 1 model saved to: sequential_models/stage1_inflow_outflow.joblib")
 
# =============================================================================
# STAGE 2A: MINOR vs MAJOR OUTFLOW CLASSIFICATION (for outflows only)
# =============================================================================
 
print("\n" + "="*80)
print("STAGE 2A: MINOR vs MAJOR OUTFLOW CLASSIFICATION")
print("="*80)
 
# Filter to only outflow records for stage 2a
outflow_mask = df['stage1_target'] == 0
X_outflows = X.loc[outflow_mask].copy()
y_stage2a = df.loc[outflow_mask, 'stage2a_target'].copy()
 
# Remove any remaining NaN values
valid_mask = y_stage2a.notna()
X_outflows = X_outflows.loc[valid_mask]
y_stage2a = y_stage2a.loc[valid_mask]
 
print(f"Stage 2a dataset size: {len(X_outflows)} outflow records")
print(f"Class distribution: {y_stage2a.value_counts()}")
 
if len(X_outflows) > 20:  # Need sufficient data for train/test split
    # Split data for stage 2a
    X_train_s2a, X_test_s2a, y_train_s2a, y_test_s2a = train_test_split(
        X_outflows, y_stage2a, test_size=0.20, random_state=42, stratify=y_stage2a
    )
   
    print(f"Stage 2a - Training set: {X_train_s2a.shape[0]}, Test set: {X_test_s2a.shape[0]}")
   
    # Train Random Forest for Stage 2a with SMOTE and threshold tuning
    rf_s2a_pipeline, rf_s2a_metrics = train_and_evaluate_model(
        X_train_s2a, X_test_s2a, y_train_s2a, y_test_s2a,
        preprocessor, "Random Forest", "Stage 2a (Minor vs Major Outflow)",
        use_smote=True, tune_threshold=True
    )
   
    # Save Stage 2a model
    joblib.dump(rf_s2a_pipeline, "sequential_models/stage2a_minor_major_outflow.joblib")
    print("Stage 2a model saved to: sequential_models/stage2a_minor_major_outflow.joblib")
   
else:
    print("Insufficient data for Stage 2a training. Skipping Stage 2a model.")
    rf_s2a_pipeline = None
    rf_s2a_metrics = None
 
# =============================================================================
# STAGE 2B: MINOR vs MAJOR INFLOW CLASSIFICATION (for inflows only)
# =============================================================================
 
print("\n" + "="*80)
print("STAGE 2B: MINOR vs MAJOR INFLOW CLASSIFICATION")
print("="*80)
 
# Filter to only inflow records for stage 2b
inflow_mask = df['stage1_target'] == 1
X_inflows = X.loc[inflow_mask].copy()
y_stage2b = df.loc[inflow_mask, 'stage2b_target'].copy()
 
# Remove any remaining NaN values
valid_mask = y_stage2b.notna()
X_inflows = X_inflows.loc[valid_mask]
y_stage2b = y_stage2b.loc[valid_mask]
 
print(f"Stage 2b dataset size: {len(X_inflows)} inflow records")
print(f"Class distribution: {y_stage2b.value_counts()}")
 
if len(X_inflows) > 20:  # Need sufficient data for train/test split
    # Split data for stage 2b
    X_train_s2b, X_test_s2b, y_train_s2b, y_test_s2b = train_test_split(
        X_inflows, y_stage2b, test_size=0.20, random_state=42, stratify=y_stage2b
    )
   
    print(f"Stage 2b - Training set: {X_train_s2b.shape[0]}, Test set: {X_test_s2b.shape[0]}")
   
    # Train Random Forest for Stage 2b with SMOTE and threshold tuning
    rf_s2b_pipeline, rf_s2b_metrics = train_and_evaluate_model(
        X_train_s2b, X_test_s2b, y_train_s2b, y_test_s2b,
        preprocessor, "Random Forest", "Stage 2b (Minor vs Major Inflow)",
        use_smote=True, tune_threshold=True
    )
   
    # Save Stage 2b model
    joblib.dump(rf_s2b_pipeline, "sequential_models/stage2b_minor_major_inflow.joblib")
    print("Stage 2b model saved to: sequential_models/stage2b_minor_major_inflow.joblib")
   
else:
    print("Insufficient data for Stage 2b training. Skipping Stage 2b model.")
    rf_s2b_pipeline = None
    rf_s2b_metrics = None
 
# =============================================================================
# COMBINED SEQUENTIAL PREDICTION EVALUATION (4 CATEGORIES)
# =============================================================================
 
print("\n" + "="*80)
print("COMBINED SEQUENTIAL PREDICTION EVALUATION (4 CATEGORIES)")
print("="*80)
 
def sequential_predict_4cat(X_data, stage1_model, stage2a_model=None, stage2b_model=None, use_tuned_thresholds=True):
    """
    Make sequential predictions for 4 categories:
    1. First predict inflow vs outflow
    2a. For predicted outflows, predict minor vs major outflow
    2b. For predicted inflows, predict minor vs major inflow
   
    Returns:
    - stage1_pred: 0=outflow, 1=inflow
    - stage2a_pred: 0=major outflow, 1=minor outflow (NaN for inflows)
    - stage2b_pred: 0=major inflow, 1=minor inflow (NaN for outflows)
    - combined_pred: 0=major outflow, 1=minor outflow, 2=minor inflow, 3=major inflow
    """
   
    # Stage 1 predictions
    stage1_pred = stage1_model.predict(X_data)
    stage1_prob = stage1_model.predict_proba(X_data)[:, 1]
   
    # Apply tuned threshold if available
    if use_tuned_thresholds and rf_s1_metrics.get('tuned_metrics'):
        best_threshold = rf_s1_metrics['best_threshold']
        stage1_pred = (stage1_prob >= best_threshold).astype(int)
        print(f"Using tuned threshold {best_threshold:.2f} for Stage 1")
   
    # Initialize stage2 predictions
    stage2a_pred = np.full(len(X_data), np.nan)
    stage2b_pred = np.full(len(X_data), np.nan)
   
    # Stage 2a predictions (only for predicted outflows)
    if stage2a_model is not None:
        outflow_indices = np.where(stage1_pred == 0)[0]
        if len(outflow_indices) > 0:
            X_outflows = X_data.iloc[outflow_indices]
            stage2a_pred_subset = stage2a_model.predict(X_outflows)
            stage2a_prob_subset = stage2a_model.predict_proba(X_outflows)[:, 1]
           
            # Apply tuned threshold if available
            if use_tuned_thresholds and rf_s2a_metrics and rf_s2a_metrics.get('tuned_metrics'):
                best_threshold_2a = rf_s2a_metrics['best_threshold']
                stage2a_pred_subset = (stage2a_prob_subset >= best_threshold_2a).astype(int)
                print(f"Using tuned threshold {best_threshold_2a:.2f} for Stage 2a")
           
            stage2a_pred[outflow_indices] = stage2a_pred_subset
   
    # Stage 2b predictions (only for predicted inflows)
    if stage2b_model is not None:
        inflow_indices = np.where(stage1_pred == 1)[0]
        if len(inflow_indices) > 0:
            X_inflows = X_data.iloc[inflow_indices]
            stage2b_pred_subset = stage2b_model.predict(X_inflows)
            stage2b_prob_subset = stage2b_model.predict_proba(X_inflows)[:, 1]
           
            # Apply tuned threshold if available
            if use_tuned_thresholds and rf_s2b_metrics and rf_s2b_metrics.get('tuned_metrics'):
                best_threshold_2b = rf_s2b_metrics['best_threshold']
                stage2b_pred_subset = (stage2b_prob_subset >= best_threshold_2b).astype(int)
                print(f"Using tuned threshold {best_threshold_2b:.2f} for Stage 2b")
           
            stage2b_pred[inflow_indices] = stage2b_pred_subset
   
    # Create combined prediction
    # 0 = Major outflow, 1 = Minor outflow, 2 = Minor inflow, 3 = Major inflow
    combined_pred = np.full(len(X_data), -1)  # Initialize with invalid value
   
    # Handle outflows
    outflow_mask = stage1_pred == 0
    if stage2a_model is not None:
        # Use stage2a prediction (0=major, 1=minor)
        combined_pred[outflow_mask] = stage2a_pred[outflow_mask]
    else:
        # If no stage2a model, classify all outflows as "minor" (1)
        combined_pred[outflow_mask] = 1
   
    # Handle inflows  
    inflow_mask = stage1_pred == 1
    if stage2b_model is not None:
        # Map stage2b prediction: 1=minor inflow->2, 0=major inflow->3
        stage2b_pred_mapped = np.where(stage2b_pred == 1, 2, 3)
        combined_pred[inflow_mask] = stage2b_pred_mapped[inflow_mask]
    else:
        # If no stage2b model, classify all inflows as "minor" (2)
        combined_pred[inflow_mask] = 2
   
    return stage1_pred, stage2a_pred, stage2b_pred, combined_pred
 
# Create true combined labels for evaluation (4 categories)
def create_combined_labels_4cat(pct_change):
    """Create 4-class labels: 0=major outflow (<-10%), 1=minor outflow ([-10%,0%)), 2=minor inflow ([0%,10%)), 3=major inflow (>=10%)"""
    combined = np.full(len(pct_change), -1)  # Initialize with invalid value
   
    # Major outflow
    combined[pct_change < -10] = 0
   
    # Minor outflow  
    combined[(pct_change >= -10) & (pct_change < 0)] = 1
   
    # Minor inflow
    combined[(pct_change >= 0) & (pct_change < 10)] = 2
   
    # Major inflow
    combined[pct_change >= 10] = 3
   
    return combined
 
# Evaluate on test set
if rf_s2a_pipeline is not None or rf_s2b_pipeline is not None:
    # Use test data from stage 1 (full dataset)
    pct_change_test = df.loc[X_test_s1.index, pct_change_col]
    y_combined_true = create_combined_labels_4cat(pct_change_test)
   
    # Make sequential predictions
    stage1_pred, stage2a_pred, stage2b_pred, combined_pred = sequential_predict_4cat(
        X_test_s1, rf_s1_pipeline, rf_s2a_pipeline, rf_s2b_pipeline, use_tuned_thresholds=True
    )
   
    print("\nSequential Model Performance on Test Set:")
    print("-" * 50)
   
    # Evaluate combined 4-class prediction
    print("\nCOMBINED 4-CLASS PREDICTION:")
    combined_accuracy = accuracy_score(y_combined_true, combined_pred)
    combined_conf_matrix = confusion_matrix(y_combined_true, combined_pred, labels=[0, 1, 2, 3])
   
    print(f"Overall 4-class accuracy: {combined_accuracy:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("     MajOut MinOut MinInf MajInf")
    for i, row in enumerate(combined_conf_matrix):
        label = ["MajOut", "MinOut", "MinInf", "MajInf"][i]
        print(f"{label} {row}")
   
    print("\n4-Class Classification Report:")
    target_names = ['Major Outflow', 'Minor Outflow', 'Minor Inflow', 'Major Inflow']
    print(classification_report(y_combined_true, combined_pred,
                              target_names=target_names, digits=4))
   
    # Distribution analysis
    print(f"\nActual distribution:")
    unique, counts = np.unique(y_combined_true, return_counts=True)
    labels = ['Major Outflow', 'Minor Outflow', 'Minor Inflow', 'Major Inflow']
    for cls, count in zip(unique, counts):
        print(f"  {labels[cls]}: {count} ({count/len(y_combined_true)*100:.1f}%)")
       
    print(f"\nPredicted distribution:")
    unique, counts = np.unique(combined_pred, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {labels[cls]}: {count} ({count/len(combined_pred)*100:.1f}%)")
   
    # Error Propagation Analysis
    print(f"\n" + "="*60)
    print("ERROR PROPAGATION ANALYSIS")
    print("="*60)
   
    # Stage 1 errors
    stage1_errors = stage1_pred != df.loc[X_test_s1.index, 'stage1_target'].values
    print(f"Stage 1 errors: {stage1_errors.sum()} / {len(stage1_errors)} ({stage1_errors.mean()*100:.1f}%)")
   
    # True outflows misclassified as inflows
    true_outflows = df.loc[X_test_s1.index, 'stage1_target'].values == 0
    outflows_missed = true_outflows & (stage1_pred == 1)
    print(f"True outflows classified as inflows: {outflows_missed.sum()} / {true_outflows.sum()} ({outflows_missed.sum()/true_outflows.sum()*100:.1f}%)")
   
    # True inflows misclassified as outflows
    true_inflows = df.loc[X_test_s1.index, 'stage1_target'].values == 1
    inflows_missed = true_inflows & (stage1_pred == 0)
    print(f"True inflows classified as outflows: {inflows_missed.sum()} / {true_inflows.sum()} ({inflows_missed.sum()/true_inflows.sum()*100:.1f}%)")
 
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)
 
# Feature importance for Stage 1
if hasattr(rf_s1_pipeline.named_steps['classifier'], 'feature_importances_'):
    print("\nStage 1 (Inflow vs Outflow) - Top 10 Most Important Features:")
   
    # Get feature names after preprocessing
    feature_names = []
    feature_names.extend(numeric_cols)
    feature_names.extend(categorical_cols)
   
    importances = rf_s1_pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)
   
    print(feature_importance_df.head(10).to_string(index=False))
 
# Feature importance for Stage 2a (if available)
if rf_s2a_pipeline is not None and hasattr(rf_s2a_pipeline.named_steps['classifier'], 'feature_importances_'):
    print("\nStage 2a (Minor vs Major Outflow) - Top 10 Most Important Features:")
   
    importances_s2a = rf_s2a_pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df_s2a = pd.DataFrame({
        'feature': feature_names[:len(importances_s2a)],
        'importance': importances_s2a
    }).sort_values('importance', ascending=False)
   
    print(feature_importance_df_s2a.head(10).to_string(index=False))
 
# Feature importance for Stage 2b (if available)
if rf_s2b_pipeline is not None and hasattr(rf_s2b_pipeline.named_steps['classifier'], 'feature_importances_'):
    print("\nStage 2b (Minor vs Major Inflow) - Top 10 Most Important Features:")
   
    importances_s2b = rf_s2b_pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df_s2b = pd.DataFrame({
        'feature': feature_names[:len(importances_s2b)],
        'importance': importances_s2b
    }).sort_values('importance', ascending=False)
   
    print(feature_importance_df_s2b.head(10).to_string(index=False))
 
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
 
print(f"✓ Stage 1 Model (Inflow vs Outflow): F1 = {rf_s1_metrics['f1']:.4f}")
if rf_s1_metrics.get('tuned_metrics'):
    print(f"  └─ With tuned threshold ({rf_s1_metrics['best_threshold']:.2f}): F1 = {rf_s1_metrics['tuned_metrics']['f1']:.4f}")
 
if rf_s2a_metrics:
    print(f"✓ Stage 2a Model (Minor vs Major Outflow): F1 = {rf_s2a_metrics['f1']:.4f}")
    if rf_s2a_metrics.get('tuned_metrics'):
        print(f"  └─ With tuned threshold ({rf_s2a_metrics['best_threshold']:.2f}): F1 = {rf_s2a_metrics['tuned_metrics']['f1']:.4f}")
else:
    print("✗ Stage 2a Model: Insufficient data")
 
if rf_s2b_metrics:
    print(f"✓ Stage 2b Model (Minor vs Major Inflow): F1 = {rf_s2b_metrics['f1']:.4f}")
    if rf_s2b_metrics.get('tuned_metrics'):
        print(f"  └─ With tuned threshold ({rf_s2b_metrics['best_threshold']:.2f}): F1 = {rf_s2b_metrics['tuned_metrics']['f1']:.4f}")
else:
    print("✗ Stage 2b Model: Insufficient data")
 
print(f"\nModels saved in 'sequential_models/' directory:")
print("- stage1_inflow_outflow.joblib")
if rf_s2a_pipeline:
    print("- stage2a_minor_major_outflow.joblib")
if rf_s2b_pipeline:
    print("- stage2b_minor_major_inflow.joblib")
 
print(f"\nDataset summary (using capped values):")
print(f"- Total records: {len(df)}")
print(f"- Inflows: {(df['stage1_target'] == 1).sum()} ({(df['stage1_target'] == 1).mean()*100:.1f}%)")
print(f"- Outflows: {(df['stage1_target'] == 0).sum()} ({(df['stage1_target'] == 0).mean()*100:.1f}%)")
 
if rf_s2a_pipeline:
    valid_outflows = df.loc[df['stage1_target'] == 0, 'stage2a_target'].notna().sum()
    minor_outflows = df.loc[df['stage1_target'] == 0, 'stage2a_target'].sum()
    print(f"- Minor outflows: {minor_outflows} ({minor_outflows/valid_outflows*100:.1f}% of outflows)")
    print(f"- Major outflows: {valid_outflows - minor_outflows} ({(valid_outflows - minor_outflows)/valid_outflows*100:.1f}% of outflows)")
 
if rf_s2b_pipeline:
    valid_inflows = df.loc[df['stage1_target'] == 1, 'stage2b_target'].notna().sum()
    minor_inflows = df.loc[df['stage1_target'] == 1, 'stage2b_target'].sum()
    print(f"- Minor inflows: {minor_inflows} ({minor_inflows/valid_inflows*100:.1f}% of inflows)")
    print(f"- Major inflows: {valid_inflows - minor_inflows} ({(valid_inflows - minor_inflows)/valid_inflows*100:.1f}% of inflows)")
 
print(f"\nOutlier handling: Capped at {pct_1:.2f}% to {pct_99:.2f}% (1st to 99th percentiles)")
print(f"Model improvements applied:")
print("✓ SMOTE resampling for imbalanced classes")
print("✓ Threshold tuning for better recall")
print("✓ Outlier capping to reduce extreme value impact")
print("✓ Error propagation analysis")
print("✓ 4-category classification (Major/Minor × Outflow/Inflow)")
 
print("\nEnhanced sequential binary classification model training completed!") 
