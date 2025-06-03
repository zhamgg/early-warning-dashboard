import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Fund Outflow Early Warning System",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warning-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warning-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .performance-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sequential-stage {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #333;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'sequential_models_trained' not in st.session_state:
    st.session_state.sequential_models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Auto-training functions
@st.cache_resource
def train_binary_model(df):
    """Train binary classification model if not found"""
    try:
        # Create target variable
        df_model = df.copy()
        df_model['target'] = (df_model['Net Cash Flows'] < -1000000).astype(int)
        
        # Remove target leakage columns
        leak_columns = [
            "Cash Outflows", "Cash Inflows", "Net Cash Flows",
            "Net Market & Other", "Net Change AUM", "Ending Net Assets"
        ]
        columns_to_remove = ['target'] + leak_columns
        available_columns_to_remove = [col for col in columns_to_remove if col in df_model.columns]
        
        # Prepare features and target
        X = df_model.drop(columns=available_columns_to_remove, axis=1)
        y = df_model['target']
        
        # Identify column types
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        model_pipeline.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model_pipeline, 'EarlyWarningWeights.joblib')
        
        return model_pipeline, "Binary model trained successfully!"
        
    except Exception as e:
        return None, f"Error training binary model: {e}"

@st.cache_resource
def train_sequential_models(df):
    """Train sequential classification models if not found"""
    try:
        # Compute percentage change in AUM
        df_seq = df.copy()
        df_seq["Pct Change in AUM"] = (df_seq["Net Cash Flows"] / df_seq["Beginning Net Assets"]) * 100.0
        df_seq = df_seq.replace([np.inf, -np.inf], np.nan)
        df_seq = df_seq.dropna(subset=["Pct Change in AUM", "Beginning Net Assets"]).reset_index(drop=True)
        
        # Cap extreme outliers
        pct_99 = df_seq["Pct Change in AUM"].quantile(0.99)
        pct_1 = df_seq["Pct Change in AUM"].quantile(0.01)
        df_seq["Pct Change in AUM_capped"] = df_seq["Pct Change in AUM"].clip(lower=pct_1, upper=pct_99)
        
        # Create sequential targets
        df_seq['stage1_target'] = (df_seq["Pct Change in AUM_capped"] >= 0).astype(int)
        
        # Stage 2a: For outflows only
        outflow_mask = df_seq["Pct Change in AUM_capped"] < 0
        df_seq['stage2a_target'] = np.nan
        df_seq.loc[outflow_mask, 'stage2a_target'] = (df_seq.loc[outflow_mask, "Pct Change in AUM_capped"] >= -10).astype(int)
        
        # Stage 2b: For inflows only
        inflow_mask = df_seq["Pct Change in AUM_capped"] >= 0
        df_seq['stage2b_target'] = np.nan
        df_seq.loc[inflow_mask, 'stage2b_target'] = (df_seq.loc[inflow_mask, "Pct Change in AUM_capped"] < 10).astype(int)
        
        # Prepare features
        leak_cols = [
            "Cash Outflows", "Cash Inflows", "Net Cash Flows",
            "Net Market & Other", "Net Change AUM", "Ending Net Assets",
            "Pct Change in AUM", "Pct Change in AUM_capped", 
            "stage1_target", "stage2a_target", "stage2b_target"
        ]
        
        feature_cols = [c for c in df_seq.columns if c not in leak_cols]
        X = df_seq[feature_cols].copy()
        
        # Identify column types
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        # Create preprocessing pipeline
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
        
        # Create models directory
        os.makedirs("sequential_models", exist_ok=True)
        
        # Train Stage 1 Model
        y_stage1 = df_seq['stage1_target']
        X_train_s1, X_test_s1, y_train_s1, y_test_s1 = train_test_split(
            X, y_stage1, test_size=0.2, random_state=42, stratify=y_stage1
        )
        
        stage1_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=3)),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            ))
        ])
        
        stage1_pipeline.fit(X_train_s1, y_train_s1)
        joblib.dump(stage1_pipeline, "sequential_models/stage1_inflow_outflow.joblib")
        
        # Train Stage 2a Model (Outflows)
        stage2a_pipeline = None
        outflow_data = df_seq[df_seq['stage1_target'] == 0].copy()
        if len(outflow_data) > 20:
            X_outflows = X.loc[outflow_data.index]
            y_stage2a = outflow_data['stage2a_target'].dropna()
            X_outflows = X_outflows.loc[y_stage2a.index]
            
            if len(X_outflows) > 10:
                X_train_s2a, X_test_s2a, y_train_s2a, y_test_s2a = train_test_split(
                    X_outflows, y_stage2a, test_size=0.2, random_state=42, stratify=y_stage2a
                )
                
                stage2a_pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=42, k_neighbors=3)),
                    ('classifier', RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced"
                    ))
                ])
                
                stage2a_pipeline.fit(X_train_s2a, y_train_s2a)
                joblib.dump(stage2a_pipeline, "sequential_models/stage2a_minor_major_outflow.joblib")
        
        # Train Stage 2b Model (Inflows)
        stage2b_pipeline = None
        inflow_data = df_seq[df_seq['stage1_target'] == 1].copy()
        if len(inflow_data) > 20:
            X_inflows = X.loc[inflow_data.index]
            y_stage2b = inflow_data['stage2b_target'].dropna()
            X_inflows = X_inflows.loc[y_stage2b.index]
            
            if len(X_inflows) > 10:
                X_train_s2b, X_test_s2b, y_train_s2b, y_test_s2b = train_test_split(
                    X_inflows, y_stage2b, test_size=0.2, random_state=42, stratify=y_stage2b
                )
                
                stage2b_pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=42, k_neighbors=3)),
                    ('classifier', RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced"
                    ))
                ])
                
                stage2b_pipeline.fit(X_train_s2b, y_train_s2b)
                joblib.dump(stage2b_pipeline, "sequential_models/stage2b_minor_major_inflow.joblib")
        
        return stage1_pipeline, stage2a_pipeline, stage2b_pipeline, "Sequential models trained successfully!"
        
    except Exception as e:
        return None, None, None, f"Error training sequential models: {e}"

def check_and_load_models(df=None):
    """Check if models exist, train if not, then load them"""
    binary_model = None
    stage1_model = None
    stage2a_model = None
    stage2b_model = None
    messages = []
    
    # Check and handle binary model
    if os.path.exists('EarlyWarningWeights.joblib'):
        try:
            binary_model = joblib.load('EarlyWarningWeights.joblib')
            messages.append("✅ Binary model loaded from file")
        except Exception as e:
            messages.append(f"❌ Error loading binary model: {e}")
    else:
        if df is not None and 'Net Cash Flows' in df.columns:
            with st.spinner("Training binary model (this may take a few minutes)..."):
                binary_model, msg = train_binary_model(df)
                messages.append(f"🔄 {msg}")
        else:
            messages.append("⚠️ Binary model not found and cannot train without data")
    
    # Check and handle sequential models
    stage1_exists = os.path.exists('sequential_models/stage1_inflow_outflow.joblib')
    stage2a_exists = os.path.exists('sequential_models/stage2a_minor_major_outflow.joblib')
    stage2b_exists = os.path.exists('sequential_models/stage2b_minor_major_inflow.joblib')
    
    if stage1_exists:
        try:
            stage1_model = joblib.load('sequential_models/stage1_inflow_outflow.joblib')
            messages.append("✅ Stage 1 model loaded from file")
        except Exception as e:
            messages.append(f"❌ Error loading stage 1 model: {e}")
    
    if stage2a_exists:
        try:
            stage2a_model = joblib.load('sequential_models/stage2a_minor_major_outflow.joblib')
            messages.append("✅ Stage 2a model loaded from file")
        except Exception as e:
            messages.append(f"❌ Error loading stage 2a model: {e}")
    
    if stage2b_exists:
        try:
            stage2b_model = joblib.load('sequential_models/stage2b_minor_major_inflow.joblib')
            messages.append("✅ Stage 2b model loaded from file")
        except Exception as e:
            messages.append(f"❌ Error loading stage 2b model: {e}")
    
    # Train sequential models if any are missing and we have data
    if not (stage1_exists and stage2a_exists and stage2b_exists):
        if df is not None and 'Net Cash Flows' in df.columns and 'Beginning Net Assets' in df.columns:
            with st.spinner("Training sequential models (this may take several minutes)..."):
                s1, s2a, s2b, msg = train_sequential_models(df)
                if s1 is not None:
                    stage1_model = s1
                if s2a is not None:
                    stage2a_model = s2a
                if s2b is not None:
                    stage2b_model = s2b
                messages.append(f"🔄 {msg}")
        else:
            messages.append("⚠️ Sequential models not found and cannot train without data")
    
    return binary_model, stage1_model, stage2a_model, stage2b_model, messages

# Main title
st.markdown('<h1 class="main-header">🚨 Fund Outflow Early Warning System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Overview", "Data Analysis", "Fund Risk Assessment", "Early Warning Dashboard"]
)

# File uploader in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload fund data (Excel file)",
    type=['xlsx', 'xls'],
    help="Upload the joined_fund_data2.xlsx file"
)

# Load data function
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            # Try to load from default path
            df = pd.read_excel("joined_fund_data2.xlsx")
        
        # Clean data for Streamlit compatibility
        df = clean_dataframe_for_streamlit(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to clean dataframe for Streamlit compatibility
def clean_dataframe_for_streamlit(df):
    """Clean dataframe to avoid Arrow serialization issues"""
    df_clean = df.copy()
    
    # Convert object columns that contain mixed types
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try to convert to numeric first
            numeric_converted = pd.to_numeric(df_clean[col], errors='ignore')
            if numeric_converted.dtype != 'object':
                df_clean[col] = numeric_converted
            else:
                # Convert to string to avoid mixed types
                df_clean[col] = df_clean[col].astype(str)
    
    # Handle any remaining problematic columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Replace NaN with empty string for object columns
            df_clean[col] = df_clean[col].fillna('')
    
    # Convert any remaining float columns with NaN to handle properly
    for col in df_clean.select_dtypes(include=['float64']).columns:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(0.0)
    
    return df_clean

# Sequential model functions
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

def sequential_predict_4cat(X_data, stage1_model, stage2a_model=None, stage2b_model=None):
    """
    Make sequential predictions for 4 categories:
    Returns combined_pred: 0=major outflow, 1=minor outflow, 2=minor inflow, 3=major inflow
    """
    # Stage 1 predictions
    stage1_pred = stage1_model.predict(X_data)
    
    # Initialize stage2 predictions
    stage2a_pred = np.full(len(X_data), np.nan)
    stage2b_pred = np.full(len(X_data), np.nan)
    
    # Stage 2a predictions (only for predicted outflows)
    if stage2a_model is not None:
        outflow_indices = np.where(stage1_pred == 0)[0]
        if len(outflow_indices) > 0:
            X_outflows = X_data.iloc[outflow_indices]
            stage2a_pred_subset = stage2a_model.predict(X_outflows)
            stage2a_pred[outflow_indices] = stage2a_pred_subset
    
    # Stage 2b predictions (only for predicted inflows)
    if stage2b_model is not None:
        inflow_indices = np.where(stage1_pred == 1)[0]
        if len(inflow_indices) > 0:
            X_inflows = X_data.iloc[inflow_indices]
            stage2b_pred_subset = stage2b_model.predict(X_inflows)
            stage2b_pred[inflow_indices] = stage2b_pred_subset
    
    # Create combined prediction
    combined_pred = np.full(len(X_data), -1)
    
    # Handle outflows
    outflow_mask = stage1_pred == 0
    if stage2a_model is not None:
        combined_pred[outflow_mask] = stage2a_pred[outflow_mask]
    else:
        combined_pred[outflow_mask] = 1
    
    # Handle inflows  
    inflow_mask = stage1_pred == 1
    if stage2b_model is not None:
        stage2b_pred_mapped = np.where(stage2b_pred == 1, 2, 3)
        combined_pred[inflow_mask] = stage2b_pred_mapped[inflow_mask]
    else:
        combined_pred[inflow_mask] = 2
    
    return stage1_pred, stage2a_pred, stage2b_pred, combined_pred

# Load data
if uploaded_file is not None or os.path.exists("joined_fund_data2.xlsx"):
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Check and load/train models automatically
                st.sidebar.markdown("---")
                st.sidebar.subheader("🤖 Model Status")
                with st.sidebar:
                    binary_model, stage1_model, stage2a_model, stage2b_model, messages = check_and_load_models(df)
                    
                    # Store models in session state
                    st.session_state.binary_model = binary_model
                    st.session_state.stage1_model = stage1_model
                    st.session_state.stage2a_model = stage2a_model
                    st.session_state.stage2b_model = stage2b_model
                    
                    # Display model status messages
                    for message in messages:
                        if "✅" in message:
                            st.success(message)
                        elif "🔄" in message:
                            st.info(message)
                        elif "❌" in message:
                            st.error(message)
                        elif "⚠️" in message:
                            st.warning(message)
            else:
                st.sidebar.error("❌ Failed to load data")
    else:
        # Data already loaded, just show model status
        st.sidebar.success(f"✅ Data loaded! Shape: {st.session_state.df.shape}")
        
        # Check if models are already in session state
        if 'binary_model' not in st.session_state:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🤖 Model Status")
            with st.sidebar:
                binary_model, stage1_model, stage2a_model, stage2b_model, messages = check_and_load_models(st.session_state.df)
                
                # Store models in session state
                st.session_state.binary_model = binary_model
                st.session_state.stage1_model = stage1_model
                st.session_state.stage2a_model = stage2a_model
                st.session_state.stage2b_model = stage2b_model
                
                # Display model status messages
                for message in messages:
                    if "✅" in message:
                        st.success(message)
                    elif "🔄" in message:
                        st.info(message)
                    elif "❌" in message:
                        st.error(message)
                    elif "⚠️" in message:
                        st.warning(message)
        else:
            # Models already loaded, just show status
            st.sidebar.markdown("---")
            st.sidebar.subheader("🤖 Model Status")
            
            if st.session_state.binary_model is not None:
                st.sidebar.success("✅ Binary model ready")
            else:
                st.sidebar.error("❌ Binary model unavailable")
                
            if st.session_state.stage1_model is not None:
                st.sidebar.success("✅ Sequential models ready")
            else:
                st.sidebar.error("❌ Sequential models unavailable")
else:
    st.sidebar.warning("⚠️ Please upload the fund data file")
    # Initialize empty models in session state
    if 'binary_model' not in st.session_state:
        st.session_state.binary_model = None
        st.session_state.stage1_model = None
        st.session_state.stage2a_model = None
        st.session_state.stage2b_model = None

# Page routing
if page == "Overview":
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Objective")
        st.write("""
        **Goal**: Create early warning systems to identify funds at risk of outflows using:
        - Morningstar performance data
        - Great Gray flow data
        - Machine learning classification models
        """)
        
        st.subheader("🔬 Model Approaches")
        st.write("""
        **1. Binary Classification**: Predicts whether a fund will experience outflows > $1M
        
        **2. Sequential Classification**: Multi-stage prediction system:
        - Stage 1: Inflow vs Outflow
        - Stage 2a: Minor vs Major Outflow (for outflows)
        - Stage 2b: Minor vs Major Inflow (for inflows)
        """)
    
    with col2:
        st.subheader("📊 Model Performance")
        
        # Binary model performance
        st.markdown("""
        <div class="performance-metric">
            <h4>🏆 Binary Classification Model</h4>
            <p><strong>Accuracy:</strong> 84.25%</p>
            <p><strong>Precision:</strong> 49.5%</p>
            <p><strong>Recall:</strong> 42.7%</p>
            <p><strong>F1 Score:</strong> 0.4584</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sequential model performance
        st.markdown("""
        <div class="sequential-stage">
            <h4>🔄 Sequential Classification Model</h4>
            <p><strong>Overall 4-Class Accuracy:</strong> 73.78%</p>
            <p><strong>Stage 1 F1:</strong> 80.88% (tuned)</p>
            <p><strong>Stage 2a F1:</strong> 95.85% (tuned)</p>
            <p><strong>Stage 2b F1:</strong> 87.72% (tuned)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed Performance Breakdown
    st.subheader("📈 Detailed Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Binary Classification Results**")
        st.markdown("""
        - **Target**: Large outflows (>$1,000,000)
        - **Algorithm**: Random Forest with balanced classes
        - **Evaluation**: 80/20 train-test split
        - **Strength**: Good for high-level risk screening
        - **Use Case**: Flag funds for immediate attention
        """)
        
        # Binary metrics table
        binary_metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Score': ['84.25%', '49.5%', '42.7%', '0.4584']
        })
        st.table(binary_metrics)
    
    with col2:
        st.markdown("**🔄 Sequential Classification Results**")
        st.markdown("""
        - **Target**: 4-category flow classification
        - **Algorithm**: Multi-stage Random Forest + SMOTE
        - **Categories**: Major/Minor × Outflow/Inflow
        - **Strength**: Granular risk categorization
        - **Use Case**: Nuanced decision support
        """)
        
        # Sequential metrics table
        sequential_metrics = pd.DataFrame({
            'Stage': ['Stage 1 (Flow Direction)', 'Stage 2a (Outflow Severity)', 'Stage 2b (Inflow Magnitude)', 'Combined 4-Class'],
            'F1 Score': ['80.88%', '95.85%', '87.72%', 'N/A'],
            'ROC AUC': ['83.49%', '75.23%', '81.51%', 'N/A'],
            'Accuracy': ['76.54%', '88.42%', '78.63%', '73.78%']
        })
        st.table(sequential_metrics)
    
    # Key insights
    st.subheader("🔍 Model Comparison & Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Binary Classification**
        - Simple threshold-based prediction
        - 84.25% overall accuracy
        - 49.5% precision (low false positives)
        - Good for binary go/no-go decisions
        - Single decision boundary
        """)
    
    with col2:
        st.markdown("""
        **🔄 Sequential Classification**
        - Multi-stage decision process
        - 95.85% F1 for outflow severity detection
        - 87.72% F1 for inflow magnitude
        - Excellent granular categorization
        - Better handles flow magnitude differences
        """)
    
    with col3:
        st.markdown("""
        **📊 Business Impact**
        - **Binary**: Quick risk screening
        - **Sequential**: Detailed action guidance
        - **Combined**: Comprehensive risk assessment
        - **ROI**: Proactive fund management
        - **Efficiency**: Automated early warning
        """)
    
    # Data insights
    st.subheader("📋 Dataset Summary")
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", "14,872")
        with col2:
            st.metric("Features Used", "19")
        with col3:
            st.metric("Inflow Rate", "58.2%")
        with col4:
            st.metric("Outflow Rate", "41.8%")
        
        # Flow distribution insights
        st.markdown("**Flow Distribution (After Outlier Capping):**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background-color: #8B0000; color: white; padding: 0.5rem; border-radius: 0.5rem; text-align: center;">
                <h4>Major Outflow</h4>
                <h3>485</h3>
                <p>7.8% of outflows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #FF6347; color: white; padding: 0.5rem; border-radius: 0.5rem; text-align: center;">
                <h4>Minor Outflow</h4>
                <h3>5,732</h3>
                <p>92.2% of outflows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background-color: #90EE90; color: black; padding: 0.5rem; border-radius: 0.5rem; text-align: center;">
                <h4>Minor Inflow</h4>
                <h3>6,375</h3>
                <p>73.7% of inflows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background-color: #006400; color: white; padding: 0.5rem; border-radius: 0.5rem; text-align: center;">
                <h4>Major Inflow</h4>
                <h3>2,280</h3>
                <p>26.3% of inflows</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick data summary
        if 'Net Cash Flows' in df.columns:
            st.markdown("---")
            st.markdown("**Current Dataset Overview:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Funds", len(df))
            with col2:
                large_outflows = (df['Net Cash Flows'] < -1000000).sum()
                st.metric("Large Outflows (>$1M)", large_outflows)
            with col3:
                outflow_rate = large_outflows / len(df) * 100
                st.metric("Current Outflow Rate", f"{outflow_rate:.1f}%")
            with col4:
                avg_flow = df['Net Cash Flows'].mean()
                st.metric("Avg Net Cash Flow", f"${avg_flow:,.0f}")
    else:
        st.warning("⚠️ Please upload data to see current dataset metrics")
    
    # Model availability status
    st.subheader("🤖 Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'binary_model' in st.session_state and st.session_state.binary_model is not None:
            st.success("✅ Binary Classification Model Available")
        elif os.path.exists('EarlyWarningWeights.joblib'):
            st.info("📁 Binary Model File Found (will load with data)")
        else:
            st.warning("⚠️ Binary Model Will Be Trained When Data Is Loaded")
    
    with col2:
        if 'stage1_model' in st.session_state and st.session_state.stage1_model is not None:
            st.success("✅ Sequential Classification Models Available")
        else:
            stage1_exists = os.path.exists('sequential_models/stage1_inflow_outflow.joblib')
            stage2a_exists = os.path.exists('sequential_models/stage2a_minor_major_outflow.joblib')
            stage2b_exists = os.path.exists('sequential_models/stage2b_minor_major_inflow.joblib')
            
            if stage1_exists and stage2a_exists and stage2b_exists:
                st.info("📁 Sequential Model Files Found (will load with data)")
            elif stage1_exists:
                st.warning("⚠️ Sequential Models Partially Available")
            else:
                st.warning("⚠️ Sequential Models Will Be Trained When Data Is Loaded")
    
    # Feature importance summary
    st.subheader("🔍 Key Predictive Features")
    
    st.markdown("""
    **Top Features Across All Models:**
    1. **As-of Date** - Temporal patterns in fund flows
    2. **Beginning Net Assets** - Fund size correlation with flow patterns
    3. **Quarter** - Seasonal flow trends
    4. **CUSIP & Fund ID** - Fund-specific characteristics
    5. **IR_1Y & IR_3Y** - Information ratios indicating performance
    6. **Fund Name** - Fund family/strategy effects
    7. **Percentile Rankings** - Relative performance measures
    """)

elif page == "Data Analysis":
    st.header("🔍 Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first")
    else:
        df = st.session_state.df
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Funds", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            if 'Net Cash Flows' in df.columns:
                st.metric("Avg Cash Flow", f"${df['Net Cash Flows'].mean():,.0f}")
        with col4:
            if 'Net Cash Flows' in df.columns:
                outflows = (df['Net Cash Flows'] < -1000000).sum()
                st.metric("Large Outflows (>$1M)", outflows)
        
        # Target variable analysis for both models
        if 'Net Cash Flows' in df.columns and 'Beginning Net Assets' in df.columns:
            # Compute percentage change for sequential model
            df_analysis = df.copy()
            df_analysis["Pct Change in AUM"] = (df_analysis["Net Cash Flows"] / df_analysis["Beginning Net Assets"]) * 100.0
            df_analysis = df_analysis.replace([np.inf, -np.inf], np.nan)
            df_analysis = df_analysis.dropna(subset=["Pct Change in AUM"]).reset_index(drop=True)
            
            # Cap extreme outliers
            pct_99 = df_analysis["Pct Change in AUM"].quantile(0.99)
            pct_1 = df_analysis["Pct Change in AUM"].quantile(0.01)
            df_analysis["Pct Change in AUM_capped"] = df_analysis["Pct Change in AUM"].clip(lower=pct_1, upper=pct_99)
            
            st.subheader("🎯 Target Variable Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Binary Classification Target**")
                # Binary target
                binary_target = (df['Net Cash Flows'] < -1000000).astype(int)
                binary_counts = binary_target.value_counts()
                
                fig = px.pie(
                    values=binary_counts.values,
                    names=['No Large Outflow', 'Large Outflow (>$1M)'],
                    title="Binary Target Distribution",
                    color_discrete_sequence=['#2E8B57', '#DC143C']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Sequential Classification Target**")
                # Sequential 4-category target
                seq_target = create_combined_labels_4cat(df_analysis["Pct Change in AUM_capped"])
                seq_counts = pd.Series(seq_target).value_counts().sort_index()
                
                fig = px.pie(
                    values=seq_counts.values,
                    names=['Major Outflow', 'Minor Outflow', 'Minor Inflow', 'Major Inflow'],
                    title="4-Category Target Distribution",
                    color_discrete_sequence=['#8B0000', '#FF6347', '#90EE90', '#006400']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribution comparison
            st.subheader("📊 Flow Distribution Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Absolute cash flows
                fig = px.histogram(
                    df, 
                    x='Net Cash Flows', 
                    nbins=50,
                    title="Net Cash Flows Distribution (Absolute $)",
                    color_discrete_sequence=['#1f77b4']
                )
                fig.add_vline(x=-1000000, line_dash="dash", line_color="red", 
                             annotation_text="Binary Threshold (-$1M)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Percentage change
                fig = px.histogram(
                    df_analysis, 
                    x='Pct Change in AUM_capped', 
                    nbins=50,
                    title="Percentage Change in AUM Distribution",
                    color_discrete_sequence=['#ff7f0e']
                )
                fig.add_vline(x=-10, line_dash="dash", line_color="red", annotation_text="Major Outflow (-10%)")
                fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Inflow/Outflow (0%)")
                fig.add_vline(x=10, line_dash="dash", line_color="green", annotation_text="Major Inflow (+10%)")
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Binary - No Large Outflow", binary_counts.get(0, 0))
            with col2:
                st.metric("Binary - Large Outflow", binary_counts.get(1, 0))
            with col3:
                imbalance_ratio = binary_counts.get(0, 0) / binary_counts.get(1, 1)
                st.metric("Binary Imbalance Ratio", f"{imbalance_ratio:.1f}:1")
            with col4:
                positive_rate = binary_counts.get(1, 0) / len(df) * 100
                st.metric("Binary Positive Rate", f"{positive_rate:.1f}%")
            
            # Sequential model statistics
            st.markdown("**Sequential Model Class Distribution:**")
            col1, col2, col3, col4 = st.columns(4)
            class_names = ['Major Outflow', 'Minor Outflow', 'Minor Inflow', 'Major Inflow']
            for i, (col, name) in enumerate(zip([col1, col2, col3, col4], class_names)):
                count = seq_counts.get(i, 0)
                pct = count / len(seq_target) * 100
                with col:
                    st.metric(name, f"{count} ({pct:.1f}%)")
        
        # Data preview and column info (existing code)
        st.subheader("📋 Data Preview")
        try:
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying data preview: {e}")
            st.write("Data preview unavailable. Showing basic info:")
            st.write(f"Shape: {df.shape}")
            st.write(f"Columns: {list(df.columns)}")

elif page == "Fund Risk Assessment":
    st.header("🔍 Fund Risk Assessment")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first")
    else:
        df = st.session_state.df
        
        # Model selection
        st.subheader("🤖 Select Model Type")
        model_type = st.radio(
            "Choose assessment model:",
            ["Binary Classification", "Sequential Classification"],
            help="Binary: Simple outflow >$1M prediction. Sequential: 4-category granular prediction."
        )
        
        if model_type == "Binary Classification":
            if st.session_state.binary_model is None:
                st.warning("⚠️ Binary model not available. Please ensure data is loaded to enable automatic training.")
            else:
                try:
                    # Use binary model from session state
                    binary_model = st.session_state.binary_model
                    
                    # Fund selection
                    st.subheader("Select a Fund for Binary Risk Assessment")
                    
                    if 'Fund Name_x' in df.columns:
                        fund_options = df['Fund Name_x'].dropna().unique()
                        selected_fund = st.selectbox("Choose a fund:", fund_options, key="binary_fund_select")
                        fund_data = df[df['Fund Name_x'] == selected_fund].iloc[0:1]
                        fund_name = selected_fund
                    else:
                        fund_idx = st.number_input("Select fund index:", 0, len(df)-1, 0, key="binary_fund_idx")
                        fund_data = df.iloc[fund_idx:fund_idx+1]
                        fund_name = f"Fund #{fund_idx}"
                    
                    # Prepare features
                    leak_columns = [
                        "Cash Outflows", "Cash Inflows", "Net Cash Flows",
                        "Net Market & Other", "Net Change AUM", "Ending Net Assets"
                    ]
                    fund_features = fund_data.drop(leak_columns, axis=1, errors='ignore')
                    
                    # Make prediction
                    risk_probability = binary_model.predict_proba(fund_features)[0][1]
                    risk_prediction = binary_model.predict(fund_features)[0]
                    
                    # Display results
                    st.subheader(f"Binary Risk Assessment: {fund_name}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = risk_probability * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Large Outflow Risk (%)"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50}}))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Risk classification
                        if risk_prediction == 1:
                            st.markdown(f"""
                            <div class="warning-high">
                                <h3>⚠️ HIGH RISK ALERT</h3>
                                <p><strong>Prediction:</strong> Large outflow likely (>$1M)</p>
                                <p><strong>Confidence:</strong> {risk_probability:.1%}</p>
                                <p><strong>Recommendation:</strong> Immediate attention required</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            if risk_probability > 0.3:
                                st.markdown(f"""
                                <div class="warning-medium">
                                    <h3>🔶 MODERATE RISK</h3>
                                    <p><strong>Prediction:</strong> No large outflow expected</p>
                                    <p><strong>Confidence:</strong> {risk_probability:.1%}</p>
                                    <p><strong>Recommendation:</strong> Monitor closely</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Display actual outcome if available
                    if 'Net Cash Flows' in fund_data.columns:
                        actual_flow = fund_data['Net Cash Flows'].iloc[0]
                        actual_large_outflow = actual_flow < -1000000
                        
                        st.subheader("📊 Actual vs Predicted")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Actual Net Cash Flow", f"${actual_flow:,.0f}")
                        with col2:
                            st.metric("Actual Large Outflow", "Yes" if actual_large_outflow else "No")
                        with col3:
                            if actual_large_outflow == risk_prediction:
                                st.metric("Prediction Accuracy", "✅ Correct")
                            else:
                                st.metric("Prediction Accuracy", "❌ Incorrect")
                
                except Exception as e:
                    st.error(f"❌ Error with binary model assessment: {e}")
        
        else:  # Sequential Classification
            # Check if sequential models exist in session state
            if st.session_state.stage1_model is None:
                st.warning("⚠️ Sequential models not available. Please ensure data is loaded to enable automatic training.")
            else:
                try:
                    # Use sequential models from session state
                    stage1_model = st.session_state.stage1_model
                    stage2a_model = st.session_state.stage2a_model
                    stage2b_model = st.session_state.stage2b_model
                    
                    # Fund selection
                    st.subheader("Select a Fund for Sequential Risk Assessment")
                    
                    if 'Fund Name_x' in df.columns:
                        fund_options = df['Fund Name_x'].dropna().unique()
                        selected_fund = st.selectbox("Choose a fund:", fund_options, key="seq_fund_select")
                        fund_data = df[df['Fund Name_x'] == selected_fund].iloc[0:1]
                        fund_name = selected_fund
                    else:
                        fund_idx = st.number_input("Select fund index:", 0, len(df)-1, 0, key="seq_fund_idx")
                        fund_data = df.iloc[fund_idx:fund_idx+1]
                        fund_name = f"Fund #{fund_idx}"
                    
                    # Prepare features
                    leak_columns = [
                        "Cash Outflows", "Cash Inflows", "Net Cash Flows",
                        "Net Market & Other", "Net Change AUM", "Ending Net Assets"
                    ]
                    fund_features = fund_data.drop(leak_columns, axis=1, errors='ignore')
                    
                    # Make sequential predictions
                    stage1_pred, stage2a_pred, stage2b_pred, combined_pred = sequential_predict_4cat(
                        fund_features, stage1_model, stage2a_model, stage2b_model
                    )
                    
                    combined_prediction = combined_pred[0]
                    stage1_prediction = stage1_pred[0]
                    
                    # Map predictions to labels
                    category_names = ['Major Outflow', 'Minor Outflow', 'Minor Inflow', 'Major Inflow']
                    category_colors = ['#8B0000', '#FF6347', '#90EE90', '#006400']
                    predicted_category = category_names[combined_prediction]
                    predicted_color = category_colors[combined_prediction]
                    
                    # Display results
                    st.subheader(f"Sequential Risk Assessment: {fund_name}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sequential prediction visualization
                        fig = go.Figure()
                        
                        # Add gauge for stage 1
                        fig.add_trace(go.Indicator(
                            mode = "gauge+number",
                            value = stage1_prediction,
                            title = {'text': "Stage 1: Flow Direction"},
                            domain = {'row': 0, 'column': 0},
                            gauge = {
                                'axis': {'range': [0, 1], 'tickvals': [0, 1], 'ticktext': ['Outflow', 'Inflow']},
                                'bar': {'color': "lightblue"},
                                'steps': [
                                    {'range': [0, 0.5], 'color': "#ffcccb"},
                                    {'range': [0.5, 1], 'color': "#90ee90"}]
                            }
                        ))
                        
                        fig.update_layout(
                            grid = {'rows': 1, 'columns': 1, 'pattern': "independent"},
                            height=250
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Final category display
                        st.markdown(f"""
                        <div style="background-color: {predicted_color}; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; margin: 1rem 0;">
                            <h3>📊 Final Prediction</h3>
                            <h2>{predicted_category}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Sequential decision tree visualization
                        st.markdown("**🔄 Sequential Decision Process:**")
                        
                        if stage1_prediction == 0:  # Outflow
                            st.markdown("1. **Stage 1**: Predicted Outflow")
                            if stage2a_model is not None and not np.isnan(stage2a_pred[0]):
                                stage2a_result = "Minor Outflow" if stage2a_pred[0] == 1 else "Major Outflow"
                                st.markdown(f"2. **Stage 2a**: {stage2a_result}")
                            else:
                                st.markdown("2. **Stage 2a**: Model not available")
                        else:  # Inflow
                            st.markdown("1. **Stage 1**: Predicted Inflow")
                            if stage2b_model is not None and not np.isnan(stage2b_pred[0]):
                                stage2b_result = "Minor Inflow" if stage2b_pred[0] == 1 else "Major Inflow"
                                st.markdown(f"2. **Stage 2b**: {stage2b_result}")
                            else:
                                st.markdown("2. **Stage 2b**: Model not available")
                        
                        # Recommendation based on category
                        recommendations = {
                            0: ("🚨 URGENT ACTION REQUIRED", "Major outflow risk - immediate intervention needed"),
                            1: ("⚠️ MONITOR CLOSELY", "Minor outflow risk - increased monitoring recommended"),
                            2: ("✅ STABLE", "Minor inflow expected - continue regular monitoring"),
                            3: ("🎉 EXCELLENT", "Major inflow expected - fund performing well")
                        }
                        
                        rec_title, rec_desc = recommendations[combined_prediction]
                        
                        if combined_prediction <= 1:  # Outflows
                            alert_class = "warning-high" if combined_prediction == 0 else "warning-medium"
                        else:  # Inflows
                            alert_class = "warning-low"
                        
                        st.markdown(f"""
                        <div class="{alert_class}">
                            <h4>{rec_title}</h4>
                            <p>{rec_desc}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Compare with actual if available
                    if 'Net Cash Flows' in fund_data.columns and 'Beginning Net Assets' in fund_data.columns:
                        actual_flow = fund_data['Net Cash Flows'].iloc[0]
                        beginning_assets = fund_data['Beginning Net Assets'].iloc[0]
                        
                        if beginning_assets != 0:
                            actual_pct_change = (actual_flow / beginning_assets) * 100
                            actual_category = create_combined_labels_4cat(np.array([actual_pct_change]))[0]
                            actual_category_name = category_names[actual_category]
                            
                            st.subheader("📊 Actual vs Predicted Comparison")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Actual Cash Flow", f"${actual_flow:,.0f}")
                            with col2:
                                st.metric("Actual % Change", f"{actual_pct_change:.2f}%")
                            with col3:
                                st.metric("Actual Category", actual_category_name)
                            with col4:
                                if actual_category == combined_prediction:
                                    st.metric("Sequential Accuracy", "✅ Correct")
                                else:
                                    st.metric("Sequential Accuracy", "❌ Incorrect")
                
                except Exception as e:
                    st.error(f"❌ Error with sequential model assessment: {e}")
        
        # Fund details section (common for both models)
        st.subheader("📋 Fund Details")
        try:
            fund_data_clean = clean_dataframe_for_streamlit(fund_data)
            
            # Select key columns to display
            display_cols = []
            key_columns = [
                'Fund Name_x', 'Beginning Net Assets', 'Category_x', 
                'Total Return Month', 'Total Return 3 Month', 'Total Return YTD',
                'Total Return 1 Year', 'Total Return 3 Year', 'Total Return 5 Year'
            ]
            
            for col in key_columns:
                if col in fund_data_clean.columns:
                    display_cols.append(col)
            
            if display_cols:
                st.dataframe(fund_data_clean[display_cols].T, use_container_width=True)
            else:
                st.dataframe(fund_data_clean.T, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error displaying fund details: {e}")

elif page == "Early Warning Dashboard":
    st.header("🚨 Early Warning Dashboard")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first")
    else:
        df = st.session_state.df
        
        # Model selection for dashboard
        st.subheader("🤖 Select Dashboard Model")
        dashboard_model = st.radio(
            "Choose model for dashboard analysis:",
            ["Binary Classification", "Sequential Classification"],
            help="Select which model predictions to display in the dashboard"
        )
        
        if dashboard_model == "Binary Classification":
            # Binary model dashboard
            if st.session_state.binary_model is None:
                st.warning("⚠️ Binary model not available. Please ensure data is loaded to enable automatic training.")
            else:
                try:
                    model = st.session_state.binary_model
                    
                    # Prepare features (remove target leakage columns)
                    leak_columns = [
                        "Cash Outflows", "Cash Inflows", "Net Cash Flows",
                        "Net Market & Other", "Net Change AUM", "Ending Net Assets"
                    ]
                    X = df.drop(leak_columns, axis=1, errors='ignore')
                    
                    # Get predictions and probabilities
                    risk_probs = model.predict_proba(X)[:, 1]
                    risk_predictions = model.predict(X)
                    
                    # Create risk dashboard dataframe
                    df_risk = df.copy()
                    df_risk['Risk_Probability'] = risk_probs
                    df_risk['High_Risk_Flag'] = risk_predictions
                    df_risk['Actual_Large_Outflow'] = (df_risk['Net Cash Flows'] < -1000000).astype(int) if 'Net Cash Flows' in df_risk.columns else 0
                    
                    # Dashboard overview
                    st.subheader("📊 Binary Model Risk Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_funds = len(df_risk)
                    high_risk_funds = df_risk['High_Risk_Flag'].sum()
                    avg_risk = df_risk['Risk_Probability'].mean()
                    actual_outflows = df_risk['Actual_Large_Outflow'].sum() if 'Net Cash Flows' in df.columns else 0
                    
                    with col1:
                        st.metric("Total Funds", total_funds)
                    with col2:
                        st.metric("High Risk Flags", high_risk_funds)
                    with col3:
                        st.metric("Actual Large Outflows", actual_outflows)
                    with col4:
                        st.metric("Average Risk Score", f"{avg_risk:.1%}")
                    
                    # Risk threshold selector
                    st.subheader("🎯 Risk Threshold Analysis")
                    risk_threshold = st.slider(
                        "Risk Probability Threshold (%)", 
                        0, 100, 50, 5,
                        help="Adjust threshold to see how many funds would be flagged at different risk levels",
                        key="binary_threshold"
                    )
                    
                    threshold_decimal = risk_threshold / 100
                    funds_above_threshold = (df_risk['Risk_Probability'] > threshold_decimal).sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Funds Above Threshold", funds_above_threshold)
                    with col2:
                        pct_flagged = funds_above_threshold / total_funds * 100
                        st.metric("% of Funds Flagged", f"{pct_flagged:.1f}%")
                    with col3:
                        if actual_outflows > 0:
                            caught_outflows = ((df_risk['Risk_Probability'] > threshold_decimal) & 
                                             (df_risk['High_Risk_Flag'] == 1)).sum()
                            catch_rate = caught_outflows / actual_outflows * 100
                            st.metric("Outflows Caught", f"{catch_rate:.1f}%")
                    
                    # Risk distribution visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk probability histogram
                        fig = px.histogram(
                            df_risk, 
                            x='Risk_Probability',
                            nbins=20,
                            title="Distribution of Risk Probabilities",
                            labels={'Risk_Probability': 'Risk Probability', 'count': 'Number of Funds'}
                        )
                        fig.add_vline(x=threshold_decimal, line_dash="dash", line_color="red", 
                                     annotation_text=f"Threshold ({risk_threshold}%)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'Net Cash Flows' in df.columns:
                            # Actual vs Predicted confusion matrix
                            confusion_data = pd.crosstab(
                                df_risk['Actual_Large_Outflow'], 
                                df_risk['High_Risk_Flag'],
                                rownames=['Actual'], 
                                colnames=['Predicted']
                            )
                            
                            fig = px.imshow(
                                confusion_data.values,
                                text_auto=True,
                                title="Model Performance: Actual vs Predicted",
                                labels=dict(x="Predicted", y="Actual"),
                                x=['No Large Outflow', 'Large Outflow'],
                                y=['No Large Outflow', 'Large Outflow']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Actual outcomes not available for performance evaluation")
                    
                    # High-risk funds table
                    st.subheader(f"⚠️ High-Risk Funds (>{risk_threshold}% probability)")
                    high_risk_df = df_risk[df_risk['Risk_Probability'] > threshold_decimal].copy()
                    high_risk_df = high_risk_df.sort_values('Risk_Probability', ascending=False)
                    
                    if len(high_risk_df) > 0:
                        # Select relevant columns for display
                        display_cols = ['Risk_Probability']
                        if 'Fund Name_x' in high_risk_df.columns:
                            display_cols = ['Fund Name_x'] + display_cols
                        if 'Net Cash Flows' in high_risk_df.columns:
                            display_cols = display_cols + ['Net Cash Flows']
                        if 'Beginning Net Assets' in high_risk_df.columns:
                            display_cols = display_cols + ['Beginning Net Assets']
                        if 'Category_x' in high_risk_df.columns:
                            display_cols = display_cols + ['Category_x']
                        
                        # Format the dataframe
                        display_df = high_risk_df[display_cols].copy()
                        display_df['Risk_Probability'] = display_df['Risk_Probability'].apply(lambda x: f"{x:.1%}")
                        
                        if 'Net Cash Flows' in display_df.columns:
                            display_df['Net Cash Flows'] = display_df['Net Cash Flows'].apply(lambda x: f"${x:,.0f}")
                        if 'Beginning Net Assets' in display_df.columns:
                            display_df['Beginning Net Assets'] = display_df['Beginning Net Assets'].apply(lambda x: f"${x:,.0f}")
                        
                        try:
                            display_df_clean = clean_dataframe_for_streamlit(display_df)
                            st.dataframe(display_df_clean, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying high-risk funds table: {e}")
                        
                        # Download button for high-risk funds
                        csv = high_risk_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download High-Risk Funds CSV",
                            data=csv,
                            file_name=f"binary_high_risk_funds_{risk_threshold}pct.csv",
                            mime="text/csv"
                        )
                    else:
                        st.success("✅ No funds exceed the selected risk threshold!")
                    
                    # Model performance metrics
                    if 'Net Cash Flows' in df.columns:
                        st.subheader("📈 Binary Model Performance Summary")
                        
                        # Calculate performance metrics
                        y_true = df_risk['Actual_Large_Outflow']
                        y_pred = df_risk['High_Risk_Flag']
                        
                        accuracy = accuracy_score(y_true, y_pred)
                        precision = precision_score(y_true, y_pred) if y_pred.sum() > 0 else 0
                        recall = recall_score(y_true, y_pred) if y_true.sum() > 0 else 0
                        f1 = f1_score(y_true, y_pred) if (y_pred.sum() > 0 and y_true.sum() > 0) else 0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.3f}")
                        with col2:
                            st.metric("Precision", f"{precision:.3f}")
                        with col3:
                            st.metric("Recall", f"{recall:.3f}")
                        with col4:
                            st.metric("F1 Score", f"{f1:.4f}")
                
                except Exception as e:
                    st.error(f"❌ Error creating binary risk dashboard: {e}")
        
        else:  # Sequential Classification Dashboard
            # Check if sequential models exist in session state
            if st.session_state.stage1_model is None:
                st.warning("⚠️ Sequential models not available. Please ensure data is loaded to enable automatic training.")
            else:
                try:
                    # Use sequential models from session state
                    stage1_model = st.session_state.stage1_model
                    stage2a_model = st.session_state.stage2a_model
                    stage2b_model = st.session_state.stage2b_model
                    
                    # Prepare features
                    leak_columns = [
                        "Cash Outflows", "Cash Inflows", "Net Cash Flows",
                        "Net Market & Other", "Net Change AUM", "Ending Net Assets"
                    ]
                    X = df.drop(leak_columns, axis=1, errors='ignore')
                    
                    # Make sequential predictions for all funds
                    stage1_pred, stage2a_pred, stage2b_pred, combined_pred = sequential_predict_4cat(
                        X, stage1_model, stage2a_model, stage2b_model
                    )
                    
                    # Create sequential dashboard dataframe
                    df_seq_dash = df.copy()
                    df_seq_dash['Stage1_Prediction'] = stage1_pred  # 0=outflow, 1=inflow
                    df_seq_dash['Combined_Prediction'] = combined_pred  # 0=major outflow, 1=minor outflow, 2=minor inflow, 3=major inflow
                    
                    # Create actual categories if possible
                    if 'Net Cash Flows' in df.columns and 'Beginning Net Assets' in df.columns:
                        df_seq_dash["Pct_Change_AUM"] = (df_seq_dash["Net Cash Flows"] / df_seq_dash["Beginning Net Assets"]) * 100.0
                        df_seq_dash = df_seq_dash.replace([np.inf, -np.inf], np.nan)
                        
                        # Cap outliers for comparison
                        pct_99 = df_seq_dash["Pct_Change_AUM"].quantile(0.99)
                        pct_1 = df_seq_dash["Pct_Change_AUM"].quantile(0.01)
                        df_seq_dash["Pct_Change_AUM_capped"] = df_seq_dash["Pct_Change_AUM"].clip(lower=pct_1, upper=pct_99)
                        
                        # Create actual combined labels
                        df_seq_dash['Actual_Combined'] = create_combined_labels_4cat(df_seq_dash["Pct_Change_AUM_capped"])
                    
                    # Dashboard overview
                    st.subheader("📊 Sequential Model Risk Overview")
                    
                    category_names = ['Major Outflow', 'Minor Outflow', 'Minor Inflow', 'Major Inflow']
                    category_colors = ['#8B0000', '#FF6347', '#90EE90', '#006400']
                    
                    # Count predictions by category
                    pred_counts = pd.Series(combined_pred).value_counts().sort_index()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    for i, (col, name, color) in enumerate(zip([col1, col2, col3, col4], category_names, category_colors)):
                        count = pred_counts.get(i, 0)
                        pct = count / len(df_seq_dash) * 100
                        with col:
                            st.markdown(f"""
                            <div style="background-color: {color}; color: white; padding: 0.5rem; border-radius: 0.5rem; text-align: center; margin: 0.2rem 0;">
                                <h4>{name}</h4>
                                <h3>{count}</h3>
                                <p>{pct:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Category distribution charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Predicted distribution
                        fig = px.pie(
                            values=pred_counts.values,
                            names=[category_names[i] for i in pred_counts.index],
                            title="Predicted Category Distribution",
                            color_discrete_sequence=category_colors
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'Actual_Combined' in df_seq_dash.columns:
                            # Actual vs Predicted comparison
                            actual_counts = pd.Series(df_seq_dash['Actual_Combined']).value_counts().sort_index()
                            
                            comparison_df = pd.DataFrame({
                                'Predicted': pred_counts,
                                'Actual': actual_counts
                            }).fillna(0)
                            comparison_df.index = [category_names[i] for i in comparison_df.index]
                            
                            fig = px.bar(
                                comparison_df,
                                title="Predicted vs Actual Distribution",
                                barmode='group',
                                color_discrete_sequence=['#1f77b4', '#ff7f0e']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Stage 1 distribution only
                            stage1_counts = pd.Series(stage1_pred).value_counts()
                            fig = px.pie(
                                values=stage1_counts.values,
                                names=['Outflow', 'Inflow'],
                                title="Stage 1: Flow Direction Distribution",
                                color_discrete_sequence=['#ff6b6b', '#4ecdc4']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk filtering
                    st.subheader("🎯 Risk Category Filter")
                    
                    selected_categories = st.multiselect(
                        "Select categories to highlight:",
                        category_names,
                        default=['Major Outflow'],
                        help="Choose which risk categories to focus on"
                    )
                    
                    # Filter funds by selected categories
                    if selected_categories:
                        selected_indices = [category_names.index(cat) for cat in selected_categories]
                        filtered_mask = df_seq_dash['Combined_Prediction'].isin(selected_indices)
                        filtered_funds = df_seq_dash[filtered_mask].copy()
                        
                        st.subheader(f"📋 Funds in Selected Categories ({len(filtered_funds)} funds)")
                        
                        if len(filtered_funds) > 0:
                            # Add category names to display
                            filtered_funds['Predicted_Category'] = filtered_funds['Combined_Prediction'].map(
                                lambda x: category_names[x]
                            )
                            
                            # Select relevant columns for display
                            display_cols = ['Predicted_Category']
                            if 'Fund Name_x' in filtered_funds.columns:
                                display_cols = ['Fund Name_x'] + display_cols
                            if 'Net Cash Flows' in filtered_funds.columns:
                                display_cols = display_cols + ['Net Cash Flows']
                            if 'Pct_Change_AUM' in filtered_funds.columns:
                                display_cols = display_cols + ['Pct_Change_AUM']
                            if 'Beginning Net Assets' in filtered_funds.columns:
                                display_cols = display_cols + ['Beginning Net Assets']
                            if 'Category_x' in filtered_funds.columns:
                                display_cols = display_cols + ['Category_x']
                            
                            # Format the dataframe
                            display_df = filtered_funds[display_cols].copy()
                            
                            if 'Net Cash Flows' in display_df.columns:
                                display_df['Net Cash Flows'] = display_df['Net Cash Flows'].apply(lambda x: f"${x:,.0f}")
                            if 'Pct_Change_AUM' in display_df.columns:
                                display_df['Pct_Change_AUM'] = display_df['Pct_Change_AUM'].apply(lambda x: f"{x:.2f}%")
                            if 'Beginning Net Assets' in display_df.columns:
                                display_df['Beginning Net Assets'] = display_df['Beginning Net Assets'].apply(lambda x: f"${x:,.0f}")
                            
                            try:
                                display_df_clean = clean_dataframe_for_streamlit(display_df)
                                st.dataframe(display_df_clean, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error displaying filtered funds table: {e}")
                            
                            # Download button for filtered funds
                            csv = filtered_funds.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Filtered Funds CSV",
                                data=csv,
                                file_name=f"sequential_filtered_funds.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No funds found in the selected categories.")
                    
                    # Sequential model performance
                    if 'Actual_Combined' in df_seq_dash.columns:
                        st.subheader("📈 Sequential Model Performance Summary")
                        
                        # Calculate performance metrics
                        y_true = df_seq_dash['Actual_Combined']
                        y_pred = df_seq_dash['Combined_Prediction']
                        
                        # Remove any invalid predictions
                        valid_mask = (y_true >= 0) & (y_pred >= 0)
                        y_true_valid = y_true[valid_mask]
                        y_pred_valid = y_pred[valid_mask]
                        
                        if len(y_true_valid) > 0:
                            accuracy = accuracy_score(y_true_valid, y_pred_valid)
                            
                            # Calculate per-class metrics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Overall 4-Class Accuracy", f"{accuracy:.3f}")
                                
                                # Confusion matrix
                                cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1, 2, 3])
                                fig = px.imshow(
                                    cm,
                                    text_auto=True,
                                    title="4-Class Confusion Matrix",
                                    labels=dict(x="Predicted", y="Actual"),
                                    x=category_names,
                                    y=category_names
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Classification report
                                class_report = classification_report(
                                    y_true_valid, y_pred_valid,
                                    target_names=category_names,
                                    output_dict=True
                                )
                                
                                # Convert to DataFrame and display key metrics
                                report_df = pd.DataFrame(class_report).transpose()
                                
                                # Show summary metrics
                                if 'macro avg' in report_df.index:
                                    macro_avg = report_df.loc['macro avg']
                                    st.markdown("**Macro Averaged Metrics:**")
                                    col1_inner, col2_inner, col3_inner = st.columns(3)
                                    with col1_inner:
                                        st.metric("Precision", f"{macro_avg['precision']:.3f}")
                                    with col2_inner:
                                        st.metric("Recall", f"{macro_avg['recall']:.3f}")
                                    with col3_inner:
                                        st.metric("F1-Score", f"{macro_avg['f1-score']:.3f}")
                                
                                # Detailed per-class metrics
                                st.markdown("**Per-Class Performance:**")
                                class_metrics = report_df.loc[category_names][['precision', 'recall', 'f1-score']].round(3)
                                st.dataframe(class_metrics, use_container_width=True)
                        else:
                            st.warning("Unable to calculate performance metrics - no valid predictions")
                    
                except Exception as e:
                    st.error(f"❌ Error creating sequential risk dashboard: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏢 Great Gray Fund Outflow Early Warning System</p>
    <p>Binary & Sequential Classification Models • Built with Streamlit • Powered by Random Forest</p>
    <p><em>Using pre-trained models for binary classification and 4-category sequential analysis</em></p>
</div>
""", unsafe_allow_html=True) 
