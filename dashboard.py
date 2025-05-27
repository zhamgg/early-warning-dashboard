import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Main title
st.markdown('<h1 class="main-header">🚨 Fund Outflow Early Warning System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Overview", "Data Analysis", "Model Training", "Predictions", "Early Warning Dashboard"]
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

# Load data
if uploaded_file is not None or os.path.exists("joined_fund_data2.xlsx"):
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            else:
                st.sidebar.error("❌ Failed to load data")
else:
    st.sidebar.warning("⚠️ Please upload the fund data file")

# Page routing
if page == "Overview":
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Objective")
        st.write("""
        **Goal**: Create an early warning system for potential fund outflows by combining:
        - Morningstar performance data
        - Great Gray flow data
        - Machine learning models for prediction
        """)
        
        st.subheader("📈 Models Implemented")
        st.write("""
        1. **Binary Classifier**: Flags outflows > $1M
        2. **Regression Model**: Predicts exact cash flow amounts
        3. **AUM Change Model**: Predicts percentage change in AUM
        """)
    
    with col2:
        st.subheader("📊 Key Metrics Achieved")
        
        # Display metrics in attractive cards
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Binary Classification</h4>
            <p><strong>Accuracy:</strong> ~95%+</p>
            <p><strong>Purpose:</strong> Flag large outflows (>$1M)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Regression Model</h4>
            <p><strong>R² Score:</strong> 0.9355</p>
            <p><strong>MAE:</strong> $487,719.76</p>
            <p><strong>Purpose:</strong> Predict exact cash flows</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>💰 AUM Change Model</h4>
            <p><strong>Target:</strong> AUM % Change</p>
            <p><strong>Features:</strong> Performance + Flow data</p>
            <p><strong>Purpose:</strong> Predict AUM changes</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
        st.success("✅ Data is loaded and ready for analysis!")
    else:
        st.warning("⚠️ Please upload data to proceed with analysis")

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
            st.metric("Total Columns", len(df.columns))
        with col3:
            if 'Net Cash Flows' in df.columns:
                st.metric("Avg Cash Flow", f"${df['Net Cash Flows'].mean():,.0f}")
        with col4:
            if 'Net Cash Flows' in df.columns:
                outflows = (df['Net Cash Flows'] < -1000000).sum()
                st.metric("Large Outflows (>$1M)", outflows)
        
        # Data preview
        st.subheader("📋 Data Preview")
        try:
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying data preview: {e}")
            # Fallback: show basic info
            st.write("Data preview unavailable. Showing basic info:")
            st.write(f"Shape: {df.shape}")
            st.write(f"Columns: {list(df.columns)}")
        
        # Column information
        st.subheader("Column Information")
        try:
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying column information: {e}")
            # Fallback: show basic column info
            st.write("Column information:")
            for i, col in enumerate(df.columns):
                st.write(f"{i+1}. {col} ({df[col].dtype})")
        
        # Visualizations
        if 'Net Cash Flows' in df.columns:
            st.subheader("Cash Flow Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    df, 
                    x='Net Cash Flows', 
                    nbins=50,
                    title="Distribution of Net Cash Flows"
                )
                fig.update_layout(
                    xaxis_title="Net Cash Flows ($)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    df, 
                    y='Net Cash Flows',
                    title="Cash Flow Box Plot"
                )
                fig.update_layout(
                    yaxis_title="Net Cash Flows ($)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_stats = df[numeric_cols].describe()
                st.dataframe(summary_stats, use_container_width=True)
            else:
                st.write("No numeric columns found for summary statistics.")
        except Exception as e:
            st.error(f"Error displaying summary statistics: {e}")
            st.write("Summary statistics unavailable.")

elif page == "Model Training":
    st.header("Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first")
    else:
        df = st.session_state.df
        
        # Model selection
        st.subheader("Select Models to Train")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            train_binary = st.checkbox("Binary Classifier (Outflow >$1M)", value=True)
        with col2:
            train_regression = st.checkbox("Cash Flow Regression", value=True)
        with col3:
            train_aum = st.checkbox("AUM Change Prediction", value=True)
        
        if st.button("Train Selected Models", type="primary"):
            results = {}
            
            # Prepare data
            try:
                # Binary Classification Model
                if train_binary and 'Net Cash Flows' in df.columns:
                    with st.spinner("Training Binary Classifier..."):
                        st.subheader("Binary Classification Results")
                        
                        # Create target
                        df_binary = df.copy()
                        df_binary['target'] = (df_binary['Net Cash Flows'] < -1000000).astype(int)
                        
                        # Prepare features
                        X = df_binary.drop(['target', 'Net Cash Flows'], axis=1, errors='ignore')
                        y = df_binary['target']
                        
                        # Preprocessing
                        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        
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
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                        
                        # Train model
                        binary_pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                        ])
                        binary_pipeline.fit(X_train, y_train)
                        binary_pred = binary_pipeline.predict(X_test)
                        
                        # Metrics
                        accuracy = accuracy_score(y_test, binary_pred)
                        precision = precision_score(y_test, binary_pred)
                        recall = recall_score(y_test, binary_pred)
                        f1 = f1_score(y_test, binary_pred)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.4f}")
                        with col2:
                            st.metric("Precision", f"{precision:.4f}")
                        with col3:
                            st.metric("Recall", f"{recall:.4f}")
                        with col4:
                            st.metric("F1 Score", f"{f1:.4f}")
                        
                        # Confusion Matrix
                        cm = confusion_matrix(y_test, binary_pred)
                        fig = px.imshow(cm, 
                                       text_auto=True, 
                                       title="Confusion Matrix - Binary Classifier",
                                       labels=dict(x="Predicted", y="Actual"))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save model
                        joblib.dump(binary_pipeline, 'binary_model.joblib')
                        results['binary'] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1
                        }
                
                # Regression Model
                if train_regression and 'Net Cash Flows' in df.columns:
                    with st.spinner("Training Regression Model..."):
                        st.subheader("Regression Model Results")
                        
                        # Prepare data
                        df_reg = df.copy()
                        X = df_reg.drop(['Net Cash Flows'], axis=1, errors='ignore')
                        y = df_reg['Net Cash Flows']
                        
                        # Remove infinite values
                        mask = np.isfinite(y)
                        X = X[mask]
                        y = y[mask]
                        
                        # Preprocessing (same as binary)
                        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        
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
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Train model
                        reg_pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
                        ])
                        reg_pipeline.fit(X_train, y_train)
                        reg_pred = reg_pipeline.predict(X_test)
                        
                        # Metrics
                        mae = mean_absolute_error(y_test, reg_pred)
                        mse = mean_squared_error(y_test, reg_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, reg_pred)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MAE", f"${mae:,.0f}")
                        with col2:
                            st.metric("RMSE", f"${rmse:,.0f}")
                        with col3:
                            st.metric("R² Score", f"{r2:.4f}")
                        with col4:
                            st.metric("MSE", f"${mse:,.0f}")
                        
                        # Prediction vs Actual scatter plot
                        fig = px.scatter(
                            x=y_test, 
                            y=reg_pred,
                            title="Actual vs Predicted Cash Flows",
                            labels={'x': 'Actual Cash Flows ($)', 'y': 'Predicted Cash Flows ($)'}
                        )
                        # Add diagonal line
                        min_val = min(y_test.min(), reg_pred.min())
                        max_val = max(y_test.max(), reg_pred.max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val], 
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='red')
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save model
                        joblib.dump(reg_pipeline, 'regression_model.joblib')
                        results['regression'] = {
                            'mae': mae,
                            'rmse': rmse,
                            'r2': r2,
                            'mse': mse
                        }
                
                # AUM Change Model
                if train_aum and 'Beginning Net Assets' in df.columns and 'Ending Net Assets' in df.columns:
                    with st.spinner("Training AUM Change Model..."):
                        st.subheader("💰 AUM Change Model Results")
                        
                        # Calculate AUM percentage change
                        df_aum = df.copy()
                        df_aum['Beginning Net Assets'] = pd.to_numeric(df_aum['Beginning Net Assets'], errors='coerce')
                        df_aum['Ending Net Assets'] = pd.to_numeric(df_aum['Ending Net Assets'], errors='coerce')
                        df_aum['AUM_Pct_Change'] = ((df_aum['Ending Net Assets'] - df_aum['Beginning Net Assets']) / df_aum['Beginning Net Assets']) * 100
                        
                        # Clean data
                        df_aum = df_aum.replace([np.inf, -np.inf], np.nan)
                        df_aum = df_aum.dropna(subset=['AUM_Pct_Change'])
                        df_aum['AUM_Pct_Change'] = df_aum['AUM_Pct_Change'].clip(-50, 50)
                        
                        # Prepare features
                        columns_to_drop = ['AUM_Pct_Change', 'Ending Net Assets', 'Net Change AUM', 'Beginning Net Assets']
                        X = df_aum.drop(columns_to_drop, axis=1, errors='ignore')
                        y = df_aum['AUM_Pct_Change']
                        
                        # Preprocessing
                        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        
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
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Train model
                        aum_pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
                        ])
                        aum_pipeline.fit(X_train, y_train)
                        aum_pred = aum_pipeline.predict(X_test)
                        
                        # Metrics
                        mae = mean_absolute_error(y_test, aum_pred)
                        mse = mean_squared_error(y_test, aum_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, aum_pred)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MAE", f"{mae:.2f}%")
                        with col2:
                            st.metric("RMSE", f"{rmse:.2f}%")
                        with col3:
                            st.metric("R² Score", f"{r2:.4f}")
                        with col4:
                            st.metric("Mean AUM Change", f"{y.mean():.2f}%")
                        
                        # AUM change distribution
                        fig = px.histogram(
                            x=y_test,
                            nbins=30,
                            title="Distribution of AUM Percentage Changes"
                        )
                        fig.update_layout(
                            xaxis_title="AUM Change (%)",
                            yaxis_title="Frequency"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save model
                        joblib.dump(aum_pipeline, 'aum_model.joblib')
                        results['aum'] = {
                            'mae': mae,
                            'rmse': rmse,
                            'r2': r2
                        }
                
                st.session_state.models_trained = True
                st.success("✅ Model training completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during model training: {e}")

elif page == "Predictions":
    st.header("Make Predictions")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first")
    elif not st.session_state.models_trained and not any(os.path.exists(f) for f in ['binary_model.joblib', 'regression_model.joblib', 'aum_model.joblib']):
        st.warning("⚠️ Please train models first or ensure model files are available")
    else:
        df = st.session_state.df
        
        st.subheader("Select a Fund for Prediction")
        
        # Fund selection
        if 'Fund Name_x' in df.columns:
            fund_options = df['Fund Name_x'].dropna().unique()
            selected_fund = st.selectbox("Choose a fund:", fund_options)
            fund_data = df[df['Fund Name_x'] == selected_fund].iloc[0:1]
        else:
            # Random selection if no fund name column
            fund_idx = st.number_input("Select fund index:", 0, len(df)-1, 0)
            fund_data = df.iloc[fund_idx:fund_idx+1]
            selected_fund = f"Fund #{fund_idx}"
        
        st.subheader(f"Predictions for: {selected_fund}")
        
        # Make predictions
        col1, col2, col3 = st.columns(3)
        
        # Binary prediction
        with col1:
            if os.path.exists('binary_model.joblib'):
                try:
                    binary_model = joblib.load('binary_model.joblib')
                    fund_features = fund_data.drop(['Net Cash Flows'], axis=1, errors='ignore')
                    binary_pred = binary_model.predict(fund_features)[0]
                    binary_prob = binary_model.predict_proba(fund_features)[0][1]
                    
                    st.markdown("### 🚨 Outflow Risk")
                    if binary_pred == 1:
                        st.markdown(f"""
                        <div class="warning-high">
                            <h4>⚠️ HIGH RISK</h4>
                            <p>Predicted outflow > $1M</p>
                            <p><strong>Probability: {binary_prob:.1%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-low">
                            <h4>✅ LOW RISK</h4>
                            <p>No major outflow expected</p>
                            <p><strong>Probability: {binary_prob:.1%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Binary prediction error: {e}")
        
        # Regression prediction
        with col2:
            if os.path.exists('regression_model.joblib'):
                try:
                    reg_model = joblib.load('regression_model.joblib')
                    fund_features = fund_data.drop(['Net Cash Flows'], axis=1, errors='ignore')
                    cash_flow_pred = reg_model.predict(fund_features)[0]
                    
                    st.markdown("### 💰 Cash Flow Prediction")
                    
                    if 'Net Cash Flows' in fund_data.columns:
                        actual_flow = fund_data['Net Cash Flows'].iloc[0]
                        error = abs(actual_flow - cash_flow_pred)
                        st.metric(
                            "Predicted Cash Flow", 
                            f"${cash_flow_pred:,.0f}",
                            delta=f"${cash_flow_pred - actual_flow:,.0f}"
                        )
                        st.metric("Actual Cash Flow", f"${actual_flow:,.0f}")
                        st.metric("Prediction Error", f"${error:,.0f}")
                    else:
                        st.metric("Predicted Cash Flow", f"${cash_flow_pred:,.0f}")
                except Exception as e:
                    st.error(f"Regression prediction error: {e}")
        
        # AUM change prediction
        with col3:
            if os.path.exists('aum_model.joblib'):
                try:
                    aum_model = joblib.load('aum_model.joblib')
                    columns_to_drop = ['AUM_Pct_Change', 'Ending Net Assets', 'Net Change AUM', 'Beginning Net Assets']
                    fund_features = fund_data.drop(columns_to_drop, axis=1, errors='ignore')
                    aum_change_pred = aum_model.predict(fund_features)[0]
                    
                    st.markdown("### 📈 AUM Change Prediction")
                    
                    if 'Beginning Net Assets' in fund_data.columns and 'Ending Net Assets' in fund_data.columns:
                        beginning = pd.to_numeric(fund_data['Beginning Net Assets'].iloc[0], errors='coerce')
                        ending = pd.to_numeric(fund_data['Ending Net Assets'].iloc[0], errors='coerce')
                        if pd.notna(beginning) and pd.notna(ending) and beginning != 0:
                            actual_change = ((ending - beginning) / beginning) * 100
                            st.metric(
                                "Predicted AUM Change", 
                                f"{aum_change_pred:.2f}%",
                                delta=f"{aum_change_pred - actual_change:.2f}%"
                            )
                            st.metric("Actual AUM Change", f"{actual_change:.2f}%")
                        else:
                            st.metric("Predicted AUM Change", f"{aum_change_pred:.2f}%")
                    else:
                        st.metric("Predicted AUM Change", f"{aum_change_pred:.2f}%")
                except Exception as e:
                    st.error(f"AUM prediction error: {e}")
        
        # Fund details
        st.subheader("📋 Fund Details")
        try:
            # Clean the fund data before displaying
            fund_data_clean = clean_dataframe_for_streamlit(fund_data)
            st.dataframe(fund_data_clean.T, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying fund details: {e}")
            # Fallback: show key information
            st.write("Fund details (basic view):")
            for col in fund_data.columns:
                try:
                    value = fund_data[col].iloc[0]
                    st.write(f"**{col}**: {value}")
                except:
                    st.write(f"**{col}**: [Error displaying value]")

elif page == "Early Warning Dashboard":
    st.header("Early Warning Dashboard")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first")
    else:
        df = st.session_state.df
        
        # Calculate risk scores for all funds
        if os.path.exists('binary_model.joblib') and 'Net Cash Flows' in df.columns:
            try:
                binary_model = joblib.load('binary_model.joblib')
                X = df.drop(['Net Cash Flows'], axis=1, errors='ignore')
                
                # Get predictions and probabilities
                risk_probs = binary_model.predict_proba(X)[:, 1]
                risk_predictions = binary_model.predict(X)
                
                # Create risk dashboard
                df_risk = df.copy()
                df_risk['Risk_Probability'] = risk_probs
                df_risk['High_Risk'] = risk_predictions
                
                # Risk summary
                st.subheader("Risk Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                total_funds = len(df_risk)
                high_risk_funds = df_risk['High_Risk'].sum()
                avg_risk = df_risk['Risk_Probability'].mean()
                
                with col1:
                    st.metric("Total Funds", total_funds)
                with col2:
                    st.metric("High Risk Funds", high_risk_funds)
                with col3:
                    st.metric("Average Risk", f"{avg_risk:.1%}")
                with col4:
                    risk_pct = high_risk_funds / total_funds if total_funds > 0 else 0
                    st.metric("% High Risk", f"{risk_pct:.1%}")
                
                # Risk distribution
                st.subheader("Risk Distribution")
                fig = px.histogram(
                    df_risk, 
                    x='Risk_Probability',
                    nbins=20,
                    title="Distribution of Outflow Risk Probabilities"
                )
                fig.update_layout(
                    xaxis_title="Risk Probability",
                    yaxis_title="Number of Funds"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # High-risk funds table
                st.subheader("⚠️ High-Risk Funds (>50% probability)")
                high_risk_df = df_risk[df_risk['Risk_Probability'] > 0.5].copy()
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
                    
                    # Format the dataframe
                    display_df = high_risk_df[display_cols].copy()
                    display_df['Risk_Probability'] = display_df['Risk_Probability'].apply(lambda x: f"{x:.1%}")
                    
                    try:
                        # Clean the display dataframe
                        display_df_clean = clean_dataframe_for_streamlit(display_df)
                        st.dataframe(display_df_clean, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying high-risk funds table: {e}")
                        # Fallback: show basic list
                        st.write("High-risk funds (basic view):")
                        for idx, row in display_df.iterrows():
                            st.write(f"- {row.get('Fund Name_x', f'Fund {idx}')}: {row['Risk_Probability']}")
                    
                    # Download button for high-risk funds
                    csv = high_risk_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download High-Risk Funds CSV",
                        data=csv,
                        file_name="high_risk_funds.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("✅ No high-risk funds identified!")
                
                # Cash flow vs risk scatter plot
                if 'Net Cash Flows' in df_risk.columns:
                    st.subheader("💰 Cash Flow vs Risk Analysis")
                    fig = px.scatter(
                        df_risk,
                        x='Net Cash Flows',
                        y='Risk_Probability',
                        title="Cash Flow vs Outflow Risk Probability",
                        hover_data=['Fund Name_x'] if 'Fund Name_x' in df_risk.columns else None
                    )
                    fig.update_layout(
                        xaxis_title="Net Cash Flows ($)",
                        yaxis_title="Risk Probability"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating risk dashboard: {e}")
        else:
            st.warning("⚠️ Binary classification model not found. Please train the model first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏢 Great Gray Fund Outflow Early Warning System</p>
    <p>Built with Streamlit • Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True) 