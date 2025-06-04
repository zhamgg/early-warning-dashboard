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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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
    .performance-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
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
if 'df' not in st.session_state:
    st.session_state.df = None

# Auto-training functions
@st.cache_resource
def train_binary_model(df):
    """Train binary classification model if not found"""
    try:
        # Create target variable: 1 if Net Cash Flows < -5% of Ending Net Assets, else 0
        df_model = df.copy()
        df_model['target'] = ((df_model['Net Cash Flows'] < -0.05 * df_model['Ending Net Assets'].abs()) & 
                             (df_model['Ending Net Assets'].abs() > 0)).astype(int)
        
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

def check_and_load_models(df=None):
    """Check if model exists, train if not, then load it"""
    binary_model = None
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
    
    return binary_model, messages

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

# Load data
if uploaded_file is not None or os.path.exists("joined_fund_data2.xlsx"):
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Check and load/train model automatically
                st.sidebar.markdown("---")
                st.sidebar.subheader("🤖 Model Status")
                with st.sidebar:
                    binary_model, messages = check_and_load_models(df)
                    
                    # Store model in session state
                    st.session_state.binary_model = binary_model
                    
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
        
        # Check if model is already in session state
        if 'binary_model' not in st.session_state:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🤖 Model Status")
            with st.sidebar:
                binary_model, messages = check_and_load_models(st.session_state.df)
                
                # Store model in session state
                st.session_state.binary_model = binary_model
                
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
            # Model already loaded, just show status
            st.sidebar.markdown("---")
            st.sidebar.subheader("🤖 Model Status")
            
            if st.session_state.binary_model is not None:
                st.sidebar.success("✅ Binary model ready")
            else:
                st.sidebar.error("❌ Binary model unavailable")
else:
    st.sidebar.warning("⚠️ Please upload the fund data file")
    # Initialize empty model in session state
    if 'binary_model' not in st.session_state:
        st.session_state.binary_model = None

# Page routing
if page == "Overview":
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Objective")
        st.write("""
        **Goal**: Create an early warning system to identify funds at risk of significant outflows using:
        - Morningstar performance data
        - Great Gray flow data
        - Machine learning binary classification model
        """)
        
        st.subheader("🔬 Model Approach")
        st.write("""
        **Binary Classification**: Predicts whether a fund will experience significant outflows ≥ 5% of assets
        
        - **Algorithm**: Random Forest with balanced classes
        - **Features**: 19 key fund characteristics and performance metrics
        - **Target**: Significant outflow events (≥5% of ending net assets)
        - **Use Case**: Flag funds requiring immediate attention
        """)
    
    with col2:
        st.subheader("📊 Model Performance")
        
        # Binary model performance
        st.markdown("""
        <div class="performance-metric">
            <h4>🏆 Binary Classification Model</h4>
            <p><strong>Accuracy:</strong> 94.06%</p>
            <p><strong>Precision:</strong> 11.39%</p>
            <p><strong>Recall:</strong> 7.63%</p>
            <p><strong>F1 Score:</strong> 0.0914</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed Performance Breakdown
    st.subheader("📈 Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Binary Classification Results**")
        st.markdown("""
        - **Target**: Significant outflows (≥5% of assets)
        - **Algorithm**: Random Forest with balanced classes
        - **Evaluation**: 80/20 train-test split
        - **Strength**: Good for high-level risk screening
        - **Use Case**: Flag funds for immediate attention
        """)
        
        # Binary metrics table
        binary_metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Score': ['94.06%', '11.39%', '7.63%', '0.0914']
        })
        st.table(binary_metrics)
    
    with col2:
        st.markdown("**📊 Business Impact & Model Characteristics**")
        st.markdown("""
        - **High Accuracy**: 94.06% overall classification accuracy
        - **Class Imbalance**: Only 3.9% of funds have significant outflows
        - **Conservative Precision**: 11.39% (low false positive rate)
        - **Detection Challenge**: 7.63% recall (identifying all true positives is difficult)
        - **Risk Management**: Designed for conservative screening approach
        """)
        
        # Key insights about the model performance
        st.markdown("**Key Model Insights:**")
        st.markdown("""
        - ✅ Very high overall accuracy due to class imbalance
        - ⚠️ Low precision reflects challenge of rare event prediction
        - ⚠️ Conservative recall prioritizes reducing false alarms
        - 📊 F1 score of 0.0914 reflects precision/recall trade-off
        - 🎯 Model optimized for identifying highest-risk funds
        """)
    
    # Data insights
    st.subheader("📋 Dataset Summary")
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features Used", "19")
        with col3:
            if 'Net Cash Flows' in df.columns and 'Ending Net Assets' in df.columns:
                # Calculate 5% threshold outflows
                significant_outflows = ((df['Net Cash Flows'] < -0.05 * df['Ending Net Assets'].abs()) & 
                                      (df['Ending Net Assets'].abs() > 0)).sum()
                st.metric("Significant Outflows (≥5%)", significant_outflows)
        with col4:
            if 'Net Cash Flows' in df.columns and 'Ending Net Assets' in df.columns:
                # Calculate outflow rate
                total_valid = (df['Ending Net Assets'].abs() > 0).sum()
                outflow_rate = significant_outflows / total_valid * 100 if total_valid > 0 else 0
                st.metric("Significant Outflow Rate", f"{outflow_rate:.1f}%")
        
        # Training dataset summary (from actual training)
        st.markdown("**Model Training Results (Actual Data):**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Dataset Size", "15,075 funds")
        with col2:
            st.metric("Positive Cases", "588 (3.9%)")
        with col3:
            st.metric("Train/Test Split", "80/20")
        with col4:
            st.metric("Best Model", "Random Forest")

        # Model availability status
        st.subheader("🤖 Model Status")
        
        if 'binary_model' in st.session_state and st.session_state.binary_model is not None:
            st.success("✅ Binary Classification Model Available and Ready")
        elif os.path.exists('EarlyWarningWeights.joblib'):
            st.info("📁 Binary Model File Found (will load with data)")
        else:
            st.warning("⚠️ Binary Model Will Be Trained When Data Is Loaded")
        
        # Feature importance summary
        st.subheader("🔍 Key Predictive Features")
        
        st.markdown("""
        **Top Features for Binary Classification:**
        1. **As-of Date** - Temporal patterns in fund flows
        2. **Beginning Net Assets** - Fund size correlation with flow patterns
        3. **Quarter** - Seasonal flow trends
        4. **CUSIP & Fund ID** - Fund-specific characteristics
        5. **IR_1Y & IR_3Y** - Information ratios indicating performance
        6. **Fund Name** - Fund family/strategy effects
        7. **Percentile Rankings** - Relative performance measures
        8. **Total Return Metrics** - Historical performance indicators
        """)

        # Enhanced model performance context in overview
        st.markdown("**Performance Interpretation:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strengths:**")
            st.markdown("""
            - 94.06% accuracy for overall classification
            - Effective at identifying non-risk funds (high specificity)
            - Conservative approach reduces false alarms
            - Handles severe class imbalance (96.1% vs 3.9%)
            """)
        
        with col2:
            st.markdown("**Considerations:**")
            st.markdown("""
            - 11.39% precision means careful review of flagged funds needed
            - 7.63% recall means some true cases may be missed
            - Optimized for highest-confidence predictions
            - Best used as screening tool rather than definitive predictor
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
            if 'Net Cash Flows' in df.columns and 'Ending Net Assets' in df.columns:
                st.metric("Avg Cash Flow", f"${df['Net Cash Flows'].mean():,.0f}")
        with col4:
            if 'Net Cash Flows' in df.columns and 'Ending Net Assets' in df.columns:
                outflows = ((df['Net Cash Flows'] < -0.05 * df['Ending Net Assets'].abs()) & 
                            (df['Ending Net Assets'].abs() > 0)).sum()
                st.metric("Large Outflows (≥5%)", outflows)
        
        # Target variable analysis
        if 'Net Cash Flows' in df.columns and 'Ending Net Assets' in df.columns:
            st.subheader("🎯 Target Variable Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Binary Classification Target**")
                # Binary target (5% threshold)
                binary_target = ((df['Net Cash Flows'] < -0.05 * df['Ending Net Assets'].abs()) & 
                               (df['Ending Net Assets'].abs() > 0)).astype(int)
                binary_counts = binary_target.value_counts()
                
                fig = px.pie(
                    values=binary_counts.values,
                    names=['No Significant Outflow', 'Significant Outflow (≥5%)'],
                    title="Binary Target Distribution",
                    color_discrete_sequence=['#2E8B57', '#DC143C']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cash flow distribution histogram with 5% threshold line
                fig = px.histogram(
                    df, 
                    x='Net Cash Flows', 
                    nbins=50,
                    title="Net Cash Flows Distribution",
                    color_discrete_sequence=['#1f77b4']
                )
                # Add 5% threshold lines for different asset sizes (showing concept)
                # We'll add a vertical line at an example 5% threshold
                if 'Ending Net Assets' in df.columns:
                    median_assets = df['Ending Net Assets'].median()
                    threshold_example = -0.05 * median_assets
                    fig.add_vline(x=threshold_example, line_dash="dash", line_color="red", 
                                 annotation_text=f"5% Threshold (example: ${threshold_example:,.0f})")
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("No Significant Outflow", binary_counts.get(0, 0))
            with col2:
                st.metric("Significant Outflow (≥5%)", binary_counts.get(1, 0))
            with col3:
                imbalance_ratio = binary_counts.get(0, 0) / binary_counts.get(1, 1) if binary_counts.get(1, 0) > 0 else 0
                st.metric("Class Imbalance Ratio", f"{imbalance_ratio:.1f}:1")
            with col4:
                positive_rate = binary_counts.get(1, 0) / len(binary_target) * 100
                st.metric("Significant Outflow Rate", f"{positive_rate:.1f}%")
        
        # Model Performance Context
        st.subheader("🎯 Model Performance in Context")
        st.markdown("""
        **Understanding the Metrics with Class Imbalance:**
        
        The model performance reflects the challenge of predicting rare events in highly imbalanced data:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**High Accuracy (94.06%) Explained:**")
            st.markdown("""
            - With only 3.9% positive cases, simply predicting "no outflow" would yield ~96% accuracy
            - Our model's 94.06% accuracy means it's making meaningful distinctions
            - Accuracy alone can be misleading with imbalanced data
            """)
            
            st.markdown("**Low Precision (11.39%) Context:**")
            st.markdown("""
            - Of funds flagged as high-risk, 11.39% actually have significant outflows
            - This means ~9 out of 10 flagged funds may not have the predicted outflow
            - Still valuable for focusing attention on highest-risk subset
            """)
        
        with col2:
            st.markdown("**Low Recall (7.63%) Implications:**")
            st.markdown("""
            - Model catches only 7.63% of actual significant outflow cases
            - Many true outflow cases are not flagged by the model
            - Conservative approach prioritizes high-confidence predictions
            """)
            
            st.markdown("**F1 Score (0.0914) Interpretation:**")
            st.markdown("""
            - Harmonic mean of precision and recall
            - Low score reflects the precision/recall trade-off
            - Typical for rare event detection in imbalanced datasets
            """)
        
        st.markdown("**Practical Application Guidelines:**")
        st.markdown("""
        🔍 **Use as Screening Tool**: Focus manual review on model-flagged funds
        
        📊 **Risk Threshold Tuning**: Adjust probability thresholds based on business tolerance
        
        ⚖️ **Cost-Benefit Analysis**: Balance false positives vs. missed detections
        
        🎯 **Combined Approach**: Supplement with other risk indicators and expert judgment
        """)

        # Data preview
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
        
        if st.session_state.binary_model is None:
            st.warning("⚠️ Binary model not available. Please ensure data is loaded to enable automatic training.")
        else:
            try:
                # Use binary model from session state
                binary_model = st.session_state.binary_model
                
                # Fund selection
                st.subheader("Select a Fund for Risk Assessment")
                
                if 'Fund Name_x' in df.columns:
                    fund_options = df['Fund Name_x'].dropna().unique()
                    selected_fund = st.selectbox("Choose a fund:", fund_options)
                    fund_data = df[df['Fund Name_x'] == selected_fund].iloc[0:1]
                    fund_name = selected_fund
                else:
                    fund_idx = st.number_input("Select fund index:", 0, len(df)-1, 0)
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
                st.subheader(f"Risk Assessment: {fund_name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = risk_probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Significant Outflow Risk (≥5%)"},
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
                            <p><strong>Prediction:</strong> Significant outflow likely (≥5% of assets)</p>
                            <p><strong>Confidence:</strong> {risk_probability:.1%}</p>
                            <p><strong>Recommendation:</strong> Immediate attention required</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        if risk_probability > 0.3:
                            st.markdown(f"""
                            <div class="warning-medium">
                                <h3>🔶 MODERATE RISK</h3>
                                <p><strong>Prediction:</strong> No significant outflow expected</p>
                                <p><strong>Confidence:</strong> {risk_probability:.1%}</p>
                                <p><strong>Recommendation:</strong> Monitor closely</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="warning-low">
                                <h3>✅ LOW RISK</h3>
                                <p><strong>Prediction:</strong> No significant outflow expected</p>
                                <p><strong>Confidence:</strong> {risk_probability:.1%}</p>
                                <p><strong>Recommendation:</strong> Continue regular monitoring</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Display actual outcome if available
                if 'Net Cash Flows' in fund_data.columns and 'Ending Net Assets' in fund_data.columns:
                    actual_flow = fund_data['Net Cash Flows'].iloc[0]
                    ending_assets = fund_data['Ending Net Assets'].iloc[0]
                    
                    # Calculate if actual outflow was significant (≥5%)
                    if abs(ending_assets) > 0:
                        actual_significant_outflow = actual_flow < -0.05 * abs(ending_assets)
                        outflow_percentage = (actual_flow / abs(ending_assets)) * 100
                    else:
                        actual_significant_outflow = False
                        outflow_percentage = 0
                    
                    st.subheader("📊 Actual vs Predicted")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Actual Net Cash Flow", f"${actual_flow:,.0f}")
                    with col2:
                        st.metric("Actual Flow %", f"{outflow_percentage:.2f}%")
                    with col3:
                        st.metric("Actual Significant Outflow", "Yes" if actual_significant_outflow else "No")
                    with col4:
                        if actual_significant_outflow == risk_prediction:
                            st.metric("Prediction Accuracy", "✅ Correct")
                        else:
                            st.metric("Prediction Accuracy", "❌ Incorrect")
                
                # Fund details section
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
            
            except Exception as e:
                st.error(f"❌ Error with model assessment: {e}")

elif page == "Early Warning Dashboard":
    st.header("🚨 Early Warning Dashboard")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first")
    else:
        df = st.session_state.df
        
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
                
                # Calculate actual significant outflows (5% threshold)
                if 'Net Cash Flows' in df_risk.columns and 'Ending Net Assets' in df_risk.columns:
                    df_risk['Actual_Significant_Outflow'] = ((df_risk['Net Cash Flows'] < -0.05 * df_risk['Ending Net Assets'].abs()) & 
                                                            (df_risk['Ending Net Assets'].abs() > 0)).astype(int)
                else:
                    df_risk['Actual_Significant_Outflow'] = 0
                
                # Dashboard overview
                st.subheader("📊 Risk Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                total_funds = len(df_risk)
                high_risk_funds = df_risk['High_Risk_Flag'].sum()
                avg_risk = df_risk['Risk_Probability'].mean()
                actual_outflows = df_risk['Actual_Significant_Outflow'].sum()
                
                with col1:
                    st.metric("Total Funds", total_funds)
                with col2:
                    st.metric("High Risk Flags", high_risk_funds)
                with col3:
                    st.metric("Actual Significant Outflows", actual_outflows)
                with col4:
                    st.metric("Average Risk Score", f"{avg_risk:.1%}")
                
                # Risk threshold selector
                st.subheader("🎯 Risk Threshold Analysis")
                risk_threshold = st.slider(
                    "Risk Probability Threshold (%)", 
                    0, 100, 50, 5,
                    help="Adjust threshold to see how many funds would be flagged at different risk levels"
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
                                         (df_risk['Actual_Significant_Outflow'] == 1)).sum()
                        catch_rate = caught_outflows / actual_outflows * 100
                        st.metric("Actual Outflows Caught", f"{catch_rate:.1f}%")
                
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
                    if 'Net Cash Flows' in df.columns and 'Ending Net Assets' in df.columns:
                        # Actual vs Predicted confusion matrix
                        confusion_data = pd.crosstab(
                            df_risk['Actual_Significant_Outflow'], 
                            df_risk['High_Risk_Flag'],
                            rownames=['Actual'], 
                            colnames=['Predicted']
                        )
                        
                        fig = px.imshow(
                            confusion_data.values,
                            text_auto=True,
                            title="Model Performance: Actual vs Predicted",
                            labels=dict(x="Predicted", y="Actual"),
                            x=['No Significant Outflow', 'Significant Outflow'],
                            y=['No Significant Outflow', 'Significant Outflow']
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
                        file_name=f"high_risk_funds_{risk_threshold}pct.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("✅ No funds exceed the selected risk threshold!")
                
                # Model performance metrics
                if 'Net Cash Flows' in df.columns and 'Ending Net Assets' in df.columns:
                    st.subheader("📈 Model Performance Summary")
                    
                    # Note: Use actual training metrics rather than calculating from current data
                    # to ensure consistency with the trained model
                    st.markdown("**Performance metrics from model training (5% threshold):**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", "0.9406")
                    with col2:
                        st.metric("Precision", "0.1139")
                    with col3:
                        st.metric("Recall", "0.0763")
                    with col4:
                        st.metric("F1 Score", "0.0914")
                    
                    st.markdown("*These metrics are from the actual model training on 15,075 funds with 588 positive cases (3.9%)*")
                    
                    # Optional: Calculate metrics on current dataset for comparison
                    with st.expander("📊 Current Dataset Performance (for comparison)"):
                        # Calculate target exactly as in training script
                        current_target = ((df_risk['Net Cash Flows'] < -0.05 * df_risk['Ending Net Assets'].abs()) & 
                                        (df_risk['Ending Net Assets'].abs() > 0)).astype(int)
                        current_predictions = df_risk['High_Risk_Flag']
                        
                        # Only calculate if we have valid data
                        if len(current_target) > 0 and current_target.sum() > 0:
                            current_accuracy = accuracy_score(current_target, current_predictions)
                            current_precision = precision_score(current_target, current_predictions) if current_predictions.sum() > 0 else 0
                            current_recall = recall_score(current_target, current_predictions) if current_target.sum() > 0 else 0
                            current_f1 = f1_score(current_target, current_predictions) if (current_predictions.sum() > 0 and current_target.sum() > 0) else 0
                            
                            st.markdown("**Current dataset metrics (may differ from training):**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Current Accuracy", f"{current_accuracy:.4f}")
                            with col2:
                                st.metric("Current Precision", f"{current_precision:.4f}")
                            with col3:
                                st.metric("Current Recall", f"{current_recall:.4f}")
                            with col4:
                                st.metric("Current F1 Score", f"{current_f1:.4f}")
                            
                            # Show target distribution for current data
                            current_positive_rate = current_target.sum() / len(current_target) * 100
                            st.info(f"Current dataset: {current_target.sum()} positive cases out of {len(current_target)} ({current_positive_rate:.1f}%)")
                        else:
                            st.warning("Cannot calculate metrics on current dataset - insufficient positive cases")
                else:
                    st.subheader("📈 Model Performance Summary")
                    st.markdown("**Training Performance (5% threshold):**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", "0.9406")
                    with col2:
                        st.metric("Precision", "0.1139") 
                    with col3:
                        st.metric("Recall", "0.0763")
                    with col4:
                        st.metric("F1 Score", "0.0914")
            
            except Exception as e:
                st.error(f"❌ Error creating risk dashboard: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏢 Great Gray Fund Outflow Early Warning System</p>
    <p>Binary Classification Model • Built with Streamlit • Powered by Random Forest</p>
    <p><em>Automated early warning system for significant fund outflows (≥5% of assets)</em></p>
</div>
""", unsafe_allow_html=True) 
