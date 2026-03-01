import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="AQI Prediction Dashboard", page_icon="🌍", layout="wide")

# Custom CSS for colorful background and dashboard styling
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%); }
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.5);
    }
    .dashboard-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .dashboard-header h1 { color: white; margin-bottom: 5px; font-weight: 700; }
    .dashboard-header p { font-size: 1.1em; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dashboard-header">
    <h1>🌍 AQI Prediction & Analysis Dashboard</h1>
    <p>City & Station Level Air Quality Modeling (AI1110 Project)</p>
</div>
""", unsafe_allow_html=True)

# Data Cleaning Function (Safely handles NaNs)
def clean_dataset(df):
    df_clean = df.copy()
    if "AQI" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["AQI"])
    
    # Safely fill missing values only for numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    if "Date" in df_clean.columns:
        try:
            df_clean["Date"] = pd.to_datetime(df_clean["Date"])
        except Exception:
            pass # Skip if date parsing fails
            
    return df_clean

# ML Training Function
def train_models(df, features):
    X = df[features]
    y = df["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    
    return scaler, rf, lr, y_test, y_pred_rf, rmse_lr, r2_lr, rmse_rf, r2_rf

uploaded_file = st.sidebar.file_uploader("📂 Load Dataset (city_day.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        df = clean_dataset(raw_df)
        
        features = ["PM2.5", "PM10", "NO", "NO2", "SO2", "CO", "O3"]
        has_features = all(f in df.columns for f in features + ["AQI"])

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Overview", 
            "🔍 Exploratory Data Analysis", 
            "🤖 ML Model Training",
            "🎯 Predict AQI"
        ])

        # TAB 1: DATA OVERVIEW
        with tab1:
            st.markdown("### Data Cleaning & Overview")
            c1, c2 = st.columns(2)
            c1.metric("Raw Data Rows", raw_df.shape[0])
            c2.metric("Cleaned Data Rows", df.shape[0], delta=f"-{raw_df.shape[0]-df.shape[0]} rows (Dropped AQI NaNs)")
            # Removed .head(50) so the full dataset is loaded in the viewer
            st.dataframe(df, use_container_width=True)

        # TAB 2: EDA
        with tab2:
            st.markdown("### Exploratory Data Analysis")
            
            if "AQI" in df.columns:
                c1, c2 = st.columns(2)
                with c1:
                    fig_hist = px.histogram(df, x="AQI", nbins=30, marginal="box", title="Distribution of AQI", color_discrete_sequence=['#3b82f6'])
                    st.plotly_chart(fig_hist, use_container_width=True)
                with c2:
                    if "City" in df.columns:
                        top_cities = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10).reset_index()
                        fig_bar = px.bar(top_cities, x="City", y="AQI", title="Top 10 Cities by Average AQI", color="AQI", color_continuous_scale="Reds")
                        st.plotly_chart(fig_bar, use_container_width=True)
                    elif "StationId" in df.columns:
                        top_stations = df.groupby("StationId")["AQI"].mean().sort_values(ascending=False).head(10).reset_index()
                        fig_bar = px.bar(top_stations, x="StationId", y="AQI", title="Top 10 Stations by Average AQI", color="AQI", color_continuous_scale="Reds")
                        st.plotly_chart(fig_bar, use_container_width=True)

                if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
                    trend = df.groupby("Date")["AQI"].mean().reset_index()
                    fig_line = px.line(trend, x="Date", y="AQI", title="Average AQI Over Time")
                    st.plotly_chart(fig_line, use_container_width=True)

                # Safe Correlation Matrix
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty and len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    # Fixed the colorscale issue here: changed "coolwarm" to "RdBu_r"
                    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
                    st.plotly_chart(fig_corr, use_container_width=True)

        # TAB 3: ML MODELS
        with tab3:
            if has_features:
                st.markdown("### Model Evaluation (Linear Regression vs Random Forest)")
                with st.spinner("Training Models..."):
                    scaler, rf, lr, y_test, y_pred_rf, rmse_lr, r2_lr, rmse_rf, r2_rf = train_models(df, features)
                
                m1, m2 = st.columns(2)
                m1.info(f"**Linear Regression**\n\nRMSE: {rmse_lr:.2f}\nR² Score: {r2_lr:.4f}")
                m2.success(f"**Random Forest**\n\nRMSE: {rmse_rf:.2f}\nR² Score: {r2_rf:.4f}")
                
                c1, c2 = st.columns(2)
                with c1:
                    fig_scatter = px.scatter(x=y_test, y=y_pred_rf, labels={'x': 'Actual AQI', 'y': 'Predicted AQI'}, title="Actual vs Predicted AQI (Random Forest)", opacity=0.6)
                    fig_scatter.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="red", dash="dash"))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with c2:
                    importance = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=True)
                    fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', title="Feature Importance (Random Forest)")
                    st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning(f"Dataset must contain these columns for ML: {', '.join(features + ['AQI'])}")

        # TAB 4: PREDICTION TOOL
        with tab4:
            if has_features:
                st.markdown("### 🎯 Predict AQI on New Data")
                st.write("Adjust the environmental features below to predict the AQI using the trained Random Forest model.")
                
                # Train if user skips Tab 3
                if 'rf' not in locals():
                    scaler, rf, lr, _, _, _, _, _, _ = train_models(df, features)
                
                p1, p2, p3 = st.columns(3)
                with p1:
                    pm25 = st.number_input("PM2.5", value=80.0)
                    pm10 = st.number_input("PM10", value=120.0)
                    no = st.number_input("NO", value=5.0)
                with p2:
                    no2 = st.number_input("NO2", value=40.0)
                    so2 = st.number_input("SO2", value=10.0)
                    co = st.number_input("CO", value=1.2)
                with p3:
                    o3 = st.number_input("O3", value=30.0)
                    
                if st.button("Predict AQI", type="primary"):
                    sample = np.array([[pm25, pm10, no, no2, so2, co, o3]])
                    sample_scaled = scaler.transform(sample)
                    pred_aqi = rf.predict(sample_scaled)[0]
                    
                    st.metric(label="Predicted AQI (Random Forest)", value=f"{pred_aqi:.2f}")
            else:
                st.warning("Prediction unavailable due to missing feature columns.")
                
    except Exception as e:
        st.error(f"An error occurred reading or processing the file: {e}")
else:
    st.info("👈 Please upload `city_day.csv` or `station_day.csv` from the sidebar to begin.")