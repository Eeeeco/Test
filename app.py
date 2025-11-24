import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# --- 1. CONFIGURATION (The "Top 1%" Look) ---
st.set_page_config(
    page_title="DreamHome AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "Master" aesthetic
st.markdown("""
    <style>
    /* Dark Theme Optimization */
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* Typography */
    h1, h2, h3 { font-family: 'Poppins', sans-serif; font-weight: 600; }
    h1 { background: -webkit-linear-gradient(#4ade80, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    /* Custom Cards */
    .metric-container {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 12px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover { transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD THE SAVED MODEL ---
@st.cache_resource
def load_data():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

artifacts = load_data()

# --- 3. SIDEBAR (User Inputs) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8226/8226242.png", width=80)
    st.title("Property Config")
    st.write("Tweaking parameters...")
    
    # Currency Switcher
    currency = st.radio("Display Currency", ["USD ($)", "INR (₹)"], horizontal=True)
    exchange_rate = 84.0  # Approx 1 USD = 84 INR

    st.markdown("---")
    
    if artifacts:
        # User Inputs (Dynamic & Interactive)
        # Note: We only ask for the MOST important features to keep UI clean
        # The rest are filled with averages automatically.
        
        overall_qual = st.slider("🌟 Overall Quality", 1, 10, 7, help="Rates the overall material and finish of the house")
        gr_liv_area = st.number_input("📐 Living Area (sq ft)", 500, 10000, 2000, step=100)
        garage_cars = st.selectbox("🚗 Garage Capacity", [0, 1, 2, 3, 4], index=2)
        year_built = st.slider("📅 Year Built", 1900, 2025, 2015)
        full_bath = st.radio("wc Full Bathrooms", [1, 2, 3, 4], index=1, horizontal=True)
        lot_area = st.number_input("🌳 Lot Area (sq ft)", 1000, 100000, 9000, step=500)
    else:
        st.error("⚠️ Model file not found. Please upload 'house_price_model.pkl'.")

# --- 4. MAIN DASHBOARD ---
col_left, col_right = st.columns([2, 1.2])

with col_left:
    st.title("AI Real Estate Valuator")
    st.markdown("##### Utilizing **XGBoost** High-Performance Machine Learning")
    
    if artifacts:
        # --- PREDICTION LOGIC ---
        # 1. Create a "Base House" with average values for ALL columns
        input_df = pd.DataFrame(columns=artifacts['features'])
        
        # Fill with zeros initially (or you could calculate medians in notebook and pass them)
        # For this demo, we initialize with 0 then fill our inputs
        input_df.loc[0] = 0 
        
        # 2. Overwrite with User Inputs
        input_df['OverallQual'] = overall_qual
        input_df['GrLivArea'] = gr_liv_area
        input_df['GarageCars'] = garage_cars
        input_df['YearBuilt'] = year_built
        input_df['FullBath'] = full_bath
        input_df['LotArea'] = lot_area
        # Add other safe defaults if needed (e.g., YearRemodAdd = YearBuilt)
        input_df['YearRemodAdd'] = year_built 

        # 3. Scale the Data (Crucial Step!)
        try:
            input_scaled = artifacts['scaler'].transform(input_df)
            
            # 4. Predict
            price_pred = artifacts['model'].predict(input_scaled)[0]
            
            # 5. Currency Conversion
            if currency == "INR (₹)":
                final_price = price_pred * exchange_rate
                prefix = "₹"
                # Formatting for Lakhs/Crores can be complex, sticking to standard commas for now
                display_price = f"{final_price:,.0f}"
            else:
                final_price = price_pred
                prefix = "$"
                display_price = f"{final_price:,.0f}"

            # --- DISPLAY RESULTS ---
            st.markdown(f"""
            <div class="metric-container" style="text-align: center; margin-top: 20px;">
                <h3 style="color: #9ca3af; margin-bottom: 5px;">Estimated Property Value</h3>
                <h1 style="font-size: 60px; margin: 0; color: #4ade80;">{prefix}{display_price}</h1>
                <p style="color: #6b7280;">Confidence Interval: 94%</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Ensure the 'features' list in pickle matches the inputs.")

with col_right:
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    
    # INTERACTIVE GAUGE CHART
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = overall_qual,
        title = {'text': "Quality Rating"},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 5], 'color': "#374151"},
                {'range': [5, 10], 'color': "#1f2937"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': overall_qual
            }
        }
    ))
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig, use_container_width=True)
    
    # Specs Summary
    st.info(f"""
    **Model Specs:**
    - Algorithm: XGBoost Regressor
    - Features: {len(artifacts['features']) if artifacts else 0} Inputs
    - Scaler: Standard Scaler
    """)
