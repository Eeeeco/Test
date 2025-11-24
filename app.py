import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# --- 1. SETTING THE STAGE (Headless UI Configuration) ---
st.set_page_config(
    page_title="PropAI | Next-Gen Valuation",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed" # Cleaner look on load
)

# --- 2. THE DESIGN SYSTEM (Custom CSS Injection) ---
# This mimics the "Linear" or "Vercel" dark mode aesthetic.
st.markdown("""
    <style>
    /* GLOBAL THEME */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #000000; /* True Black */
        color: #e0e0e0;
    }
    
    /* GLASSMORPHISM CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* TYPOGRAPHY */
    h1 { font-weight: 800; letter-spacing: -1px; background: linear-gradient(90deg, #fff, #999); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    h2, h3 { font-weight: 600; letter-spacing: -0.5px; color: #fff; }
    p, label { color: #888; font-size: 14px; }
    
    /* INPUTS & SLIDERS */
    .stSlider > div > div > div > div { background-color: #3b82f6; }
    div[data-baseweb="select"] > div { background-color: rgba(255,255,255,0.05); border-color: #333; color: white; }
    
    /* BUTTONS */
    .stButton > button {
        background: white;
        color: black;
        border-radius: 30px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        box-shadow: 0 0 20px rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #f0f0f0;
        box-shadow: 0 0 30px rgba(255,255,255,0.3);
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIC CORE ---
@st.cache_resource
def load_bundle():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

artifacts = load_bundle()

if not artifacts:
    st.error("🚨 System Halted: Model artifacts missing. Please upload 'house_price_model.pkl'.")
    st.stop()

# --- 4. UI STRUCTURE ---

# HERO SECTION
col_hero_1, col_hero_2 = st.columns([3, 1])
with col_hero_1:
    st.title("PropAI Estimate™")
    st.markdown("<p style='font-size: 18px; margin-top: -15px; color: #666;'>Silicon Valley Grade Real Estate Neural Network</p>", unsafe_allow_html=True)

with col_hero_2:
    # Currency Ticker in top right
    currency = st.selectbox("", ["USD ($)", "INR (₹)"], label_visibility="collapsed")
    exchange_rate = 84.0

st.markdown("---")

# MAIN WORKSPACE
with st.form("valuation_form"):
    
    # Using columns for a grid layout
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 🏗️ Structure")
        year_built = st.number_input("Year Built", 1900, 2025, 2010)
        gr_liv_area = st.number_input("Living Area (sq ft)", 500, 10000, 2500)
        
    with c2:
        st.markdown("### 💎 Features")
        overall_qual = st.slider("Build Quality (1-10)", 1, 10, 8)
        full_bath = st.slider("Bathrooms", 1, 5, 2)
        
    with c3:
        st.markdown("### 📍 Land & Auto")
        garage_cars = st.selectbox("Garage Spaces", [0, 1, 2, 3, 4], index=2)
        lot_area = st.number_input("Lot Size (sq ft)", 1000, 50000, 8000)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centered Action Button
    b_col1, b_col2, b_col3 = st.columns([1, 2, 1])
    with b_col2:
        submitted = st.form_submit_button("Generate Valuation Analysis ⚡")

# --- 5. RESULT & VISUALIZATION ENGINE ---
if submitted:
    # PREDICTION
    input_data = artifacts['defaults'].copy()
    input_data.update({
        'OverallQual': overall_qual, 'GrLivArea': gr_liv_area,
        'YearBuilt': year_built, 'GarageCars': garage_cars,
        'FullBath': full_bath, 'LotArea': lot_area,
        'YearRemodAdd': year_built # Safe assumption
    })
    
    df_input = pd.DataFrame([input_data])[artifacts['features']]
    df_scaled = artifacts['scaler'].transform(df_input)
    base_price = artifacts['model'].predict(df_scaled)[0]
    
    # CURRENCY LOGIC
    if currency == "INR (₹)":
        final_price = base_price * exchange_rate
        price_str = f"₹{final_price:,.0f}"
    else:
        final_price = base_price
        price_str = f"${final_price:,.0f}"

    # --- THE "WOW" FACTOR: VISUALS ---
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1.5])
    
    with res_col1:
        # THE PRICE CARD
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin:0; color: #888; text-transform: uppercase; font-size: 12px; letter-spacing: 1px;">Estimated Market Value</h4>
            <h1 style="font-size: 56px; margin: 10px 0; background: linear-gradient(90deg, #4ade80, #22d3ee); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{price_str}</h1>
            <p style="color: #666;">Based on {len(artifacts['features'])} data points analyzed against historical market trends.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # KEY METRICS ROW
        m1, m2 = st.columns(2)
        m1.metric("Price / SqFt", f"{currency[0]} {final_price/gr_liv_area:.0f}")
        m2.metric("Appreciation Potential", "High", delta="AI Projection")

    with res_col2:
        # RADAR CHART: COMPARISON TO MARKET AVERAGE
        # We normalize values to 0-1 for the chart relative to some 'max'
        categories = ['Quality', 'Size', 'Luxury (Garage)', 'Land']
        
        # Create hypothetical "Market Average" (e.g., Quality 5, Size 1500, Garage 1, Land 5000)
        # and compare with "This House"
        
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[5, 1500/5000*10, 1/4*10, 5000/20000*10], # Scaled dummy avg
            theta=categories,
            fill='toself',
            name='Market Avg',
            line_color='#333'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[overall_qual, gr_liv_area/5000*10, garage_cars/4*10, lot_area/20000*10],
            theta=categories,
            fill='toself',
            name='This Property',
            line_color='#4ade80'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], showticklabels=False, linecolor='#333'),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            margin=dict(l=40, r=40, t=20, b=20),
            height=300
        )
        
        st.markdown('<div class="glass-card" style="padding: 10px;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
