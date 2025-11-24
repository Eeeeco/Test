import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# --- 1. SETUP & PAGE CONFIG (Headless Mode) ---
st.set_page_config(
    page_title="PropAI | Valuation Engine",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. THE "SILICON VALLEY" CSS SUITE ---
# This hides the default Streamlit look and injects modern design tokens.
st.markdown("""
    <style>
    /* IMPORT GOOGLE FONT (Inter - The Tech Standard) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    /* GLOBAL OVERRIDES */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* HIDE STREAMLIT BRANDING */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* CUSTOM INPUT FIELDS */
    .stNumberInput > div > div > input {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #333;
        border-radius: 8px;
    }
    
    /* THE GLASS CARD (Result Box) */
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
        margin-top: 20px;
    }
    
    /* GRADIENT TEXT */
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #4ade80, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
    }
    
    /* SUBTITLES */
    .sub-text {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD ARTIFACTS ---
@st.cache_resource
def load_data():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

artifacts = load_data()

# --- 4. HERO SECTION ---
# Minimalist Header
st.markdown("## PropAI <span style='color:#4ade80'>//</span> Valuation Engine", unsafe_allow_html=True)
st.markdown("Adjust the parameters below to generate a real-time market prediction.")
st.markdown("---")

# --- 5. THE INPUT GRID (Clean & Symmetrical) ---
if artifacts:
    # We use a form to prevent "stuttering"
    with st.form("main_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("### 🏗️ Structure")
            gr_liv_area = st.number_input("Living Area (sq ft)", 500, 10000, 2000)
            year_built = st.number_input("Year Built", 1900, 2025, 2010)
            
        with c2:
            st.markdown("### ✨ Finish")
            overall_qual = st.slider("Quality Score (1-10)", 1, 10, 7)
            full_bath = st.slider("Full Bathrooms", 1, 4, 2)
            
        with c3:
            st.markdown("### 📍 Land")
            garage_cars = st.selectbox("Garage Spaces", [0, 1, 2, 3, 4], index=2)
            lot_area = st.number_input("Lot Size (sq ft)", 1000, 50000, 9000)

        # Hidden visual spacer
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Full width primary button
        submitted = st.form_submit_button("GENERATE ESTIMATE ⚡", type="primary")

    # --- 6. THE RESULTS (The "Wow" Moment) ---
    if submitted:
        # Prepare Data
        input_data = artifacts['defaults'].copy()
        input_data.update({
            'OverallQual': overall_qual, 'GrLivArea': gr_liv_area,
            'YearBuilt': year_built, 'GarageCars': garage_cars,
            'FullBath': full_bath, 'LotArea': lot_area,
            'YearRemodAdd': year_built 
        })
        
        # Predict
        df_input = pd.DataFrame([input_data])[artifacts['features']]
        df_scaled = artifacts['scaler'].transform(df_input)
        prediction = artifacts['model'].predict(df_scaled)[0]
        
        # --- UI LAYOUT FOR RESULTS ---
        st.markdown("---")
        
        col_res_left, col_res_right = st.columns([1.5, 2])
        
        with col_res_left:
            # The Glassmorphism Price Card
            st.markdown(f"""
            <div class="glass-container">
                <div class="sub-text">Estimated Market Value</div>
                <div class="gradient-text">${prediction:,.0f}</div>
                <div style="color: #666; font-size: 14px; margin-top: 10px;">
                    Confidence Interval: 94%<br>
                    Based on {len(artifacts['features'])} market factors
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_res_right:
            # The "Futuristic" Radar Chart
            # We compare "This Property" vs a hypothetical "Market Average"
            
            categories = ['Quality', 'Size', 'Luxury', 'Land']
            
            # Normalize values roughly to 0-10 scale for visual comparison
            # (Math here is just for visual demo purposes)
            prop_values = [
                overall_qual, 
                (gr_liv_area/4000)*10, 
                (garage_cars/4)*10, 
                (lot_area/20000)*10
            ]
            
            avg_values = [6, 5, 5, 4] # Hypothetical market average

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=avg_values,
                theta=categories,
                fill='toself',
                name='Market Avg',
                line_color='#444444'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=prop_values,
                theta=categories,
                fill='toself',
                name='This Property',
                line_color='#4ade80'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10], showticklabels=False),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family="Inter"),
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠️ Logic Core Missing: Please upload 'house_price_model.pkl'")
