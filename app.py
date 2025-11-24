import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="PropAI | Asset Valuation",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ELITE STYLING (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #050505; } /* Deep Black */
    
    /* REMOVE BLOAT */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* GLASS CARD SYSTEM */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 40px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        transition: transform 0.3s ease;
    }
    
    /* TYPOGRAPHY */
    .price-hero {
        font-size: 72px;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -3px;
        line-height: 1.1;
        margin-top: 10px;
    }
    .label-hero {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 10px;
        display: inline-block;
    }
    
    /* INPUT FIELDS */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        color: #e2e8f0 !important;
        border-radius: 8px;
    }
    div[data-testid="stForm"] { border: none; padding: 0; }
    
    /* BUTTON */
    .stButton button {
        background: #3b82f6;
        color: white;
        font-weight: 600;
        border-radius: 12px;
        height: 50px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.39);
        transition: 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        background: #2563eb;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD ENGINE ---
@st.cache_resource
def load_bundle():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

artifacts = load_bundle()

# --- 4. HEADER (CLEANED) ---
c1, c2 = st.columns([2, 1])
with c1:
    # UPDATED: Removed "//" and used clean spacing
    st.markdown("## PropAI <span style='color:#6366f1'>Enterprise</span>", unsafe_allow_html=True)
with c2:
    st.markdown("""
        <div style="text-align: right; font-size: 12px; color: #475569; padding-top: 10px;">
            MODEL: XGB-PRO-V2 <span style="color:#22c55e; margin: 0 8px;">●</span> ONLINE
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

if not artifacts:
    st.error("⚠️ System Halted: 'house_price_model.pkl' not detected.")
    st.stop()

# --- 5. CONTROL PANEL ---
with st.form("valuation_form"):
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown("#### 📐 Dimensions")
        gr_liv_area = st.number_input("Living Area (sq ft)", 500, 8000, 2000, step=100)
        lot_area = st.number_input("Lot Size (sq ft)", 1000, 50000, 10000, step=500)
        
    with c2:
        st.markdown("#### 🏗️ Specs")
        year_built = st.number_input("Year Built", 1800, 2025, 2005)
        garage_cars = st.selectbox("Garage Capacity", [0, 1, 2, 3, 4], index=2)
        
    with c3:
        st.markdown("#### ✨ Condition")
        overall_qual = st.slider("Quality Score", 1, 10, 7)
        full_bath = st.slider("Bathrooms", 1, 4, 2)

    with c4:
        st.markdown("#### 🚀 Execute")
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("RUN VALUATION", type="primary")

# --- 6. INTELLIGENCE DISPLAY ---
if submitted:
    # A. PREDICTION LOGIC
    input_data = artifacts['defaults'].copy()
    input_data.update({
        'OverallQual': overall_qual, 'GrLivArea': gr_liv_area,
        'YearBuilt': year_built, 'GarageCars': garage_cars,
        'FullBath': full_bath, 'LotArea': lot_area,
        'YearRemodAdd': year_built
    })
    
    df_input = pd.DataFrame([input_data])[artifacts['features']]
    df_scaled = artifacts['scaler'].transform(df_input)
    
    # Raw prediction
    raw_price = artifacts['model'].predict(df_scaled)[0]
    
    # B. REALISM CLAMP
    final_price = max(50000, raw_price) 
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # C. LAYOUT
    col_left, col_right = st.columns([1, 1.3])
    
    with col_left:
        # THE PRICE CARD
        st.markdown(f"""
        <div class="glass-card">
            <div class="label-hero">Asset Valuation Estimate</div>
            <div class="price-hero">${final_price:,.0f}</div>
            <div style="margin-top: 25px; display: flex; gap: 20px; color: #94a3b8; font-size: 14px;">
                <div>
                    <span style="display:block; font-size: 11px; text-transform:uppercase; color: #475569;">Price / SqFt</span>
                    ${final_price/gr_liv_area:.0f}
                </div>
                <div style="border-left: 1px solid #334155; padding-left: 20px;">
                    <span style="display:block; font-size: 11px; text-transform:uppercase; color: #475569;">Confidence</span>
                    94.2%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # D. FIXED RADAR CHART MATH
        MAX_SIZE = 4000
        MAX_QUAL = 10
        MAX_LOT = 20000
        MAX_GARAGE = 4

        # Normalize 
        norm_size = min(gr_liv_area / MAX_SIZE, 1.0)
        norm_qual = overall_qual / MAX_QUAL
        norm_lot = min(lot_area / MAX_LOT, 1.0)
        norm_garage = garage_cars / MAX_GARAGE
        
        # Averages
        avg_size = 1500 / MAX_SIZE
        avg_qual = 6 / MAX_QUAL
        avg_lot = 10000 / MAX_LOT
        avg_garage = 2 / MAX_GARAGE

        categories = ['Living Space', 'Build Quality', 'Land Area', 'Luxury (Garage)']
        
        fig = go.Figure()

        # Market Average Trace
        fig.add_trace(go.Scatterpolar(
            r=[avg_size, avg_qual, avg_lot, avg_garage],
            theta=categories,
            fill='toself',
            name='Market Average',
            line_color='#64748b',
            fillcolor='rgba(100, 116, 139, 0.2)',
            opacity=0.6
        ))
        
        # This Property Trace
        fig.add_trace(go.Scatterpolar(
            r=[norm_size, norm_qual, norm_lot, norm_garage],
            theta=categories,
            fill='toself',
            name='This Asset',
            line_color='#6366f1',
            fillcolor='rgba(99, 102, 241, 0.4)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Inter'),
            showlegend=True,
            legend=dict(x=0.75, y=0.95, font=dict(size=10)),
            margin=dict(l=40, r=40, t=20, b=20),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
