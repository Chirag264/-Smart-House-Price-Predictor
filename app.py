"""
Smart House Price Predictor - Streamlit Web Application
A modern, beautiful, and beginner-friendly real estate valuation app
designed for home buyers, sellers, brokers, and non-technical users.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Optional interactive plotting
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM CSS & MODERN STYLING
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #2563EB 100%);
        padding: 32px 36px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
    }
    .hero-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #DBEAFE;
        margin-top: 10px;
        max-width: 820px;
        line-height: 1.5;
    }

    /* Card Containers */
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 18px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07);
    }
    .card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 14px;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 10px;
    }

    /* Primary Result Hero Card */
    .result-hero {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        color: white;
        padding: 32px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(67, 56, 202, 0.3);
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .result-label {
        font-size: 1.05rem;
        color: #C7D2FE;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-price {
        font-size: 3.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 8px 0;
        letter-spacing: -0.5px;
    }
    .result-subtext {
        font-size: 1.15rem;
        color: #E0E7FF;
        font-weight: 500;
    }
    .result-range {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        padding: 6px 18px;
        border-radius: 20px;
        margin-top: 14px;
        font-size: 0.92rem;
        color: #EEF2FF;
    }

    /* KPI Quick Stat Cards */
    .kpi-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }

    /* Preset Pill Buttons */
    .preset-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 8px;
    }

    /* Badge Tags */
    .badge-affordable {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-midrange {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-luxury {
        background-color: #FCE7F3;
        color: #9D174D;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Footer */
    .footer-bar {
        text-align: center;
        color: #94A3B8;
        font-size: 0.85rem;
        padding: 28px 0 10px 0;
        margin-top: 40px;
        border-top: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. MODEL & DATA LOADING
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'house_price_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Housing.csv')

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at: {MODEL_PATH}. Please train the model first."
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

@st.cache_data
def load_market_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

pipeline, load_error = load_trained_model()
market_df = load_market_data()

# Regional market indices (relative to national median benchmark)
CITY_MULTIPLIERS = {
    "National Baseline Benchmark": 1.00,
    "Mumbai (MMR)": 1.55,
    "Bengaluru (Silicon Valley)": 1.30,
    "Delhi NCR (Gurugram/Noida)": 1.25,
    "Hyderabad": 1.15,
    "Pune": 1.10,
    "Chennai": 1.05,
    "Kolkata": 0.95,
    "Ahmedabad": 0.90,
    "Tier-2 Cities (Jaipur, Lucknow, Indore)": 0.85
}

# ==========================================
# 4. INITIALIZE SESSION STATE / PRESETS
# ==========================================
if 'area' not in st.session_state:
    st.session_state['area'] = 4500
if 'bedrooms' not in st.session_state:
    st.session_state['bedrooms'] = 3
if 'bathrooms' not in st.session_state:
    st.session_state['bathrooms'] = 2
if 'stories' not in st.session_state:
    st.session_state['stories'] = 2
if 'parking' not in st.session_state:
    st.session_state['parking'] = 1
if 'airconditioning' not in st.session_state:
    st.session_state['airconditioning'] = "Yes"
if 'hotwaterheating' not in st.session_state:
    st.session_state['hotwaterheating'] = "No"
if 'mainroad' not in st.session_state:
    st.session_state['mainroad'] = "Yes"
if 'preferarea' not in st.session_state:
    st.session_state['preferarea'] = "Yes"
if 'furnishingstatus' not in st.session_state:
    st.session_state['furnishingstatus'] = "Semi-furnished"
if 'guestroom' not in st.session_state:
    st.session_state['guestroom'] = "No"
if 'basement' not in st.session_state:
    st.session_state['basement'] = "Yes"
if 'selected_city' not in st.session_state:
    st.session_state['selected_city'] = "National Baseline Benchmark"

def apply_preset(preset_name):
    if preset_name == "starter":
        st.session_state['area'] = 2200
        st.session_state['bedrooms'] = 2
        st.session_state['bathrooms'] = 1
        st.session_state['stories'] = 1
        st.session_state['parking'] = 0
        st.session_state['airconditioning'] = "No"
        st.session_state['hotwaterheating'] = "No"
        st.session_state['mainroad'] = "No"
        st.session_state['preferarea'] = "No"
        st.session_state['furnishingstatus'] = "Unfurnished"
        st.session_state['guestroom'] = "No"
        st.session_state['basement'] = "No"
    elif preset_name == "family":
        st.session_state['area'] = 4500
        st.session_state['bedrooms'] = 3
        st.session_state['bathrooms'] = 2
        st.session_state['stories'] = 2
        st.session_state['parking'] = 1
        st.session_state['airconditioning'] = "Yes"
        st.session_state['hotwaterheating'] = "No"
        st.session_state['mainroad'] = "Yes"
        st.session_state['preferarea'] = "Yes"
        st.session_state['furnishingstatus'] = "Semi-furnished"
        st.session_state['guestroom'] = "No"
        st.session_state['basement'] = "Yes"
    elif preset_name == "luxury":
        st.session_state['area'] = 8200
        st.session_state['bedrooms'] = 4
        st.session_state['bathrooms'] = 3
        st.session_state['stories'] = 3
        st.session_state['parking'] = 2
        st.session_state['airconditioning'] = "Yes"
        st.session_state['hotwaterheating'] = "Yes"
        st.session_state['mainroad'] = "Yes"
        st.session_state['preferarea'] = "Yes"
        st.session_state['furnishingstatus'] = "Furnished"
        st.session_state['guestroom'] = "Yes"
        st.session_state['basement'] = "Yes"

# ==========================================
# 5. HERO BANNER
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-tag">✨ AI-Powered Real Estate Valuation</div>
    <h1 class="hero-title">🏠 Smart House Price Predictor</h1>
    <p class="hero-subtitle">
        Planning to buy, sell, or evaluate a home? Get an instant, accurate market valuation 
        trained on 545 verified residential properties. No technical knowledge required!
    </p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ {load_error}")
    st.info("Run `python train_model.py` in your terminal to create the model file.")
    st.stop()

# ==========================================
# 6. ONE-CLICK PRESETS
# ==========================================
st.markdown('<div class="preset-title">⚡ Quick Templates (Click to Auto-Fill):</div>', unsafe_allow_html=True)
p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 1, 1])

with p_col1:
    if st.button("🏠 2 BHK Budget Flat", use_container_width=True, help="2,200 sq.ft, 2 Beds, 1 Bath, Unfurnished"):
        apply_preset("starter")
        st.rerun()

with p_col2:
    if st.button("🏡 3 BHK Family Home", use_container_width=True, help="4,500 sq.ft, 3 Beds, 2 Baths, AC, Semi-Furnished"):
        apply_preset("family")
        st.rerun()

with p_col3:
    if st.button("🏰 4 BHK Luxury Villa", use_container_width=True, help="8,200 sq.ft, 4 Beds, 3 Baths, AC, Furnished, Prime Area"):
        apply_preset("luxury")
        st.rerun()

with p_col4:
    if st.button("🔄 Reset Defaults", use_container_width=True):
        apply_preset("family")
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 7. USER INPUT FORM (3 ELEGANT CARDS)
# ==========================================
col_size, col_comfort, col_loc = st.columns(3)

# CARD 1: SIZE & LAYOUT
with col_size:
    st.markdown("""
    <div class="card-header">
        <span>📐</span> Size & Room Layout
    </div>
    """, unsafe_allow_html=True)
    
    area_input = st.number_input(
        "Total House Area (sq. ft)",
        min_value=1000,
        max_value=20000,
        value=int(st.session_state['area']),
        step=100,
        help="Overall built-up area in square feet. Typical homes are between 1,600 and 12,000 sq. ft."
    )
    
    bedrooms_input = st.slider(
        "🛏️ Bedrooms",
        min_value=1,
        max_value=6,
        value=int(st.session_state['bedrooms']),
        help="Total number of bedrooms in the property."
    )
    
    bathrooms_input = st.slider(
        "🛁 Bathrooms",
        min_value=1,
        max_value=4,
        value=int(st.session_state['bathrooms']),
        help="Number of attached and common bathrooms."
    )
    
    stories_input = st.slider(
        "🏢 Number of Floors / Stories",
        min_value=1,
        max_value=4,
        value=int(st.session_state['stories']),
        help="Levels or stories in the house (1 = Single Floor Bungalow, 2 = Duplex, etc.)."
    )

# CARD 2: COMFORT & AMENITIES
with col_comfort:
    st.markdown("""
    <div class="card-header">
        <span>❄️</span> Amenities & Comfort
    </div>
    """, unsafe_allow_html=True)
    
    parking_input = st.slider(
        "🚗 Covered Parking Spaces",
        min_value=0,
        max_value=3,
        value=int(st.session_state['parking']),
        help="Number of dedicated car parking spots."
    )
    
    ac_options = ["Yes", "No"]
    ac_idx = ac_options.index(st.session_state['airconditioning'])
    ac_input = st.selectbox(
        "❄️ Air Conditioning Installed?",
        options=ac_options,
        index=ac_idx,
        help="Does the house have central or split air conditioning?"
    )
    
    water_options = ["Yes", "No"]
    water_idx = water_options.index(st.session_state['hotwaterheating'])
    water_input = st.selectbox(
        "♨️ Hot Water Heating?",
        options=water_options,
        index=water_idx,
        help="Includes dedicated geyser or central boiler heating systems."
    )
    
    guest_options = ["Yes", "No"]
    guest_idx = guest_options.index(st.session_state['guestroom'])
    guest_input = st.selectbox(
        "🛏️ Dedicated Guest Room?",
        options=guest_options,
        index=guest_idx,
        help="Has an extra bedroom specifically allocated for visiting guests."
    )

# CARD 3: LOCATION & CITY ADJUSTMENT
with col_loc:
    st.markdown("""
    <div class="card-header">
        <span>📍</span> Location & Interior
    </div>
    """, unsafe_allow_html=True)
    
    city_list = list(CITY_MULTIPLIERS.keys())
    city_idx = city_list.index(st.session_state['selected_city'])
    selected_city = st.selectbox(
        "🏙️ City / Metropolitan Region",
        options=city_list,
        index=city_idx,
        help="Adjusts valuation based on regional real estate price indices."
    )
    st.session_state['selected_city'] = selected_city
    
    furnishing_options = ["Furnished", "Semi-furnished", "Unfurnished"]
    furnish_idx = furnishing_options.index(st.session_state['furnishingstatus'])
    furnishing_input = st.selectbox(
        "🛋️ Furnishing Level",
        options=furnishing_options,
        index=furnish_idx,
        help="Fully furnished (ready to move), semi-furnished (wardrobes/kitchen), or unfurnished."
    )
    
    pref_options = ["Yes", "No"]
    pref_idx = pref_options.index(st.session_state['preferarea'])
    prefer_input = st.selectbox(
        "🌟 Located in Prime / Preferred Area?",
        options=pref_options,
        index=pref_idx,
        help="Is the property situated in a top-rated, high-demand neighborhood?"
    )
    
    road_options = ["Yes", "No"]
    road_idx = road_options.index(st.session_state['mainroad'])
    road_input = st.selectbox(
        "🛣️ Direct Main Road Access?",
        options=road_options,
        index=road_idx,
        help="Property connects directly to a primary thoroughfare or main street."
    )
    
    basement_options = ["Yes", "No"]
    basement_idx = basement_options.index(st.session_state['basement'])
    basement_input = st.selectbox(
        "🏠 Finished Basement?",
        options=basement_options,
        index=basement_idx,
        help="Includes an underground basement suitable for storage or living."
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 8. PREDICT BUTTON
# ==========================================
predict_clicked = st.button("🔮 Calculate Estimated House Price", type="primary", use_container_width=True)

# Prepare DataFrame matching pipeline
current_input_df = pd.DataFrame([{
    'area': area_input,
    'bedrooms': bedrooms_input,
    'bathrooms': bathrooms_input,
    'stories': stories_input,
    'mainroad': road_input,
    'guestroom': guest_input,
    'basement': basement_input,
    'hotwaterheating': water_input,
    'airconditioning': ac_input,
    'parking': parking_input,
    'preferarea': prefer_input,
    'furnishingstatus': furnishing_input
}])

# ==========================================
# 9. VALUATION RESULTS & INSIGHTS
# ==========================================
if predict_clicked:
    try:
        raw_pred = pipeline.predict(current_input_df)[0]
        city_factor = CITY_MULTIPLIERS.get(selected_city, 1.0)
        price = max(0.0, float(raw_pred) * city_factor)
        
        # Financial formatting
        price_lakhs = price / 100_000
        price_crores = price / 10_000_000
        price_per_sqft = price / area_input
        
        # Market range estimate (±8% confidence window)
        price_min = price * 0.92
        price_max = price * 1.08
        
        # Categorization
        if price < 4_000_000:
            tier_badge = '<span class="badge-affordable">🏷️ Affordable Starter Segment</span>'
            tier_name = "Affordable Housing"
        elif price < 8_500_000:
            tier_badge = '<span class="badge-midrange">🏡 Mid-Range Family Segment</span>'
            tier_name = "Mid-Range Residential"
        else:
            tier_badge = '<span class="badge-luxury">🏰 Premium Luxury Segment</span>'
            tier_name = "Luxury Villa / Penthouse"
            
        # Display Result Hero Card
        st.markdown(f"""
        <div class="result-hero">
            <div class="result-label">Estimated Fair Market Valuation ({selected_city.split()[0]})</div>
            <div class="result-price">₹ {price:,.0f}</div>
            <div class="result-subtext">
                <b>₹ {price_lakhs:.2f} Lakhs</b> &bull; (₹ {price_crores:.2f} Crore) &bull; {tier_badge}
            </div>
            <div class="result-range">
                📊 Realistic Range: <b>₹ {price_min/100000:.1f} Lakhs</b> &mdash; <b>₹ {price_max/100000:.1f} Lakhs</b> (based on local market variance)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # KPI Badges
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Price per Sq. Ft</div>
                <div class="kpi-value">₹ {price_per_sqft:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Air Conditioning</div>
                <div class="kpi-value">{"✅ Included" if ac_input == "Yes" else "❌ None"}</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Furnishing Level</div>
                <div class="kpi-value">{furnishing_input}</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">City Index</div>
                <div class="kpi-value">{selected_city.split('(')[0].strip()} ({city_factor:.2f}x)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ==========================================
        # 10. PLOTLY VISUALIZATIONS (GAUGE & COMPARISON)
        # ==========================================
        if HAS_PLOTLY:
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode='gauge+number',
                    value=price_lakhs,
                    number={'suffix': ' L', 'valueformat': '.1f', 'font': {'size': 32, 'color': '#1E1B4B'}},
                    title={'text': 'Valuation Spectrum (in ₹ Lakhs)', 'font': {'size': 15, 'color': '#475569'}},
                    gauge={
                        'axis': {'range': [15, 160], 'tickwidth': 1, 'tickcolor': '#94A3B8'},
                        'bar': {'color': '#4F46E5', 'thickness': 0.3},
                        'bgcolor': 'white',
                        'borderwidth': 1,
                        'bordercolor': '#CBD5E1',
                        'steps': [
                            {'range': [15, 45], 'color': '#DCFCE7'},
                            {'range': [45, 90], 'color': '#FEF9C3'},
                            {'range': [90, 160], 'color': '#FCE7F3'}
                        ],
                        'threshold': {
                            'line': {'color': '#1E1B4B', 'width': 4},
                            'thickness': 0.75,
                            'value': price_lakhs
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=240,
                    margin=dict(l=20, r=20, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with chart_col2:
                if market_df is not None:
                    market_lakhs = market_df['price'] / 100_000
                    percentile = (market_df['price'] < (price / city_factor)).mean() * 100.0
                    
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=market_lakhs,
                        nbinsx=25,
                        name='Market Sales',
                        marker_color='#93C5FD',
                        opacity=0.75
                    ))
                    fig_dist.add_vline(
                        x=price_lakhs / city_factor,
                        line_width=3,
                        line_dash='dash',
                        line_color='#DC2626',
                        annotation_text=f'Your Home: Top {100-percentile:.0f}%',
                        annotation_position='top right',
                        annotation_font_color='#DC2626',
                        annotation_font_size=11
                    )
                    fig_dist.update_layout(
                        title=f'Where You Rank Among 545 Sales (Top {100-percentile:.0f}%)',
                        title_font_size=14,
                        xaxis_title='Price (₹ Lakhs)',
                        yaxis_title='Properties',
                        height=240,
                        margin=dict(l=20, r=20, t=35, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

        # ==========================================
        # 11. MORTGAGE & EMI CALCULATOR (EXPANDABLE)
        # ==========================================
        with st.expander("💳 Home Loan & Monthly EMI Calculator (Plan Your Budget)"):
            st.write("Estimate your monthly bank installment based on this property price:")
            
            emi_col1, emi_col2, emi_col3 = st.columns(3)
            with emi_col1:
                down_payment_pct = st.slider("Down Payment (%)", 10, 50, 20, step=5)
            with emi_col2:
                interest_rate = st.slider("Bank Interest Rate (% per annum)", 6.5, 12.0, 8.5, step=0.25)
            with emi_col3:
                loan_tenure_years = st.slider("Loan Tenure (Years)", 5, 30, 20, step=5)

            down_payment_amt = price * (down_payment_pct / 100.0)
            principal = price - down_payment_amt
            monthly_rate = (interest_rate / 100.0) / 12.0
            num_months = loan_tenure_years * 12
            
            if monthly_rate > 0:
                monthly_emi = (principal * monthly_rate * ((1 + monthly_rate) ** num_months)) / (((1 + monthly_rate) ** num_months) - 1)
            else:
                monthly_emi = principal / num_months
                
            total_payable = (monthly_emi * num_months) + down_payment_amt
            total_interest = (monthly_emi * num_months) - principal

            e_res1, e_res2, e_res3 = st.columns(3)
            with e_res1:
                st.metric("Estimated Monthly EMI", f"₹ {monthly_emi:,.0f} / mo")
            with e_res2:
                st.metric("Down Payment Required", f"₹ {down_payment_amt:,.0f}")
            with e_res3:
                st.metric("Total Interest Payable", f"₹ {total_interest:,.0f}")

        # ==========================================
        # 12. WHAT-IF ANALYSIS (UPGRADE VALUE EXPLORER)
        # ==========================================
        with st.expander("💡 Value Upgrade Simulator: See What Adds the Most Value"):
            st.write("Curious how specific home improvements would change this valuation?")
            
            # Scenario 1: AC Toggle
            alt_ac = "No" if ac_input == "Yes" else "Yes"
            df_alt_ac = current_input_df.copy()
            df_alt_ac['airconditioning'] = alt_ac
            price_alt_ac = pipeline.predict(df_alt_ac)[0] * city_factor
            diff_ac = price_alt_ac - price
            sign_ac = "+" if diff_ac > 0 else "-"
            st.markdown(f"- **Toggling Air Conditioning ({alt_ac})**: New Price: **₹ {price_alt_ac:,.0f}** ({sign_ac}₹ {abs(diff_ac):,.0f})")
            
            # Scenario 2: Furnishing Toggle
            if furnishing_input != "Furnished":
                df_furn = current_input_df.copy()
                df_furn['furnishingstatus'] = "Furnished"
                price_furn = pipeline.predict(df_furn)[0] * city_factor
                diff_f = price_furn - price
                st.markdown(f"- **Upgrading to Fully Furnished**: New Price: **₹ {price_furn:,.0f}** (+₹ {diff_f:,.0f})")
            else:
                df_unfurn = current_input_df.copy()
                df_unfurn['furnishingstatus'] = "Unfurnished"
                price_unfurn = pipeline.predict(df_unfurn)[0] * city_factor
                diff_u = price_unfurn - price
                st.markdown(f"- **Selling as Unfurnished**: New Price: **₹ {price_unfurn:,.0f}** (-₹ {abs(diff_u):,.0f})")
                
            # Scenario 3: Preferred Area Toggle
            alt_pref = "No" if prefer_input == "Yes" else "Yes"
            df_alt_pref = current_input_df.copy()
            df_alt_pref['preferarea'] = alt_pref
            price_alt_pref = pipeline.predict(df_alt_pref)[0] * city_factor
            diff_pref = price_alt_pref - price
            sign_p = "+" if diff_pref > 0 else "-"
            st.markdown(f"- **Relocating to {alt_pref} Preferred Area**: New Price: **₹ {price_alt_pref:,.0f}** ({sign_p}₹ {abs(diff_pref):,.0f})")

        # ==========================================
        # 13. AI MODEL TRANSPARENCY & FEATURE IMPORTANCE
        # ==========================================
        with st.expander("📊 AI Model Transparency: What Factors Influence Price the Most?"):
            st.write("Our Gradient Boosting AI algorithm analyzes features and automatically assigns importance based on market patterns:")
            
            try:
                preproc = pipeline.named_steps['preprocessor']
                est_model = pipeline.named_steps['model']
                r_names = preproc.get_feature_names_out()
                c_names = [n.replace('num__', '').replace('cat__', '') for n in r_names]
                imps = est_model.feature_importances_
                
                df_fi = pd.DataFrame({'Feature': c_names, 'Importance': imps}).sort_values('Importance', ascending=True).tail(10)
                
                if HAS_PLOTLY:
                    fig_fi = go.Figure(go.Bar(
                        x=df_fi['Importance'] * 100,
                        y=df_fi['Feature'],
                        orientation='h',
                        marker_color='#3B82F6',
                        text=(df_fi['Importance'] * 100).map('{:.1f}%'.format),
                        textposition='auto'
                    ))
                    fig_fi.update_layout(
                        title='Top 10 Factors Influencing House Valuation (% Weight)',
                        xaxis_title='Relative Importance (%)',
                        height=300,
                        margin=dict(l=10, r=20, t=35, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.dataframe(df_fi.sort_values('Importance', ascending=False), use_container_width=True)
                    
            except Exception as e:
                st.caption(f"Feature importance display note: {e}")

        # ==========================================
        # 14. DOWNLOAD VALUATION RECEIPT
        # ==========================================
        receipt_text = f"""======================================================
🏠 SMART HOUSE PRICE PREDICTOR - VALUATION REPORT
======================================================
City / Region:         {selected_city}
Estimated Market Value: ₹ {price:,.0f} ({price_lakhs:.2f} Lakhs / ₹ {price_crores:.2f} Crore)
Market Segment:        {tier_name}
Estimated Price Range: ₹ {price_min:,.0f} - ₹ {price_max:,.0f}
Price per Sq. Ft:      ₹ {price_per_sqft:,.0f}

--- PROPERTY SPECIFICATIONS ---
Total Area:            {area_input:,} sq. ft
Bedrooms:              {bedrooms_input}
Bathrooms:             {bathrooms_input}
Stories:               {stories_input}
Parking Slots:         {parking_input}
Air Conditioning:      {ac_input}
Hot Water Heating:     {water_input}
Main Road Access:      {road_input}
Preferred Area:        {prefer_input}
Furnishing Status:     {furnishing_input}
Guest Room:            {guest_input}
Basement:              {basement_input}

--- FINANCING ESTIMATE (8.5% for 20 Years) ---
Monthly EMI:           ₹ {monthly_emi:,.0f} / month
Down Payment ({down_payment_pct}%):    ₹ {down_payment_amt:,.0f}
======================================================
Generated via Antigravity ML Project
"""
        st.download_button(
            label="📥 Download Valuation Summary (TXT)",
            data=receipt_text,
            file_name=f"house_valuation_{int(price)}.txt",
            mime="text/plain",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ Error during price estimation: {str(e)}")

# ==========================================
# 15. SIMPLE NON-TECHNICAL FAQ
# ==========================================
st.markdown("---")
with st.expander("❓ Frequently Asked Questions (FAQ for Home Buyers & Sellers)"):
    st.markdown("""
    **Q1: How does this calculator decide the house price?**  
    Our AI model analyzed 545 real house sales. It looked at the relationship between house size, number of rooms, and amenities to find the market value formula automatically.
    
    **Q2: Why does square footage (area) affect price the most?**  
    In real estate, land and construction costs depend directly on total square footage. An extra 500 sq. ft adds substantial value compared to minor cosmetic changes.
    
    **Q3: Can I use this for official bank loans?**  
    This tool gives an excellent estimated market price benchmark. For official home loans and government stamp duty registration, always get a formal inspection from a licensed civil surveyor.
    """)

# ==========================================
# 16. FOOTER
# ==========================================
st.markdown("""
<div class="footer-bar">
    🏠 <b>Smart House Price Predictor</b> &bull; Built with Python &bull; Scikit-learn &bull; Streamlit &bull; Plotly<br>
    Designed for beginners, home buyers, and real estate professionals.
</div>
""", unsafe_allow_html=True)
