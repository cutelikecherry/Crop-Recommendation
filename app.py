import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Crop Advisory & Soil Deficit Engine",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts = load_pipeline()
    model = artifacts["model"]
    feature_names = artifacts["feature_names"]
    crop_list = artifacts["crop_list"]
    crop_profiles = artifacts["crop_profiles"]
except FileNotFoundError:
    st.error("⚠️ 'model.pkl' not found. Please run 'python train.py' first.")
    st.stop()


st.title("🌱 Precision Agriculture Crop & Soil Deficit Engine")
st.markdown("Predict optimal crop suitability or compute specific soil nutrient adjustments required for your target yield.")

st.sidebar.header("📋 Soil & Climate Parameters")

# Inputs
n_val = st.sidebar.slider("Nitrogen (N) - mg/kg", 0, 140, 50)
p_val = st.sidebar.slider("Phosphorus (P) - mg/kg", 5, 145, 50)
k_val = st.sidebar.slider("Potassium (K) - mg/kg", 5, 205, 50)
ph_val = st.sidebar.slider("Soil pH Level", 3.5, 10.0, 6.5, step=0.1)

st.sidebar.subheader("🌤️ Environmental Factors")
temp_val = st.sidebar.slider("Temperature (°C)", 8.0, 50.0, 25.0, step=0.5)
hum_val = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 70.0, step=1.0)
rain_val = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0, step=5.0)

input_data = pd.DataFrame([[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]], columns=feature_names)

pred_crop = model.predict(input_data)[0]
pred_probs = model.predict_proba(input_data)[0]
confidence = np.max(pred_probs) * 100

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎯 Recommended Crop Prediction")
    st.success(f"**Optimal Crop Choice:** {pred_crop.capitalize()}")
    st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
    
    st.markdown("#### Top Probable Alternatives:")
    top_indices = np.argsort(pred_probs)[::-1][1:4]
    for idx in top_indices:
        st.write(f"- **{model.classes_[idx].capitalize()}**: {pred_probs[idx]*100:.1f}% probability")


CURRENCY_CONFIG = {
    "USD ($)": {"symbol": "$", "rate": 1.0, "urea": 0.40, "dap": 0.70, "mop": 0.55, "lime": 0.15},
    "INR (₹)": {"symbol": "₹", "rate": 83.5, "urea": 33.40, "dap": 58.45, "mop": 45.92, "lime": 12.52},
    "EUR (€)": {"symbol": "€", "rate": 0.92, "urea": 0.37, "dap": 0.64, "mop": 0.51, "lime": 0.14},
    "GBP (£)": {"symbol": "£", "rate": 0.79, "urea": 0.32, "dap": 0.55, "mop": 0.43, "lime": 0.12},
    "BRL (R$)": {"symbol": "R$", "rate": 5.15, "urea": 2.06, "dap": 3.61, "mop": 2.83, "lime": 0.77}
}

#Feature: Soil Nutrient Deficit & Multi-Currency ROI Simulator Engine
with col_right:
    st.subheader("🧪 Deficit & Multi-Currency ROI Simulator")
    
    selected_currency_key = st.selectbox("🌐 Select Market Currency:", list(CURRENCY_CONFIG.keys()), index=0)
    curr_info = CURRENCY_CONFIG[selected_currency_key]
    sym = curr_info["symbol"]

    target_crop = st.selectbox("Select Target Crop for Custom Prescription:", crop_list, index=crop_list.index(pred_crop))
    
    if target_crop in crop_profiles:
        target_stats = crop_profiles[target_crop]
        
        #Calculate Nutrient Deltas
        delta_n = max(0, target_stats['N'] - n_val)
        delta_p = max(0, target_stats['P'] - p_val)
        delta_k = max(0, target_stats['K'] - k_val)
        delta_ph = target_stats['ph'] - ph_val

        #Convert Deficits to Commercial Fertilizer Quantities (kg/ha)
        urea_kg = (delta_n * 2.17) * 2
        dap_kg = (delta_p * 2.17) * 2
        mop_kg = (delta_k * 1.66) * 2
        lime_kg = abs(delta_ph * 500) if delta_ph > 0.5 else 0

        #Dynamic Cost Calculation based on Selected Currency Rates
        urea_cost = urea_kg * curr_info["urea"]
        dap_cost = dap_kg * curr_info["dap"]
        mop_cost = mop_kg * curr_info["mop"]
        lime_cost = lime_kg * curr_info["lime"]
        
        total_cost = urea_cost + dap_cost + mop_cost + lime_cost

        #Display Nutrient Deficit Metrics
        st.markdown(f"#### Required Soil Adjustment for **{target_crop.capitalize()}**:")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Nitrogen (N)", f"{target_stats['N']:.0f}", delta=f"{target_stats['N'] - n_val:+.0f} mg/kg")
        m2.metric("Phosphorus (P)", f"{target_stats['P']:.0f}", delta=f"{target_stats['P'] - p_val:+.0f} mg/kg")
        m3.metric("Potassium (K)", f"{target_stats['K']:.0f}", delta=f"{target_stats['K'] - k_val:+.0f} mg/kg")
        m4.metric("pH Level", f"{target_stats['ph']:.1f}", delta=f"{delta_ph:+.1f}")

        #Dynamic Financial Feasibility Output
        st.markdown("#### 💰 Economic Remediation Prescription")
        
        c1, c2 = st.columns(2)
        c1.metric(f"Estimated Input Overhead / Ha", f"{sym}{total_cost:,.2f}")
        
        #Financial Feasibility Thresholds (Normalized in USD terms)
        usd_equivalent = total_cost / curr_info["rate"]
        if usd_equivalent == 0:
            c2.success("FEASIBILITY: Ideal (Zero Overhead)")
        elif usd_equivalent < 75:
            c2.success("FEASIBILITY: Highly Viable")
        elif usd_equivalent < 175:
            c2.warning("FEASIBILITY: Moderate Investment")
        else:
            c2.error("FEASIBILITY: High Investment Risk")

        #Detailed Localized Cost Table
        fertilizer_table = {
            "Commercial Input": ["Urea (Nitrogen)", "DAP (Phosphorus)", "MOP (Potassium)", "Ag Lime (pH Fix)"],
            "Quantity Needed": [f"{urea_kg:.1f} kg/ha", f"{dap_kg:.1f} kg/ha", f"{mop_kg:.1f} kg/ha", f"{lime_kg:.1f} kg/ha"],
            f"Cost ({selected_currency_key.split()[0]})": [f"{sym}{urea_cost:,.2f}", f"{sym}{dap_cost:,.2f}", f"{sym}{mop_cost:,.2f}", f"{sym}{lime_cost:,.2f}"]
        }
        st.dataframe(pd.DataFrame(fertilizer_table), hide_index=True, use_container_width=True)

        if usd_equivalent > 150 and target_crop != pred_crop:
            st.caption(f"💡 **Overhead Saving Tip:** Switching to **{pred_crop.capitalize()}** eliminates most of this remediation expense.")

st.caption("Deployment Target: Streamlit Community Cloud | Technology Readiness Level: TRL 4 (Lab Validated)")