# app.py — v1 (scaffold)

import streamlit as st
import pandas as pd
from pathlib import Path

# -------------------------
# Page config & Styling
# -------------------------
st.set_page_config(
    page_title="Airbnb Price Model",
    page_icon="🏠",
    layout="wide"
)

# Minimal CSS for spacing (can be enhanced later)
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
hr {margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Simple Header
# -------------------------
st.title("Airbnb Price Prediction • XGBoost (log-price)")
st.caption("Scaffold: page config + basic structure. Subsequent commits will add preprocessing, model, and UI.")
