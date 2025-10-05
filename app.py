# app.py — v2 (preprocess)

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, List

# -------------------------
# Page config & Styling
# -------------------------
st.set_page_config(page_title="Airbnb Price Model", page_icon="🏠", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
hr {margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(default_path: Path) -> pd.DataFrame:
    return pd.read_csv(default_path, low_memory=False)

def _clean_price(x):
    if pd.isna(x): return np.nan
    s = re.sub(r'[^0-9\.]', '', str(x))
    try: return float(s) if s else np.nan
    except: return np.nan

def _extract_baths(x):
    if pd.isna(x): return np.nan
    m = re.search(r'([\d\.]+)', str(x))
    return float(m.group(1)) if m else np.nan

def _amenity_count(x):
    if pd.isna(x): return 0.0
    s = str(x)
    if s.startswith('[') and s.endswith(']'):
        try:
            inner = s.strip()[1:-1]
            return float(len([t for t in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', inner) if t.strip()]))
        except:
            return float(s.count(',') + 1)
    return float(len([a for a in s.split(',') if a.strip()]))

def _bool01(x):
    if isinstance(x, str): return 1 if x.lower() in ('t','true','yes','y','1') else 0
    if isinstance(x, (int, float, bool)): return int(bool(x))
    return 0

# ---- Notebook-aligned price filtering helpers ----
def _iqr_trim(series: pd.Series, k: float = 1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (series >= lo) & (series <= hi)

def _valid_price_mask(df: pd.DataFrame) -> pd.Series:
    s = df["price_clean"]
    return s.notna() & np.isfinite(s) & (s > 0)

def build_feature_frame(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return model-ready X and numeric price target (cleaned + filtered to match notebook)."""
    df = df_raw.copy()

    # Price cleaning
    df['price_clean'] = df['price'].apply(_clean_price) if 'price' in df.columns else np.nan

    # === notebook parity: drop 0/negative + IQR trim on price ===
    if 'price_clean' in df.columns:
        mask_valid = _valid_price_mask(df)
        if mask_valid.any():
            mask_iqr = _iqr_trim(df.loc[mask_valid, 'price_clean'])
            keep_idx = df.loc[mask_valid].index[mask_iqr]
            df = df.loc[keep_idx].copy()

    # Bathrooms numeric
    if 'bathrooms_text' in df.columns:
        df['bathrooms'] = df['bathrooms_text'].apply(_extract_baths)
    elif 'bathrooms' in df.columns:
        df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
    else:
        df['bathrooms'] = np.nan

    # Amenity count
    df['amenity_count'] = df['amenities'].apply(_amenity_count) if 'amenities' in df.columns else np.nan

    # Binary flags
    df['host_is_superhost_01'] = df['host_is_superhost'].apply(_bool01) if 'host_is_superhost' in df.columns else 0
    df['instant_bookable_01'] = df['instant_bookable'].apply(_bool01) if 'instant_bookable' in df.columns else 0

    # To-numeric for continuous columns
    for col in ['bedrooms','accommodates','minimum_nights','maximum_nights',
                'review_scores_rating','number_of_reviews']:
        df[col] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else np.nan

    num_feats = [
        'bedrooms','bathrooms','accommodates',
        'minimum_nights','maximum_nights',
        'review_scores_rating','number_of_reviews',
        'amenity_count','host_is_superhost_01','instant_bookable_01'
    ]
    cat_feats = [c for c in ['neighbourhood_cleansed','room_type','property_type'] if c in df.columns]

    # Keep only existing columns
    num_feats = [c for c in num_feats if c in df.columns]
    cat_feats = [c for c in cat_feats if c in df.columns]

    X = df[num_feats + cat_feats].copy()
    y_price = df['price_clean'] if 'price_clean' in df.columns else pd.Series(np.nan, index=df.index)
    return X, y_price

# -------------------------
# Sidebar - dataset
# -------------------------
st.sidebar.title("Controls")
default_data_path = Path("data/listings_detailed.csv")

data_choice = st.sidebar.radio("Data source", ["Use bundled dataset", "Upload CSV"], index=0)
if data_choice == "Upload CSV":
    up = st.sidebar.file_uploader("Upload Airbnb-style CSV", type=["csv"])
    df_raw = pd.read_csv(up, low_memory=False) if up is not None else load_data(default_data_path)
else:
    df_raw = load_data(default_data_path)

st.title("Airbnb Price Prediction • XGBoost (log-price)")
st.success("Preprocessing functions added. Next commit will add model training.")
