# =========================
# XGBoost Regressor with log(price) target
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, List, Dict

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, median_absolute_error, r2_score,
    mean_squared_error
)
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page config & Styling
# -------------------------
st.set_page_config(
    page_title="Airbnb Price Model",
    page_icon="🏠",
    layout="wide"
)

# Load Font Awesome icons
ICONS = """
<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
"""
st.markdown(ICONS, unsafe_allow_html=True)

# Custom CSS (tabs visible on dark theme, gray gradient metric cards)
CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}

/* ===== Metric cards ===== */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, #f6f7f9 0%, #eef1f5 100%);
    border-radius: 14px;
    padding: 18px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    text-align: center;
}
[data-testid="stMetricValue"] {
    color: #0f1115 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #4a4f57 !important;
    font-size: 0.95rem !important;
}

/* ===== Tabs: high-contrast labels + active underline ===== */
.stTabs { padding-top: 0.25rem; border-bottom: 1px solid rgba(255,255,255,0.08); }
.stTabs [role="tablist"] { gap: 0.75rem; }
.stTabs [data-baseweb="tab"] {
    font-weight: 600; color: #cfd3dc !important; background: transparent; border: none; opacity: 1 !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #ffffff !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important; border-bottom: 2px solid rgba(255,255,255,0.35);
}

/* Light theme overrides (optional) */
body[data-theme="light"] .stTabs [data-baseweb="tab"] { color:#333 !important; }
body[data-theme="light"] .stTabs [data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid rgba(0,0,0,0.25);
}

/* ===== General ===== */
hr {margin: 1rem 0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown("<style>h1 { margin-bottom: 0.5rem !important; }</style>", unsafe_allow_html=True)

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
    # robust split even if quoted commas exist
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

    # --- match notebook behavior (remove zeros/negatives and IQR-trim price) ---
    if 'price_clean' in df.columns:
        mask_valid = _valid_price_mask(df)
        if mask_valid.any():
            mask_iqr = _iqr_trim(df.loc[mask_valid, 'price_clean'])
            keep_idx = df.loc[mask_valid].index[mask_iqr]
            df = df.loc[keep_idx].copy()
    # ---------------------------------------------------------------------------------

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

    # To-numeric for key continuous columns
    for col in ['bedrooms', 'accommodates', 'minimum_nights', 'maximum_nights',
                'review_scores_rating', 'number_of_reviews']:
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

def make_pipeline(X: pd.DataFrame,
                  add_scaler: bool,
                  xgb_params: Dict) -> Tuple[Pipeline, List[str], List[str]]:
    numeric_features = [c for c in X.columns if X[c].dtype != 'object']
    categorical_features = [c for c in X.columns if X[c].dtype == 'object']

    # Numeric transformer: impute -> optional scaler
    if add_scaler:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42, n_jobs=-1, tree_method="hist",
        # keep defaults unless overridden by xgb_params
        **xgb_params
    )

    pipe = Pipeline([("pre", pre), ("xgb", model)])
    return pipe, numeric_features, categorical_features

def compute_metrics(y_true_price, y_pred_price):
    rmse = float(np.sqrt(mean_squared_error(y_true_price, y_pred_price)))
    mae = float(mean_absolute_error(y_true_price, y_pred_price))
    med_ae = float(median_absolute_error(y_true_price, y_pred_price))
    r2 = float(r2_score(y_true_price, y_pred_price))
    # clip denominator to avoid division by near-zero
    mape = float(np.mean(np.abs((y_true_price - y_pred_price) / np.clip(y_true_price, 1e-9, None)))) * 100.0
    return {"RMSE": rmse, "MAE": mae, "Median AE": med_ae, "R²": r2, "MAPE%": mape}

def baseline_predictor(y_train_price, X_test_len):
    """Baseline: always predict the mean training price."""
    base_value = float(np.mean(y_train_price))
    return np.full((X_test_len,), base_value), base_value

def get_feature_names_from_preprocessor(pre: ColumnTransformer) -> List[str]:
    """Recover final feature names after ColumnTransformer."""
    names = []
    # numeric
    try:
        num_feats = pre.transformers_[0][2]  # list of numeric features
        names.extend(list(num_feats))
    except Exception:
        pass
    # categorical (OHE)
    try:
        cat_pipe = pre.named_transformers_["cat"]
        ohe = cat_pipe.named_steps["ohe"]
        cat_raw = pre.transformers_[1][2]
        ohe_names = ohe.get_feature_names_out(cat_raw)
        names.extend(list(ohe_names))
    except Exception:
        pass
    return names

# -------------------------
# Training (deterministic) 
# -------------------------
@st.cache_resource(show_spinner=True)
def train_model(X: pd.DataFrame,
                y_price: pd.Series,
                test_size: float,
                seed: int,
                add_scaler: bool,
                xgb_params: Dict):
    # Determinism
    np.random.seed(seed)

    # Filter rows with price
    mask = y_price.notna()
    X_use, y_use = X.loc[mask].copy(), y_price.loc[mask].copy()

    # Drop rows where ALL features are NaN
    X_use = X_use.dropna(how='all')
    y_use = y_use.loc[X_use.index]

    # Log target
    y_log = np.log1p(y_use)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_use, y_log, test_size=test_size, random_state=seed, shuffle=True
    )

    pipe, num_cols, cat_cols = make_pipeline(X_tr, add_scaler, xgb_params)
    pipe.fit(X_tr, y_tr)

    # Predictions back to price
    y_pred_price = np.expm1(pipe.predict(X_te))
    y_true_price = np.expm1(y_te)

    # Baseline comparison
    base_preds, base_value = baseline_predictor(np.expm1(y_tr), len(y_te))

    mets_model = compute_metrics(y_true_price, y_pred_price)
    mets_base = compute_metrics(y_true_price, base_preds)

    # Build test-set predictions table (for download)
    test_pred_df = X_te.copy()
    test_pred_df["price_true"] = y_true_price
    test_pred_df["price_pred"] = y_pred_price
    test_pred_df["abs_error"] = np.abs(y_true_price - y_pred_price)

    # Feature importance
    xgb = pipe.named_steps["xgb"]
    importances = getattr(xgb, "feature_importances_", None)
    if importances is not None:
        pre = pipe.named_steps["pre"]
        feat_names = get_feature_names_from_preprocessor(pre)
        if len(importances) == len(feat_names):
            fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
        else:
            fi_df = pd.DataFrame({"feature": [f"f{i}" for i in range(len(importances))],
                                  "importance": importances})
    else:
        fi_df = pd.DataFrame(columns=["feature", "importance"])

    return {
        "pipeline": pipe,
        "splits": (X_tr, X_te, y_tr, y_te),
        "metrics_model": mets_model,
        "metrics_baseline": mets_base,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "test_pred_df": test_pred_df,
        "feat_importance_df": fi_df,
        "baseline_value": base_value,
        "debug_shapes": {
            "rows_used": int(len(X_use)),
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "n_features": int(X_use.shape[1])
        }
    }

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Controls")
default_data_path = Path("data/listings_detailed.csv")

data_choice = st.sidebar.radio("Data source", ["Use bundled dataset", "Upload CSV"], index=0)
if data_choice == "Upload CSV":
    up = st.sidebar.file_uploader("Upload Airbnb-style CSV", type=["csv"])
    df_raw = pd.read_csv(up, low_memory=False) if up is not None else load_data(default_data_path)
else:
    df_raw = load_data(default_data_path)

st.sidebar.markdown("---")

# ===== Notebook parity selector =====
mode = st.sidebar.selectbox(
    "Notebook parity mode",
    ["Tuned (ipynb)", "Baseline (ipynb)", "Manual"],
    index=0
)

if mode in ("Tuned (ipynb)", "Baseline (ipynb)"):
    # Fixed split & seed & scaler off to match notebook
    test_size = 0.2
    seed = 42
    add_scaler = False

    # Params: align to your IPYNB
    if mode == "Tuned (ipynb)":
        # Your final tuned run
        n_estimators = 900
        max_depth = 6
        learning_rate = 0.05
        subsample = 0.8
        colsample_bytree = 0.8
        reg_lambda = 1.0
        reg_alpha = 0.0
        # (Optional common defaults for consistency)
        min_child_weight = 1
        gamma = 0.0
    else:
        # Your IPYNB baseline run (often ~600 estimators)
        n_estimators = 600
        max_depth = 6
        learning_rate = 0.05
        subsample = 0.8
        colsample_bytree = 0.8
        reg_lambda = 1.0
        reg_alpha = 0.0
        min_child_weight = 1
        gamma = 0.0

    # Show as read-only
    st.sidebar.caption("Parameters locked to IPYNB for exact reproducibility.")
    st.sidebar.slider("n_estimators", 100, 2000, n_estimators, 50, disabled=True)
    st.sidebar.slider("max_depth", 1, 15, max_depth, 1, disabled=True)
    st.sidebar.select_slider("learning_rate", options=[0.01,0.02,0.03,0.05,0.07,0.1], value=learning_rate, disabled=True)
    st.sidebar.select_slider("subsample", options=[0.6,0.7,0.8,0.9,1.0], value=subsample, disabled=True)
    st.sidebar.select_slider("colsample_bytree", options=[0.6,0.7,0.8,0.9,1.0], value=colsample_bytree, disabled=True)
    st.sidebar.select_slider("reg_lambda", options=[0.0,0.5,1.0,2.0,5.0], value=reg_lambda, disabled=True)
    st.sidebar.select_slider("reg_alpha", options=[0.0,0.1,0.5,1.0], value=reg_alpha, disabled=True)

else:
    # Manual mode: let you experiment
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
    add_scaler = st.sidebar.checkbox("Scale numeric features", value=True)
    st.sidebar.markdown("### XGBoost Parameters")
    n_estimators = st.sidebar.slider("n_estimators", 200, 1200, 700, 50)
    max_depth = st.sidebar.slider("max_depth", 3, 12, 7, 1)
    learning_rate = st.sidebar.select_slider("learning_rate", options=[0.01, 0.02, 0.03, 0.05, 0.07, 0.1], value=0.05)
    subsample = st.sidebar.select_slider("subsample", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)
    colsample_bytree = st.sidebar.select_slider("colsample_bytree", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)
    reg_lambda = st.sidebar.select_slider("reg_lambda", options=[0.0, 0.5, 1.0, 2.0, 5.0], value=1.0)
    reg_alpha = st.sidebar.select_slider("reg_alpha", options=[0.0, 0.1, 0.5, 1.0], value=0.0)
    min_child_weight = 1
    gamma = 0.0

xgb_params = dict(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    reg_lambda=reg_lambda,
    reg_alpha=reg_alpha,
    min_child_weight=min_child_weight,
    gamma=gamma,
    eval_metric="rmse"   # explicit for parity
)


# -------------------------
# Header
# -------------------------
st.markdown("<h1><i class='fa-solid fa-house'></i> Airbnb Price Prediction • XGBoost (log-price)</h1>", unsafe_allow_html=True)
if mode == "Tuned (ipynb)":
    st.write("Trains XGBoost on `log(price)`, applies notebook-style price filtering (drop zeros + IQR trim), reports metrics on the original price scale, includes baseline comparison, diagnostic plots, and interactive predictions.")
elif mode == "Baseline (ipynb)":
    st.write("Notebook parity: **Baseline** — expect ~ **RMSE 43.14**, **R² 0.716**, **MAE 29.18**, **Median AE 18.91**")
else:
    st.write("Manual mode — experiment with parameters and preprocessing.")

# -------------------------
# Build features & Train
# -------------------------
with st.spinner("Building features and training the model..."):
    X, y_price = build_feature_frame(df_raw)
    result = train_model(
        X, y_price,
        test_size=test_size, seed=seed,
        add_scaler=add_scaler, xgb_params=xgb_params
    )

pipe = result["pipeline"]
X_tr, X_te, y_tr, y_te = result["splits"]
metrics_model = result["metrics_model"]
metrics_baseline = result["metrics_baseline"]
test_pred_df = result["test_pred_df"]
fi_df = result["feat_importance_df"]
dbg = result["debug_shapes"]

# -------------------------
# Metrics
# -------------------------
st.subheader("Model Performance (Original Price Scale)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("RMSE", f"{metrics_model['RMSE']:.2f}")
c2.metric("MAE", f"{metrics_model['MAE']:.2f}")
c3.metric("Median AE", f"{metrics_model['Median AE']:.2f}")
c4.metric("R²", f"{metrics_model['R²']:.3f}")
c5.metric("MAPE%", f"{metrics_model['MAPE%']:.1f}")

# Small sanity strip to ensure parity with the notebook
st.caption(f"Rows used: {dbg['rows_used']} • Train: {dbg['train_rows']} • Test: {dbg['test_rows']} • Features: {dbg['n_features']}")
st.markdown("---")

with st.expander("Baseline comparison (predicting the mean training price)"):
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("RMSE (baseline)", f"{metrics_baseline['RMSE']:.2f}")
    b2.metric("MAE (baseline)", f"{metrics_baseline['MAE']:.2f}")
    b3.metric("Median AE (baseline)", f"{metrics_baseline['Median AE']:.2f}")
    b4.metric("R² (baseline)", f"{metrics_baseline['R²']:.3f}")
    b5.metric("MAPE% (baseline)", f"{metrics_baseline['MAPE%']:.1f}")

# -------------------------
# Tabs (plain labels; FA icons inside each tab)
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "EDA", "Diagnostics", "Feature Importance", "Predict", "Scenario Compare"
])

# ===== Tab 1: EDA =====
with tab1:
    st.markdown("<h3 style='margin-top:0'><i class='fa-solid fa-chart-bar'></i> EDA</h3>", unsafe_allow_html=True)
    st.markdown("#### Dataset Glance")
    with st.expander("Raw head()", expanded=False):
        st.dataframe(df_raw.head(20), use_container_width=True)

    # Price histograms
    if 'price' in df_raw.columns:
        price_clean = df_raw['price'].apply(_clean_price)
        colA, colB = st.columns(2)
        with colA:
            fig = px.histogram(price_clean.dropna(), nbins=50, title="Price Distribution (cleaned)")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            fig = px.histogram(np.log1p(price_clean.dropna()), nbins=50, title="Log(1+Price) Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # Categorical counts
    cat_show = [c for c in ["room_type","property_type","neighbourhood_cleansed"] if c in df_raw.columns]
    for c in cat_show:
        vc = df_raw[c].value_counts(dropna=False).head(25)
        fig = px.bar(vc, title=f"{c} — top counts")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap for numeric
    num_for_corr = X.select_dtypes(include=[np.number])
    if not num_for_corr.empty:
        corr = num_for_corr.corr(numeric_only=True)
        fig = px.imshow(corr, title="Correlation Heatmap (numeric features)")
        st.plotly_chart(fig, use_container_width=True)

# ===== Tab 2: Diagnostics =====
with tab2:
    st.markdown("<h3 style='margin-top:0'><i class='fa-solid fa-square-poll-vertical'></i> Diagnostics</h3>", unsafe_allow_html=True)
    st.markdown("#### Predicted vs Actual (Test Set)")
    y_true = test_pred_df["price_true"].values
    y_pred = test_pred_df["price_pred"].values

    # Scatter with diagonal
    scatter_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    fig = px.scatter(scatter_df, x="Actual", y="Predicted", trendline=None)
    max_axis = float(np.nanmax([scatter_df["Actual"].max(), scatter_df["Predicted"].max()]))
    fig.add_trace(go.Scatter(x=[0, max_axis], y=[0, max_axis], mode='lines', name='Ideal'))
    fig.update_layout(xaxis_title="Actual Price", yaxis_title="Predicted Price")
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    st.markdown("#### Residuals (Actual - Predicted)")
    resid = y_true - y_pred
    resid_df = pd.DataFrame({"Predicted": y_pred, "Residual": resid})
    fig = px.scatter(resid_df, x="Predicted", y="Residual", trendline=None)
    fig.update_layout(xaxis_title="Predicted Price", yaxis_title="Residual (Actual - Predicted)")
    st.plotly_chart(fig, use_container_width=True)

    # Residual histogram
    fig = px.histogram(resid, nbins=60, title="Residuals Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Downloads
    st.download_button(
        "Download test-set predictions (CSV)",
        data=test_pred_df.to_csv(index=False).encode("utf-8"),
        file_name="test_predictions.csv",
        mime="text/csv"
    )

# ===== Tab 3: Feature Importance =====
with tab3:
    st.markdown("<h3 style='margin-top:0'><i class='fa-solid fa-brain'></i> Feature Importance</h3>", unsafe_allow_html=True)
    st.markdown("#### Top Features")
    if not fi_df.empty:
        fi_df_sorted = fi_df.sort_values("importance", ascending=False)
        # Normalize to percentage
        total = fi_df_sorted["importance"].sum()
        if total > 0:
            fi_df_sorted["importance_pct"] = 100.0 * fi_df_sorted["importance"] / total

        top_n = st.slider("Top N features", 10, min(50, len(fi_df_sorted)), min(20, len(fi_df_sorted)))
        top_df = fi_df_sorted.head(top_n)
        fig = px.bar(top_df, x="importance_pct", y="feature", orientation="h",
                     title="Feature Importance (%) — Top N")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(fi_df_sorted, use_container_width=True, height=400)
        st.download_button(
            "Download full feature importance (CSV)",
            data=fi_df_sorted.to_csv(index=False).encode("utf-8"),
            file_name="feature_importance.csv",
            mime="text/csv"
        )
    else:
        st.info("Feature importances are not available for this model/configuration.")

# ===== Tab 4: Predict (single) =====
with tab4:
    st.markdown("<h3 style='margin-top:0'><i class='fa-solid fa-calculator'></i> Predict</h3>", unsafe_allow_html=True)
    st.markdown("#### Enter Listing Details")
    with st.form("predict_form"):
        colA, colB, colC = st.columns(3)
        bedrooms = colA.number_input("Bedrooms", 0.0, 20.0, 2.0, 1.0)
        bathrooms = colA.number_input("Bathrooms", 0.0, 10.0, 1.0, 0.5)
        accommodates = colA.number_input("Accommodates", 1, 20, 3, 1)
        minimum_nights = colB.number_input("Minimum nights", 1, 365, 1, 1)
        maximum_nights = colB.number_input("Maximum nights", 1, 10000, 365, 1)
        number_of_reviews = colB.number_input("Number of reviews", 0, 5000, 10, 1)
        review_scores_rating = colC.number_input("Review score rating", 0.0, 100.0, 94.0, 0.5)
        amenity_count = colC.number_input("Amenity count", 0, 200, 12, 1)

        # Binary & categoricals from raw options if available
        host_is_superhost_01 = colA.selectbox("Superhost?", ["No","Yes"])
        instant_bookable_01 = colB.selectbox("Instant bookable?", ["No","Yes"])
        room_type = colC.selectbox(
            "Room type",
            sorted(df_raw["room_type"].dropna().unique()) if "room_type" in df_raw else ["Entire home/apt"]
        )
        property_type = colC.selectbox(
            "Property type",
            sorted(df_raw["property_type"].dropna().unique()) if "property_type" in df_raw else ["Apartment"]
        )
        neighbourhood_cleansed = colB.selectbox(
            "Neighbourhood",
            sorted(df_raw["neighbourhood_cleansed"].dropna().unique()) if "neighbourhood_cleansed" in df_raw else ["Unknown"]
        )
        submitted = st.form_submit_button("Predict")
    if submitted:
        row = {
            "bedrooms": bedrooms, "bathrooms": bathrooms, "accommodates": accommodates,
            "minimum_nights": minimum_nights, "maximum_nights": maximum_nights,
            "review_scores_rating": review_scores_rating, "number_of_reviews": number_of_reviews,
            "amenity_count": amenity_count,
            "host_is_superhost_01": 1 if host_is_superhost_01=="Yes" else 0,
            "instant_bookable_01": 1 if instant_bookable_01=="Yes" else 0,
            "room_type": room_type, "property_type": property_type, "neighbourhood_cleansed": neighbourhood_cleansed
        }
        X_pred = pd.DataFrame([row])
        price_pred = float(np.expm1(pipe.predict(X_pred))[0])

        # Typical error band from MAE
        typical_err = metrics_model["MAE"]
        lo = max(price_pred - typical_err, 0.0)
        hi = price_pred + typical_err

        st.success(f"Predicted nightly price: ${price_pred:,.2f}")
        st.caption(f"Typical error band (±MAE): ${typical_err:,.2f} → approx range: ${lo:,.2f} — ${hi:,.2f}")

# ===== Tab 5: Scenario Compare =====
with tab5:
    st.markdown("<h3 style='margin-top:0'><i class='fa-solid fa-code-compare'></i> Scenario Compare</h3>", unsafe_allow_html=True)
    st.markdown("#### Compare Two Scenarios")

    def scenario_inputs(prefix: str):
        colA, colB, colC = st.columns(3)
        b = colA.number_input(f"{prefix} Bedrooms", 0.0, 20.0, 2.0, 1.0, key=f"{prefix}_bdr")
        ba = colA.number_input(f"{prefix} Bathrooms", 0.0, 10.0, 1.0, 0.5, key=f"{prefix}_bath")
        acc = colA.number_input(f"{prefix} Accommodates", 1, 20, 3, 1, key=f"{prefix}_acc")
        minN = colB.number_input(f"{prefix} Minimum nights", 1, 365, 1, 1, key=f"{prefix}_min")
        maxN = colB.number_input(f"{prefix} Maximum nights", 1, 10000, 365, 1, key=f"{prefix}_max")
        nor = colB.number_input(f"{prefix} Number of reviews", 0, 5000, 10, 1, key=f"{prefix}_nor")
        rsr = colC.number_input(f"{prefix} Review score rating", 0.0, 100.0, 94.0, 0.5, key=f"{prefix}_rsr")
        amc = colC.number_input(f"{prefix} Amenity count", 0, 200, 12, 1, key=f"{prefix}_amc")
        sh = colA.selectbox(f"{prefix} Superhost?", ["No","Yes"], key=f"{prefix}_sh")
        ib = colB.selectbox(f"{prefix} Instant bookable?", ["No","Yes"], key=f"{prefix}_ib")
        rt = colC.selectbox(
            f"{prefix} Room type",
            sorted(df_raw["room_type"].dropna().unique()) if "room_type" in df_raw else ["Entire home/apt"],
            key=f"{prefix}_rt"
        )
        pt = colC.selectbox(
            f"{prefix} Property type",
            sorted(df_raw["property_type"].dropna().unique()) if "property_type" in df_raw else ["Apartment"],
            key=f"{prefix}_pt"
        )
        nb = colB.selectbox(
            f"{prefix} Neighbourhood",
            sorted(df_raw["neighbourhood_cleansed"].dropna().unique()) if "neighbourhood_cleansed" in df_raw else ["Unknown"],
            key=f"{prefix}_nb"
        )
        row = {
            "bedrooms": b, "bathrooms": ba, "accommodates": acc,
            "minimum_nights": minN, "maximum_nights": maxN,
            "review_scores_rating": rsr, "number_of_reviews": nor,
            "amenity_count": amc,
            "host_is_superhost_01": 1 if sh=="Yes" else 0,
            "instant_bookable_01": 1 if ib=="Yes" else 0,
            "room_type": rt, "property_type": pt, "neighbourhood_cleansed": nb
        }
        return row

    rowA = scenario_inputs("A")
    rowB = scenario_inputs("B")
    if st.button("Compare"):
        dfA = pd.DataFrame([rowA])
        dfB = pd.DataFrame([rowB])
        predA = float(np.expm1(pipe.predict(dfA))[0])
        predB = float(np.expm1(pipe.predict(dfB))[0])
        diff = predB - predA

        c1, c2, c3 = st.columns(3)
        c1.metric("Scenario A price", f"${predA:,.2f}")
        c2.metric("Scenario B price", f"${predB:,.2f}")
        c3.metric("Difference (B - A)", f"${diff:,.2f}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")


