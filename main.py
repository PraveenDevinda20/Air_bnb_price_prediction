from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple, List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

import uvicorn

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "listings_detailed.csv"
# must exist; put your video at static/video/airbnb.mp4
STATIC_DIR = BASE / "static"
TEMPLATES_DIR = BASE / "templates"    # must exist; put index.html inside

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# App
# ---------------------------------------------------------
app = FastAPI(title="StayGauge API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------
# Preprocessing utilities (same logic as your Streamlit app.py)
# ---------------------------------------------------------


def _clean_price(x):
    if pd.isna(x):
        return np.nan
    s = re.sub(r'[^0-9\.]', '', str(x))
    try:
        return float(s) if s else np.nan
    except:
        return np.nan


def _extract_baths(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r'([\d\.]+)', str(x))
    return float(m.group(1)) if m else np.nan


def _amenity_count(x):
    if pd.isna(x):
        return 0.0
    s = str(x)
    if s.startswith('[') and s.endswith(']'):
        try:
            inner = s.strip()[1:-1]
            return float(len([t for t in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', inner) if t.strip()]))
        except:
            return float(s.count(',') + 1)
    return float(len([a for a in s.split(',') if a.strip()]))


def _bool01(x):
    if isinstance(x, str):
        return 1 if x.lower() in ('t', 'true', 'yes', 'y', '1') else 0
    if isinstance(x, (int, float, bool)):
        return int(bool(x))
    return 0


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
    """Return X (13 raw features) and y (cleaned price)."""
    df = df_raw.copy()
    df['price_clean'] = df['price'].apply(
        _clean_price) if 'price' in df.columns else np.nan

    if 'price_clean' in df.columns:
        mask_valid = _valid_price_mask(df)
        if mask_valid.any():
            mask_iqr = _iqr_trim(df.loc[mask_valid, 'price_clean'])
            keep_idx = df.loc[mask_valid].index[mask_iqr]
            df = df.loc[keep_idx].copy()

    if 'bathrooms_text' in df.columns:
        df['bathrooms'] = df['bathrooms_text'].apply(_extract_baths)
    elif 'bathrooms' in df.columns:
        df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
    else:
        df['bathrooms'] = np.nan

    df['amenity_count'] = df['amenities'].apply(
        _amenity_count) if 'amenities' in df.columns else np.nan

    df['host_is_superhost_01'] = df['host_is_superhost'].apply(
        _bool01) if 'host_is_superhost' in df.columns else 0
    df['instant_bookable_01'] = df['instant_bookable'].apply(
        _bool01) if 'instant_bookable' in df.columns else 0

    for col in ['bedrooms', 'accommodates', 'minimum_nights', 'maximum_nights',
                'review_scores_rating', 'number_of_reviews']:
        df[col] = pd.to_numeric(
            df[col], errors='coerce') if col in df.columns else np.nan

    num_feats = [
        'bedrooms', 'bathrooms', 'accommodates', 'minimum_nights', 'maximum_nights',
        'review_scores_rating', 'number_of_reviews', 'amenity_count',
        'host_is_superhost_01', 'instant_bookable_01'
    ]
    cat_feats = [c for c in ['neighbourhood_cleansed',
                             'room_type', 'property_type'] if c in df.columns]
    num_feats = [c for c in num_feats if c in df.columns]
    cat_feats = [c for c in cat_feats if c in df.columns]

    X = df[num_feats + cat_feats].copy()
    y = df['price_clean'] if 'price_clean' in df.columns else pd.Series(
        np.nan, index=df.index)
    return X, y


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = [c for c in X.columns if X[c].dtype != 'object']
    categorical_features = [c for c in X.columns if X[c].dtype == 'object']

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_features)
        ],
        remainder="drop"
    )

    # tuned params aligned to your notebook "Tuned (ipynb)"
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42, n_jobs=-1, tree_method="hist",
        n_estimators=900, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.0,
        min_child_weight=1, gamma=0.0,
        eval_metric="rmse"
    )
    return Pipeline([("pre", pre), ("xgb", xgb)])


# ---------------------------------------------------------
# Train once on startup (exact 13 features)
# ---------------------------------------------------------
DATA = pd.read_csv(DATA_PATH, low_memory=False)
X_full, y_price = build_feature_frame(DATA)
mask = y_price.notna()
X_use, y_use = X_full.loc[mask].copy(), y_price.loc[mask].copy()

PIPE = make_pipeline(X_use)
y_log = np.log1p(y_use)
PIPE.fit(X_use, y_log)

# Small MAE for ±band on the original scale
try:
    y_hat_train = np.expm1(PIPE.predict(X_use))
    MAE_BAND = float(np.mean(np.abs(y_use.values - y_hat_train)))
except Exception:
    MAE_BAND = None

# Dropdown options for UI
ROOM_OPTS = sorted(DATA["room_type"].dropna().unique(
).tolist()) if "room_type" in DATA.columns else []
PROP_OPTS = sorted(DATA["property_type"].dropna().unique(
).tolist()) if "property_type" in DATA.columns else []
NEIGH_OPTS = sorted(DATA["neighbourhood_cleansed"].dropna().unique(
).tolist()) if "neighbourhood_cleansed" in DATA.columns else []

RAW_FEATURES = [
    'bedrooms', 'bathrooms', 'accommodates', 'minimum_nights', 'maximum_nights',
    'review_scores_rating', 'number_of_reviews', 'amenity_count',
    'host_is_superhost_01', 'instant_bookable_01',
    'room_type', 'property_type', 'neighbourhood_cleansed'
]

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def home():
    html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/schema")
def schema():
    return JSONResponse({
        "input_features": RAW_FEATURES,
        "room_type_options": ROOM_OPTS,
        "property_type_options": PROP_OPTS,
        "neighbourhood_cleansed_options": NEIGH_OPTS
    })


@app.get("/about", response_class=HTMLResponse)
def about():
    text = (
        "StayGauge — Airbnb Price Prediction\n\n"
        "• Dataset: Inside Airbnb NYC (cleaned)\n"
        "• Model: XGBoost on log(price)\n"
        "• Features: 13 raw attributes (rooms, reviews, amenities, and categorical type/location)\n"
        "• Output: Estimated nightly price on the original scale\n"
    )
    return HTMLResponse(text.replace("\n", "<br/>"))


@app.post("/predict")
async def predict(payload: Dict):
    # Ensure we have all 13 raw fields
    missing = [k for k in RAW_FEATURES if k not in payload]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing fields: {missing}")

    # Build single-row dataframe in exact order
    df = pd.DataFrame([{k: payload[k] for k in RAW_FEATURES}])

    # Force numeric types
    for col in ['bedrooms', 'bathrooms', 'accommodates', 'minimum_nights', 'maximum_nights',
                'review_scores_rating', 'number_of_reviews', 'amenity_count',
                'host_is_superhost_01', 'instant_bookable_01']:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        pred_log = PIPE.predict(df)[0]
        price = float(np.expm1(pred_log))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    band = None
    if MAE_BAND is not None:
        band = {"lo": max(price - MAE_BAND, 0.0), "hi": price + MAE_BAND}

    # >>> Key fix: return "price" so the UI reads it correctly
    return JSONResponse({"ok": True, "price": price, "band": band})

# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
