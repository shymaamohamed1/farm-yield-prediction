import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


st.set_page_config(
    page_title="Farm Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .main { background-color: #f7faf7; }
        .stApp { font-family: 'Segoe UI', sans-serif; }

        .hero {
            background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%);
            padding: 2rem 2.2rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 24px rgba(46,125,50,0.25);
        }
        .hero h1 { margin: 0; font-size: 2.1rem; }
        .hero p  { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1.02rem; }

        .metric-card {
            background: white;
            border-radius: 16px;
            padding: 1.4rem 1.2rem;
            text-align: center;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            border: 1px solid #eef2ee;
        }
        .metric-card h3 { margin: 0; font-size: 0.95rem; color: #667085; font-weight: 600; }
        .metric-card .value { font-size: 2.1rem; font-weight: 700; color: #2e7d32; margin-top: 0.3rem; }
        .metric-card .sub { font-size: 0.82rem; color: #98a2b3; margin-top: 0.2rem; }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1d2b1d;
            margin: 1.4rem 0 0.6rem 0;
            border-left: 5px solid #2e7d32;
            padding-left: 0.6rem;
        }

        div.stButton > button {
            background: #2e7d32;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            border: none;
            width: 100%;
        }
        div.stButton > button:hover { background: #256428; color: white; }

        [data-testid="stSidebar"] { background-color: #f0f5f0; }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_DIR = Path(".")  

NUMERIC_FEATURES = [
    "fertilizer_kg_per_hectare",
    "rainfall_mm",
    "soil_ph",
    "farm_size_hectares",
    "years_farming_experience",
]
CATEGORICAL_FEATURES = ["crop_type", "irrigation_type", "region"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES  

CROP_TYPES = ["Wheat", "Corn", "Barley", "Rice"]
IRRIGATION_TYPES = ["Flood", "Drip", "Sprinkler"]
REGIONS = ["Upper Egypt", "Sinai", "Delta"]

TARGET_COL = "yield_tons_per_hectare"


LINEAR_R2_TEST = 0.5552589574088418
POLY_R2_TEST = 0.790732674130372



@st.cache_resource(show_spinner=False)
def load_artifacts():
    
    filename = "lasso_model.pkl"
    path = MODEL_DIR / filename
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
    except Exception as e:
        return {}, [f"Could not load **{filename}**: {e}"]

    required_keys = {
        "Encoders": "label_encoder",
        "Scaler": "scaler",
        "linear_model": "linear_model",
        "Polynomial Features": "poly_features",
        "Lasso": "poly_model",
    }

    artifacts, errors = {}, []
    for src_key, internal_key in required_keys.items():
        if src_key in bundle:
            artifacts[internal_key] = bundle[src_key]
        else:
            errors.append(f"Key **'{src_key}'** was not found in {filename}.")

    return artifacts, errors


artifacts, load_errors = load_artifacts()


st.markdown(
    """
    <div class="hero">
        <h1>🌾 Farm Yield Prediction</h1>
        <p>Estimate crop yield (tons per hectare) using a Linear Regression model and a
        Polynomial Regression model trained on farming and environmental data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_errors:
    st.error(
        "The model bundle could not be fully loaded. Make sure **lasso_model.pkl** is in the "
        "same folder as this script (or update `MODEL_DIR`), and that it contains the expected keys.\n\n"
        + "\n".join(f"- {e}" for e in load_errors)
    )
    st.stop()


st.sidebar.header("📋 Farm Input Data")
st.sidebar.caption("Enter the field conditions below, then click Predict.")

with st.sidebar.form("input_form"):
    st.subheader("🌱 Crop & Location")
    crop_type = st.selectbox("Crop Type", CROP_TYPES)
    irrigation_type = st.selectbox("Irrigation Type", IRRIGATION_TYPES)
    region = st.selectbox("Region", REGIONS)

    st.subheader("🧪 Field Conditions")
    fertilizer_kg_per_hectare = st.number_input(
        "Fertilizer (kg/hectare)", min_value=0.0, max_value=1000.0, value=150.0, step=1.0
    )
    rainfall_mm = st.number_input(
        "Rainfall (mm)", min_value=0.0, max_value=3000.0, value=500.0, step=1.0
    )
    soil_ph = st.slider("Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1)

    st.subheader("🚜 Farm Profile")
    farm_size_hectares = st.number_input(
        "Farm Size (hectares)", min_value=0.1, max_value=10000.0, value=10.0, step=0.1
    )
    years_farming_experience = st.number_input(
        "Years of Farming Experience", min_value=0, max_value=80, value=5, step=1
    )

    submitted = st.form_submit_button("🔮 Predict Yield")


def build_input_dataframe():
    
    return pd.DataFrame(
        [{
            "fertilizer_kg_per_hectare": fertilizer_kg_per_hectare,
            "rainfall_mm": rainfall_mm,
            "soil_ph": soil_ph,
            "farm_size_hectares": farm_size_hectares,
            "years_farming_experience": years_farming_experience,
            "crop_type": crop_type,
            "irrigation_type": irrigation_type,
            "region": region,
        }]
    )[FEATURE_ORDER]


def encode_categoricals(df):
   
    df = df.copy()
    encoder = artifacts["label_encoder"]

    if isinstance(encoder, dict):
        for col in CATEGORICAL_FEATURES:
            le = encoder.get(col)
            if le is None:
                raise ValueError(f"No encoder found for column '{col}' in label_encoder.pkl")
            df[col] = le.transform(df[col])
    else:
        for col in CATEGORICAL_FEATURES:
            df[col] = encoder.transform(df[col])

    return df


def make_predictions(raw_df):
    encoded_df = encode_categoricals(raw_df)
    scaled = artifacts["scaler"].transform(encoded_df)

    linear_pred = artifacts["linear_model"].predict(scaled)[0]

    poly_input = artifacts["poly_features"].transform(scaled)
    poly_pred = artifacts["poly_model"].predict(poly_input)[0]

    return float(linear_pred), float(poly_pred)



col_main, col_info = st.columns([2.2, 1])

with col_main:
    st.markdown('<div class="section-title">🔍 Prediction Results</div>', unsafe_allow_html=True)

    if submitted:
        try:
            raw_df = build_input_dataframe()
            linear_pred, poly_pred = make_predictions(raw_df)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"""<div class="metric-card">
                            <h3>📐 Linear Model</h3>
                            <div class="value">{linear_pred:.2f}</div>
                            <div class="sub">tons / hectare</div>
                        </div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div class="metric-card">
                            <h3>📈 Polynomial (Lasso) Model</h3>
                            <div class="value">{poly_pred:.2f}</div>
                            <div class="sub">tons / hectare</div>
                        </div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                diff = poly_pred - linear_pred
                st.markdown(
                    f"""<div class="metric-card">
                            <h3>↔️ Difference</h3>
                            <div class="value">{diff:+.2f}</div>
                            <div class="sub">poly − linear</div>
                        </div>""",
                    unsafe_allow_html=True,
                )

           
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=["Linear Model", "Polynomial (Lasso) Model"],
                        y=[linear_pred, poly_pred],
                        marker_color=["#66bb6a", "#2e7d32"],
                        text=[f"{linear_pred:.2f}", f"{poly_pred:.2f}"],
                        textposition="outside",
                    )
                ]
            )
            fig.update_layout(
                title="Predicted Yield Comparison",
                yaxis_title="Yield (tons/hectare)",
                height=380,
                margin=dict(t=50, b=20, l=20, r=20),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📄 View submitted input data"):
                st.dataframe(raw_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("👈 Fill in the farm details in the sidebar and click **Predict Yield** to see results.")

with col_info:
    st.markdown('<div class="section-title">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This app compares two regression approaches for predicting
        **crop yield (tons/hectare)**:

        - **Linear Regression** — models a straight-line relationship
          between features and yield.
        - **Polynomial Regression** — captures non-linear relationships
          by expanding features into polynomial terms.

        Both models were trained on the same encoded & scaled features:
        fertilizer use, rainfall, soil pH, farm size, farming experience,
        crop type, irrigation type, and region.
        """
    )


st.markdown('<div class="section-title">📊 Model Accuracy</div>', unsafe_allow_html=True)
st.caption("R² score of each model, measured on the held-out test set during training.")

acc1, acc2 = st.columns(2)
with acc1:
    st.markdown(
        f"""<div class="metric-card">
                <h3>📐 Linear Model — R² (Test)</h3>
                <div class="value">{LINEAR_R2_TEST * 100:.2f}%</div>
                <div class="sub">R² = {LINEAR_R2_TEST:.4f}</div>
            </div>""",
        unsafe_allow_html=True,
    )
with acc2:
    st.markdown(
        f"""<div class="metric-card">
                <h3>📈 Polynomial (Lasso) Model — R² (Test)</h3>
                <div class="value">{POLY_R2_TEST * 100:.2f}%</div>
                <div class="sub">R² = {POLY_R2_TEST:.4f}</div>
            </div>""",
        unsafe_allow_html=True,
    )


st.markdown("---")
st.caption("🌾 Farm Yield Prediction App · Built with Streamlit, scikit-learn & Plotly")
