import streamlit as st
import pandas as pd

from ML_WHO_statistics import show_statistics
from ML_WHO_visualisations import show_visualisations
from ML_WHO_feature_engineering import run_core_feature_engineering
from ML_WHO_evaluation import evaluate_models

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="ML WHO?",
    page_icon="🧠",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
body {
    background-color: #f7f9fb;
}
.main {
    font-family: 'Segoe UI', sans-serif;
}
.section-title {
    color: #0066cc;
    font-size: 28px;
    margin-bottom: 0.5em;
}
.stButton > button {
    background-color: #0066cc;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
}
.stButton > button:hover {
    background-color: #004c99;
}
.uploaded-file {
    font-weight: bold;
    color: green;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE DEFAULTS
# ==================================================
if "section" not in st.session_state:
    st.session_state.section = "Home"

# ==================================================
# APP TITLE
# ==================================================
st.markdown(
    "<h1 style='text-align:center; color:#0c4a6e;'>🤖 ML WHO? – Regression Model Evaluation App</h1>",
    unsafe_allow_html=True
)

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("🔍 Navigation")

st.session_state.section = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Upload Data",
        "Statistics",
        "Visualisations",
        "Core Feature Engineering",
        "Model Evaluation"
    ]
)

# ==================================================
# HOME
# ==================================================
if st.session_state.section == "Home":
    st.markdown("""
    <h3 class="section-title">📊 Welcome to ML WHO?</h3>
    <p>
        This app focuses on <b>regression-based machine learning model evaluation</b>.
        You can upload your dataset, engineer features, and compare multiple regression models.
    </p>
    <ul>
        <li>📁 Upload CSV / Excel / JSON data</li>
        <li>📊 Explore basic statistics</li>
        <li>📈 Visualize data distributions</li>
        <li>🧩 Perform regression-safe feature engineering</li>
        <li>📉 Evaluate and compare regression models</li>
    </ul>
    <p>👈 Start by uploading a dataset.</p>
    """, unsafe_allow_html=True)

# ==================================================
# UPLOAD DATA
# ==================================================
elif st.session_state.section == "Upload Data":
    st.markdown("<h3 class='section-title'>📁 Upload Dataset</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a CSV, Excel, or JSON file",
        type=["csv", "xlsx", "xls", "json"]
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)

            st.session_state.df = df.copy()

            st.success("✅ Dataset uploaded successfully!")
            st.markdown("<p class='uploaded-file'>Preview:</p>", unsafe_allow_html=True)
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")

# ==================================================
# SECTIONS THAT REQUIRE DATA
# ==================================================
elif "df" in st.session_state:
    df = st.session_state.df

    # ---------------- STATISTICS ----------------
    if st.session_state.section == "Statistics":
        st.markdown("<h3 class='section-title'>📊 Dataset Statistics</h3>", unsafe_allow_html=True)
        show_statistics(df)

    # ---------------- VISUALISATIONS ----------------
    elif st.session_state.section == "Visualisations":
        st.markdown("<h3 class='section-title'>📈 Data Visualisations</h3>", unsafe_allow_html=True)
        show_visualisations(df)

    # ---------------- CORE FEATURE ENGINEERING ----------------
    elif st.session_state.section == "Core Feature Engineering":
        st.markdown("<h3 class='section-title'>🧩 Core Feature Engineering (Regression)</h3>", unsafe_allow_html=True)
        run_core_feature_engineering(df)

    # ---------------- MODEL EVALUATION ----------------
    elif st.session_state.section == "Model Evaluation":
        st.markdown("<h3 class='section-title'>📉 Regression Model Evaluation</h3>", unsafe_allow_html=True)

        if "X" in st.session_state and "y" in st.session_state:
            evaluate_models(
                st.session_state.X,
                st.session_state.y
            )
        else:
            st.warning("⚠️ Please run Core Feature Engineering first.")

# ==================================================
# NO DATA FALLBACK
# ==================================================
else:
    st.info("📁 Please upload a dataset first using **Upload Data**.")
