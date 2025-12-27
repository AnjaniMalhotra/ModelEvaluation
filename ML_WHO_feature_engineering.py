import streamlit as st
import pandas as pd
import numpy as np


def run_core_feature_engineering(df):
    st.header("🧩 Core Feature Engineering (Regression)")

    target_col = st.selectbox(
        "Select the target column (numeric)",
        ["-- Select target --"] + list(df.columns)
    )

    if target_col == "-- Select target --":
        st.info("Please select a numeric target column.")
        return

    if not st.button("🚀 Run Feature Engineering"):
        return

    y = df[target_col]

    # --- Validate numeric target ---
    if not pd.api.types.is_numeric_dtype(y):
        st.error("❌ Selected target is not numeric. Regression requires numeric target.")
        st.stop()

    # --- Features ---
    X = df.drop(columns=[target_col])

    # Encode categoricals
    X = pd.get_dummies(X)

    # Clean
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.fillna(0)

    # Save
    st.session_state["X"] = X
    st.session_state["y"] = y
    st.session_state["target"] = target_col

    st.success("✅ Feature engineering completed (regression-ready)")
    st.write("Feature matrix shape:", X.shape)
