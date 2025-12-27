import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def run_core_feature_engineering(df):
    st.header("🧩 Core Feature Engineering")

    # -------------------------------
    # Target selection
    # -------------------------------
    target_col = st.selectbox(
        "Select the target column",
        ["-- Select target --"] + list(df.columns)
    )

    if target_col == "-- Select target --":
        st.info("Please select a target column to proceed.")
        return

    if not st.button("🚀 Run Feature Engineering"):
        st.warning("Click the button to run feature engineering.")
        return

    # -------------------------------
    # Split features & target
    # -------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -------------------------------
    # Target validation (classification only)
    # -------------------------------
    if pd.api.types.is_numeric_dtype(y):
        st.error(
            "Numeric targets are not supported for model evaluation.\n\n"
            "Please choose a categorical target like:\n"
            "- Payment Status\n"
            "- Card Order Status\n"
            "- Gateway\n"
            "- Company Name\n"
            "- Month"
        )
        st.stop()

    # Encode target
    y = y.astype(str).str.lower().fillna("unknown")
    le = LabelEncoder()
    y = le.fit_transform(y)

    # -------------------------------
    # Feature encoding
    # -------------------------------
    X = pd.get_dummies(X)

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # -------------------------------
    # Save to session state
    # -------------------------------
    st.session_state["X"] = X
    st.session_state["y"] = y
    st.session_state["target"] = target_col

    st.success("✅ Feature engineering completed successfully!")

    st.write("Final feature matrix shape:", X.shape)
    st.write("Target classes:", len(set(y)))
