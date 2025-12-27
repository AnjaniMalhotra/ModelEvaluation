import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


def evaluate_models(X, y):
    st.header("📈 Regression Model Evaluation")

    if X is None or y is None:
        st.error("Run feature engineering first.")
        return

    # -------------------------------
    # Train-test split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Scaling
    # -------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    # Regression models (CLOUD SAFE)
    # -------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor(),
        "SVR": SVR()
    }

    results = []
    predictions = {}

    # -------------------------------
    # Train & evaluate
    # -------------------------------
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append([name, mse, r2])
            predictions[name] = y_pred

        except Exception as e:
            st.warning(f"{name} skipped: {e}")

    results_df = pd.DataFrame(
        results,
        columns=["Model", "MSE", "R2"]
    ).sort_values(by="R2", ascending=False)

    st.subheader("🏆 Regression Model Comparison")
    st.dataframe(results_df)

    # -------------------------------
    # Plot Top Model
    # -------------------------------
    best_model = results_df.iloc[0]["Model"]
    y_best_pred = predictions[best_model]

    st.subheader(f"📊 Actual vs Predicted – {best_model}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_best_pred, alpha=0.6)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--"
    )
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

    st.success("✅ Regression evaluation completed successfully.")
