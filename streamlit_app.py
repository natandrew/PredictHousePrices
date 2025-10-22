# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

MODEL_PATH = "model.pkl"

import os
from urllib.error import URLError
from http.client import RemoteDisconnected
from sklearn.datasets import fetch_openml

DATA_CSV = os.path.join("data", "boston.csv")  # repo path fallback
MODEL_PATH = "model.pkl"

@st.cache_data(show_spinner=False)
def load_dataset():
    """
    Try to fetch from OpenML. If it fails (network error, remote disconnect, etc.),
    try to load a local CSV at data/boston.csv. If neither exists, raise a helpful error.
    """
    # First try OpenML
    try:
        boston = fetch_openml(name="boston", version=1, as_frame=True)
        X = boston.data.copy()
        y = boston.target.astype(float).copy()
        mask = ~y.isna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        source = "openml"
    except (URLError, RemoteDisconnected, TimeoutError, Exception) as e:
        # Generic Exception catch here so any fetch_openml network failure falls back.
        st.warning(
            "Could not fetch Boston dataset from OpenML (network error). "
            "Falling back to local dataset if available. "
            "To avoid this, add `data/boston.csv` to the repo or commit a `model.pkl`."
        )
        if os.path.exists(DATA_CSV):
            df = pd.read_csv(DATA_CSV)
            # Expect target column named 'target' or 'MEDV' — try common names
            if "target" in df.columns:
                y = df["target"].astype(float).reset_index(drop=True)
                X = df.drop(columns=["target"]).reset_index(drop=True)
            elif "MEDV" in df.columns:
                y = df["MEDV"].astype(float).reset_index(drop=True)
                X = df.drop(columns=["MEDV"]).reset_index(drop=True)
            else:
                # if target column is last column
                y = df.iloc[:, -1].astype(float).reset_index(drop=True)
                X = df.iloc[:, :-1].reset_index(drop=True)
            source = "local_csv"
        else:
            # No local fallback available — raise an informative error
            raise RuntimeError(
                "Failed to fetch the Boston dataset from OpenML and no local "
                "data/boston.csv was found in the repository. "
                "Add data/boston.csv or commit model.pkl to skip training."
            ) from e

    # compute simple per-feature stats for slider bounds
    stats = {}
    for col in X.columns:
        try:
            stats[col] = {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "median": float(X[col].median())
            }
        except Exception:
            # For non-numeric columns, skip or attempt conversion
            stats[col] = {"min": 0.0, "max": 1.0, "median": 0.5}

    return X, y, stats, source

@st.cache_resource(show_spinner=False)
def load_or_train():
    """
    Load dataset using load_dataset(). If model.pkl exists, load it.
    Otherwise, train (as before) and save model.pkl.
    """
    X, y, stats, source = load_dataset()

    # If a model file already exists, load it and return immediately
    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            model = data["model"] if isinstance(data, dict) and "model" in data else data
            metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
            return model, metadata, stats, X, y, source
        except Exception as e:
            st.warning(f"Found {MODEL_PATH} but failed to load it: {e}. Will retrain model.")
            # fall-through to training

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])

    param_grid = {"model__alpha": np.logspace(-4, 4, 25)}
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    # Evaluate on test set
    y_pred = best.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metadata = {"rmse": float(rmse), "r2": float(r2), "mae": float(mae), "best_params": gs.best_params_}
    # Save the model and metadata
    joblib.dump({"model": best, "metadata": metadata}, MODEL_PATH)

    return best, metadata, stats, X, y, source

def train_and_save_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])

    param_grid = {"model__alpha": np.logspace(-4, 4, 25)}
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    # Evaluate on test set
    y_pred = best.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Save with metadata
    metadata = {"rmse": float(rmse), "r2": float(r2), "mae": float(mae), "best_params": gs.best_params_}
    joblib.dump({"model": best, "metadata": metadata}, MODEL_PATH)
    return best, metadata

def main():
    st.set_page_config(page_title="PredictHousePrices — Demo", layout="wide")
    st.title("🏠 PredictHousePrices — Interactive demo")
    st.markdown("Small demo that predicts Boston house prices using a Ridge model. Use the sidebar to change feature values and see predictions.")

    model, metadata, stats, X, y, _ = load_or_train()

    st.sidebar.markdown("### Model info")
    if metadata:
        st.sidebar.write(f"- RMSE (test): {metadata.get('rmse', 'N/A'):.3f}")
        st.sidebar.write(f"- R² (test): {metadata.get('r2', 'N/A'):.3f}")
        st.sidebar.write(f"- MAE (test): {metadata.get('mae', 'N/A'):.3f}")
        st.sidebar.write(f"- Best params: {metadata.get('best_params', {})}")
    else:
        st.sidebar.write("Model metadata not available.")

    input_df = build_input_ui(stats)
    st.subheader("Input")
    st.dataframe(input_df.T, height=200)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            try:
                pred = model.predict(input_df)[0]
                st.metric(label="Predicted median house price (in $1000s)", value=f"{pred:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.subheader("Dataset summary (used to build sliders)")
    st.write(f"Samples: {len(X)}")
    st.dataframe(pd.concat([X.describe().T[["min","50%","max"]].rename(columns={"50%":"median"})], axis=1))

    st.markdown("---")
    st.caption("Notes: The app will train and save a `model.pkl` on first run if none exists. Training is small but may take a few seconds on cold start.")

    # Allow download of the saved model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            st.download_button("Download saved model (model.pkl)", f, file_name="model.pkl")

if __name__ == "__main__":
    main()
