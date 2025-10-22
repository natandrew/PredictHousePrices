# PredictHousePrices

**House price prediction using the Boston dataset (scikit-learn)**

---

## Project overview

This repository contains an experiment that trains and evaluates linear models to predict house prices using the classic `boston` dataset from `openml` (accessed with `sklearn.datasets.fetch_openml`). The work is implemented in a Jupyter notebook and demonstrates a reproducible machine-learning pipeline: imputation, scaling, model fitting, and hyperparameter tuning using `GridSearchCV`.

## Demo

A live interactive demo is available via Streamlit. The demo allows you to change the input features (same features used in the notebook) and see predicted median house price.

**Run locally**

```bash
# (recommended) create venv and install requirements
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

# run the app
streamlit run streamlit_app.py

## What you'll find here

* A Jupyter notebook that:

  * Loads the Boston dataset
  * Applies a pipeline with `SimpleImputer` (median) and `StandardScaler`
  * Trains a `LinearRegression` baseline
  * Trains a `Ridge` regression with a grid search over `alpha` (logspace)
  * Reports evaluation metrics (RMSE, R², MAE)

## Quick results (from the notebook)

> These are the printed outputs that appear when the notebook was run:

* **Linear Regression**

  * Test RMSE: **4.929**
  * Test R² : **0.669**
  * Test MAE : **3.189**

* **Ridge (GridSearchCV)**

  * Best alpha (found by CV): **~2.154434690031882**
  * CV (approx) RMSE: **4.8628**
  * Test RMSE: **4.9334**
  * Test R²: **0.6681**
  * Test MAE: **3.1827**

> Note: CV RMSE is computed from the cross-validation fold scores; the final test RMSE is computed on the held-out test set.

## Why the pipeline is structured this way

* **Imputation**: The notebook uses median imputation (`SimpleImputer(strategy="median")`) to handle any missing values.
* **Scaling**: `StandardScaler` centers and scales the features, which helps models that rely on regularization (like Ridge).
* **Pipeline**: Using `sklearn.pipeline.Pipeline` keeps preprocessing and modeling bundled so `GridSearchCV` tunes hyperparameters correctly with preprocessing inside cross-validation.

## How to run locally

1. Clone the repo:

```bash
git clone https://github.com/natandrew/PredictHousePrices.git
cd PredictHousePrices
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install pandas numpy scikit-learn jupyterlab
```

4. Open and run the notebook:

```bash
jupyter lab  # or `jupyter notebook`
# then open the notebook file and run cells
```

5. (Headless execution) to run the notebook end-to-end from the command line:

```bash
pip install nbconvert
jupyter nbconvert --to notebook --execute your_notebook.ipynb --output executed_notebook.ipynb
```

## Reproducibility notes

* The notebook uses `train_test_split(..., random_state=42)` so the train/test split is reproducible.
* The `GridSearchCV` is run with `cv=5` and `n_jobs=-1` in the notebook; these settings affect runtime but not the randomness of results.

## Ethical note about the Boston dataset

The Boston housing dataset is a historical, educational dataset. It has known shortcomings (small size, potential fairness/representation concerns) and is not recommended for production decision-making. Consider using larger, more recent, and ethically-sourced datasets for real-world projects.


## Contact

I'm open to contract and full-time opportunities. Connect with me on LinkedIn: [Nat Andrew](https://www.linkedin.com/in/natandrew).

---
