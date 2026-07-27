# ==========================================================
# PHASE 4
# PHASE FORECASTABILITY ANALYSIS
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor

from load_supabase import supabase

# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42
TRAIN_RATIO = 0.80

# ==========================================================
# LOAD FORECAST DATASET
# ==========================================================

def load_forecast_dataset(product="Overall"):

    print("\n" + "="*70)
    print("LOADING FORECAST DATASET")
    print("="*70)

    response = (

        supabase

        .table("final_training_data")

        .select("*")

        .eq("product", product)

        .order("sales_date")

        .execute()

    )

    df = pd.DataFrame(response.data)

    if df.empty:
        raise ValueError(
            "No forecasting dataset found."
        )

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    print()

    print("Rows :", len(df))

    return df

# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

def split_train_test(df):

    print("\n" + "=" * 70)
    print("CREATING TRAIN / TEST SPLIT")
    print("=" * 70)

    df = df.sort_values(
        "sales_date"
    ).reset_index(drop=True)

    split = int(

        len(df) *

        TRAIN_RATIO

    )

    train = df.iloc[:split].copy()

    test = df.iloc[split:].copy()

    print()

    print("Train :", len(train))

    print("Test  :", len(test))

    print()

    print("Train Period")

    print(

        train["sales_date"].min(),

        "->",

        train["sales_date"].max()

    )

    print()

    print("Test Period")

    print(

        test["sales_date"].min(),

        "->",

        test["sales_date"].max()

    )

    return train, test
# ==========================================================
# PREPARE FORECAST FEATURES
# ==========================================================

def prepare_forecast_features(train_df, test_df):

    print("\n" + "=" * 70)
    print("PREPARING FORECAST FEATURES")
    print("=" * 70)

    # ------------------------------------------------------
    # Columns NOT allowed for prediction
    # ------------------------------------------------------

    exclude = [

        "id",
        "sales_date",
        "product",
        "demand",

        "prediction",
        "absolute_error",
        "squared_error",
        "percentage_error",
        "model_name",

        "created_at",

        "forecast_error",
        "forecast_error_trend",
        "forecast_error_std",

        "role"      # ← add this

    ]

    features = [

        c for c in train_df.columns

        if c not in exclude

    ]

    X_train = train_df[features].copy()

    y_train = train_df["demand"].copy()

    X_test = test_df[features].copy()

    y_test = test_df["demand"].copy()

    print()

    print("Features :", len(features))

    print()

    print(features)

    return (

        X_train,

        y_train,

        X_test,

        y_test,

        features

    )
# ==========================================================
# TRAIN BASELINE FORECAST MODEL
# ==========================================================

def train_reference_forecaster(

    X_train,
    y_train,
    X_test,
    y_test,
    test_df,
    model_name="Reference"

):

    print("\n" + "=" * 70)
    print(f"TRAINING {model_name.upper()}")
    print("=" * 70)

    model = XGBRegressor(

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=RANDOM_STATE,

        n_jobs=-1

    )

    model.fit(

        X_train,

        y_train

    )

    predictions = model.predict(

        X_test

    )
    results = test_df.copy()

    results["prediction"] = predictions

    results["absolute_error"] = np.abs(
        results["demand"] - results["prediction"]
    )

    results["squared_error"] = (
        results["demand"] - results["prediction"]
    ) ** 2

    results["percentage_error"] = (
        results["absolute_error"] /
        np.maximum(results["demand"], 1e-6)
    ) * 100

    mae = mean_absolute_error(

        y_test,

        predictions

    )

    rmse = np.sqrt(

        mean_squared_error(

            y_test,

            predictions

        )

    )

    print()

    print(f"MAE  : {mae:.3f}")

    print(f"RMSE : {rmse:.3f}")

    return model, results
# ==========================================================
# PHASE FORECASTABILITY ANALYSIS
# ==========================================================

def analyse_phase_forecastability(results):

    print("\n" + "=" * 70)
    print("PHASE FORECASTABILITY ANALYSIS")
    print("=" * 70)

    phase_summary = (

        results

        .groupby("demand_phase")

        .agg(

            observations=("demand", "count"),

            MAE=("absolute_error", "mean"),

            RMSE=("squared_error",
                  lambda x: np.sqrt(np.mean(x))),

            MAPE=("percentage_error", "mean"),

            stability=("phase_stability", "mean"),

            entropy=("phase_entropy", "mean"),

            pagerank=("pagerank", "mean"),

            confidence=("phase_confidence", "mean")

        )

        .reset_index()

    )

    # -----------------------------------------
    # Forecastability Index
    # -----------------------------------------

    phase_summary["forecastability"] = pd.cut(

        phase_summary["MAE"],

        bins=[
            -np.inf,
            phase_summary["MAE"].quantile(0.25),
            phase_summary["MAE"].quantile(0.50),
            phase_summary["MAE"].quantile(0.75),
            np.inf
        ],

        labels=[
            "Very High",
            "High",
            "Medium",
            "Low"
        ]

    )

    phase_summary = phase_summary.round(3)

    print()
    print(phase_summary)

    return phase_summary
# ==========================================================
# PREPARE BASELINE FEATURES
# ==========================================================

def prepare_baseline_features(train_df, test_df):

    exclude = [

        "id",
        "sales_date",
        "product",
        "demand",

        "prediction",
        "absolute_error",
        "squared_error",
        "percentage_error",
        "model_name",

        "created_at",

        # Phase features
        "demand_phase",
        "phase_confidence",
        "phase_stability",
        "phase_entropy",
        "rare_phase_score",
        "phase_potential",
        "phase_velocity",
        "phase_curvature",

        # Graph features
        "degree_centrality",
        "betweenness",
        "closeness",
        "pagerank",

        "role"

    ]

    features = [

        c for c in train_df.columns

        if c not in exclude

    ]

    X_train = train_df[features].copy()

    X_test = test_df[features].copy()

    y_train = train_df["demand"]

    y_test = test_df["demand"]

    print()

    print("Baseline Features :", len(features))

    print(features)

    return (

        X_train,
        y_train,
        X_test,
        y_test

    )
# ==========================================================
# MODEL COMPARISON
# ==========================================================

def compare_models(

    baseline,

    phase

):

    print("\n" + "=" * 70)
    print("BASELINE vs PHASE-AWARE")
    print("=" * 70)

    comparison = pd.DataFrame({

        "Model":[

            "Baseline",

            "Phase-Aware"

        ],

        "MAE":[

            baseline["absolute_error"].mean(),

            phase["absolute_error"].mean()

        ],

        "RMSE":[

            np.sqrt(

                baseline["squared_error"].mean()

            ),

            np.sqrt(

                phase["squared_error"].mean()

            )

        ],

        "MAPE":[

            baseline["percentage_error"].mean(),

            phase["percentage_error"].mean()

        ]

    })

    print()

    print(

        comparison.round(3)

    )

    return comparison
if __name__ == "__main__":

    forecast_df = load_forecast_dataset()

    train_df, test_df = split_train_test(
        forecast_df
    )

    print("\n" + "=" * 70)
    print("DATA LEAKAGE CHECK")
    print("=" * 70)

    assert (
        train_df["sales_date"].max()
        <
        test_df["sales_date"].min()
    )

    print("✓ Chronological split verified.")
    # ==========================================================
    # BASELINE MODEL
    # ==========================================================

    (
        X_train_base,
        y_train_base,
        X_test_base,
        y_test_base

    ) = prepare_baseline_features(

        train_df,

        test_df

    )

    baseline_model, baseline_results = train_reference_forecaster(

        X_train_base,

        y_train_base,

        X_test_base,

        y_test_base,

        test_df,
        "Baseline"

    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names

    ) = prepare_forecast_features(

        train_df,

        test_df

    )
    model, results = train_reference_forecaster(

        X_train,

        y_train,

        X_test,

        y_test,

        test_df,
        "Phase-Aware"

    )
    phase_summary = analyse_phase_forecastability(

        results

    )
    comparison = compare_models(

        baseline_results,

        results

    )
