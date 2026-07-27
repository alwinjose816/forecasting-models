# ==========================================================
# PHASE AWARE FORECASTING
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from load_supabase import supabase


# ==========================================================
# CONFIGURATION
# ==========================================================

TRAIN_RATIO = 0.80
RANDOM_STATE = 42


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(product="Overall"):

    print("\n" + "="*70)
    print("LOADING DATASET")
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
    print(df.columns.tolist())

    if df.empty:
        raise ValueError("Dataset not found.")

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    print()

    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    return df
# ==========================================================
# CREATE FORECAST TARGET
# ==========================================================

def create_target(df):

    print("\n" + "="*70)
    print("CREATING FORECAST TARGET")
    print("="*70)

    df = df.copy()

    df["target"] = (

        df["demand"]

        .shift(-1)

    )

    df = df.dropna(subset=["target"])

    print()

    print(

        df[[
            "sales_date",
            "demand",
            "target"
        ]].head(15)

    )

    return df
# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def split_train_test(df):

    print("\n" + "="*70)
    print("TRAIN TEST SPLIT")
    print("="*70)

    split = int(

        len(df)

        * TRAIN_RATIO

    )

    train = df.iloc[:split].copy()

    test = df.iloc[split:].copy()

    print()

    print("Train :", len(train))
    print("Test  :", len(test))

    print()

    print(

        train.sales_date.min(),

        "->",

        train.sales_date.max()

    )

    print()

    print(

        test.sales_date.min(),

        "->",

        test.sales_date.max()

    )

    return train, test
# ==========================================================
# PREPARE FEATURES
# ==========================================================

def prepare_features(train_df, test_df):

    print("\n" + "="*70)
    print("PREPARING FEATURES")
    print("="*70)

    exclude = [

        "id",
        "sales_date",
        "product",

        "target",
        "demand",

        "prediction",
        "absolute_error",
        "squared_error",
        "percentage_error",
        "model_name",

        "role",
        "ptri_level",

        "created_at",

        # avoid leakage
        "phase_energy",
        "dynamic_alpha",
        "latent_phase_potential",
        "latent_phase_velocity",
        "latent_phase_curvature"

    ]

    feature_columns = [

        c for c in train_df.columns

        if c not in exclude

    ]

    print()

    print("Number of Features :", len(feature_columns))

    print()

    print(feature_columns)

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]

    y_train = train_df["target"]
    y_test = test_df["target"]

    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    return (

        X_train,
        y_train,
        X_test,
        y_test,
        feature_columns

    )
# ==========================================================
# TRAIN PHASE MODELS
# ==========================================================

def train_phase_models(train_df, feature_columns):

    print("\n" + "="*70)
    print("TRAINING PHASE MODELS")
    print("="*70)

    models = {}

    phases = sorted(

        train_df["demand_phase"].unique()

    )

    for phase in phases:

        phase_data = train_df[

            train_df["demand_phase"] == phase

        ].copy()

        print()

        print(

            "Phase",

            phase,

            "Rows :",

            len(phase_data)

        )

        if len(phase_data) < 30:

            print("Skipped")

            continue

        X = phase_data[feature_columns]

        y = phase_data["target"]

        model = XGBRegressor(

            n_estimators=300,

            max_depth=5,

            learning_rate=0.05,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=42

        )

        X = X.astype(float)

        y = y.astype(float)
        model.fit(X, y)

        models[phase] = model

    return models
# ==========================================================
# PHASE-AWARE PREDICTION
# ==========================================================

def phase_aware_prediction(
    phase_models,
    test_df,
    feature_columns
):

    print("\n" + "="*70)
    print("PHASE-AWARE PREDICTION")
    print("="*70)

    predictions = []

    used_models = []

    for _, row in test_df.iterrows():

        phase = row["demand_phase"]

        if phase not in phase_models:

            predictions.append(np.nan)
            used_models.append(None)

            continue

        x = pd.DataFrame(

            [row[feature_columns].values],

            columns=feature_columns

        ).astype(float)

        pred = phase_models[phase].predict(x)[0]

        predictions.append(pred)

        used_models.append(phase)

    results = test_df.copy()

    results["prediction"] = predictions

    results["model_phase"] = used_models

    print()

    print(

        results[[

            "sales_date",
            "demand_phase",
            "prediction"

        ]].head(20)

    )

    return results
# ==========================================================
# EVALUATE PHASE-AWARE FORECAST
# ==========================================================

def evaluate_forecast(results):

    print("\n" + "="*70)
    print("PHASE-AWARE FORECAST RESULTS")
    print("="*70)

    valid = (

        results

        .dropna(

            subset=["prediction"]

        )

    )

    mae = mean_absolute_error(

        valid["target"],

        valid["prediction"]

    )

    rmse = np.sqrt(

        mean_squared_error(

            valid["target"],

            valid["prediction"]

        )

    )

    r2 = r2_score(

        valid["target"],

        valid["prediction"]

    )

    print()

    print("Samples :", len(valid))

    print()

    print("MAE  :", round(mae,2))

    print("RMSE :", round(rmse,2))

    print("R²   :", round(r2,4))

    return valid
if __name__ == "__main__":

    df = load_dataset()

    df = create_target(df)

    train_df, test_df = split_train_test(df)

    (

        X_train,
        y_train,
        X_test,
        y_test,
        feature_columns

    ) = prepare_features(

        train_df,

        test_df

    )

    phase_models = train_phase_models(

        train_df,

        feature_columns

    )

    results = phase_aware_prediction(

        phase_models,

        test_df,

        feature_columns

    )

    evaluate_forecast(results)
