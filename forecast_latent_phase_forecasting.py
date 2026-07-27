# ==========================================================
# LATENT PHASE FORECASTING
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

RANDOM_STATE = 42
TRAIN_RATIO = 0.80


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(product="Overall"):

    print("\n" + "="*70)
    print("LOADING LATENT PHASE DATASET")
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
        raise ValueError("Dataset not found.")

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    print()

    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    return df
# ==========================================================
# CREATE TARGET
# ==========================================================

def create_prediction_target(df):

    print("\n" + "="*70)
    print("CREATING LATENT PHASE TARGET")
    print("="*70)

    df = df.copy()

    df["target_phi"] = (

        df["latent_phase_potential"]

        .shift(-1)

    )

    df = (

        df

        .dropna(subset=["target_phi"])

        .reset_index(drop=True)

    )

    print()

    print(df[[
        "sales_date",
        "latent_phase_potential",
        "target_phi"
    ]].head(15))

    return df
# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def split_train_test(df):

    print("\n" + "="*70)
    print("CREATING TRAIN TEST SPLIT")
    print("="*70)

    split = int(

        len(df)

        * TRAIN_RATIO

    )

    train = (

        df.iloc[:split]

        .copy()

    )

    test = (

        df.iloc[split:]

        .copy()

    )

    print()

    print("Train :", len(train))
    print("Test  :", len(test))

    print()

    print(

        train["sales_date"].min(),

        "->",

        train["sales_date"].max()

    )

    print()

    print(

        test["sales_date"].min(),

        "->",

        test["sales_date"].max()

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

        "demand",

        "target_phi",

        "prediction",

        "created_at",
        "updated_at",

        "model_name",

        # strings
        "role",
        "ptri_level",

        # forecast outputs (LEAKAGE)
        "absolute_error",
        "squared_error",
        "percentage_error",

        # original phase variables
        "phase_potential",
        "phase_velocity",
        "phase_curvature",
        "latent_phase_velocity",
        "latent_phase_curvature",
        "phase_energy",
        "dynamic_alpha",
        "latent_phase_potential"

    ]

    features = [

        c for c in train_df.columns

        if c not in exclude

    ]

    X_train = train_df[features].copy()

    X_test = test_df[features].copy()

    y_train = train_df["target_phi"].copy()

    y_test = test_df["target_phi"].copy()

    print()

    print("Number of Features :", len(features))

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
# TRAIN LATENT PHASE MODEL
# ==========================================================

def train_latent_phase_model(

    X_train,
    y_train,
    X_test,
    y_test,
    test_df

):

    print("\n" + "="*70)
    print("TRAINING LATENT PHASE MODEL")
    print("="*70)

    model = XGBRegressor(

        n_estimators=400,

        learning_rate=0.05,

        max_depth=6,

        subsample=0.80,

        colsample_bytree=0.80,

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

    results["predicted_phi"] = predictions

    results["absolute_error"] = np.abs(

        results["target_phi"]

        -

        predictions

    )

    results["squared_error"] = (

        results["target_phi"]

        -

        predictions

    )**2

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

    r2 = r2_score(

        y_test,

        predictions

    )

    print()

    print(f"MAE  : {mae:.4f}")

    print(f"RMSE : {rmse:.4f}")

    print(f"R²   : {r2:.4f}")

    return model, results
# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    data = load_dataset()

    data = create_prediction_target(data)

    train_df, test_df = split_train_test(data)

    (

        X_train,

        y_train,

        X_test,

        y_test,

        feature_names

    ) = prepare_features(

        train_df,

        test_df

    )

    model, results = train_latent_phase_model(

        X_train,

        y_train,

        X_test,

        y_test,

        test_df

    )