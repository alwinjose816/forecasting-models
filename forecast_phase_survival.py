# ==========================================================
# PHASE SURVIVAL MODEL
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from load_supabase import supabase

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

TRAIN_RATIO = 0.80
RANDOM_STATE = 42
# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(product="Overall"):

    print("\n" + "="*70)
    print("LOADING PHASE DATASET")
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

    df["sales_date"] = pd.to_datetime(df["sales_date"])

    print()

    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    return df
# ==========================================================
# CREATE PHASE EPISODES
# ==========================================================

def create_phase_episodes(df):

    print("\n" + "="*70)
    print("CREATING PHASE EPISODES")
    print("="*70)

    df = df.copy()

    df["phase_change"] = (

        df["demand_phase"]

        !=

        df["demand_phase"].shift()

    )

    df["episode_id"] = (

        df["phase_change"]

        .cumsum()

    )

    print()

    print(

        df[[
            "sales_date",
            "demand_phase",
            "episode_id"
        ]].head(20)

    )

    print()

    print("Episodes :", df["episode_id"].nunique())

    return df
# ==========================================================
# COMPUTE REMAINING LIFE
# ==========================================================

def compute_remaining_life(df):

    print("\n" + "="*70)
    print("COMPUTING REMAINING LIFE")
    print("="*70)

    df = df.copy()

    df["phase_age"] = (

        df

        .groupby("episode_id")

        .cumcount()

        + 1

    )

    episode_length = (

        df

        .groupby("episode_id")

        .size()

        .rename("episode_length")

    )

    df = df.merge(

        episode_length,

        on="episode_id"

    )

    df["remaining_life"] = (

        df["episode_length"]

        -

        df["phase_age"]

    )

    print()

    print(

        df[[
            "sales_date",
            "demand_phase",
            "phase_age",
            "episode_length",
            "remaining_life"
        ]].head(30)

    )

    return df
# ==========================================================
# PREPARE SURVIVAL FEATURES
# ==========================================================

def prepare_features(df):

    print("\n" + "="*70)
    print("PREPARING SURVIVAL FEATURES")
    print("="*70)

    exclude = [

        "id",
        "sales_date",
        "product",

        "phase_change",
        "episode_id",

        "episode_length",
        "remaining_life",

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

        c

        for c in df.columns

        if c not in exclude

    ]

    print()

    print("Number of Features :", len(feature_columns))

    print()

    print(feature_columns)

    X = df[feature_columns].copy()

    y = df["remaining_life"]

    X = X.apply(

        pd.to_numeric,

        errors="coerce"

    )

    X = X.fillna(0).astype(float)

    return X, y, feature_columns
# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def split_train_test(X, y):

    print("\n" + "="*70)
    print("TRAIN TEST SPLIT")
    print("="*70)

    split = int(len(X) * TRAIN_RATIO)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print()

    print("Train :", len(X_train))
    print("Test  :", len(X_test))

    return X_train, X_test, y_train, y_test
# ==========================================================
# TRAIN SURVIVAL MODEL
# ==========================================================

def train_survival_model(
    X_train,
    y_train
):

    print("\n" + "="*70)
    print("TRAINING PHASE SURVIVAL MODEL")
    print("="*70)

    model = XGBRegressor(

        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,

        random_state=RANDOM_STATE

    )

    model.fit(

        X_train,

        y_train

    )

    print()

    print("Training Complete")

    return model
# ==========================================================
# PREDICT REMAINING LIFE
# ==========================================================

def predict_survival(

    model,

    X_test,

    y_test

):

    print("\n" + "="*70)
    print("PREDICTING REMAINING LIFE")
    print("="*70)

    prediction = model.predict(X_test)

    results = pd.DataFrame({

        "actual_remaining_life": y_test,

        "predicted_remaining_life": prediction

    })

    print()

    print(results.head(20))

    return prediction
# ==========================================================
# EVALUATE SURVIVAL MODEL
# ==========================================================

def evaluate_model(

    y_test,

    prediction

):

    print("\n" + "="*70)
    print("SURVIVAL MODEL RESULTS")
    print("="*70)

    mae = mean_absolute_error(

        y_test,

        prediction

    )

    rmse = np.sqrt(

        mean_squared_error(

            y_test,

            prediction

        )

    )

    r2 = r2_score(

        y_test,

        prediction

    )

    print()

    print("MAE :", round(mae,2))
    print("RMSE:", round(rmse,2))
    print("R²  :", round(r2,4))
if __name__ == "__main__":

    df = load_dataset()

    df = create_phase_episodes(df)

    df = compute_remaining_life(df)

    X, y, feature_columns = prepare_features(df)

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y
    )

    model = train_survival_model(
        X_train,
        y_train
    )

    prediction = predict_survival(
        model,
        X_test,
        y_test
    )

    evaluate_model(
        y_test,
        prediction
    )