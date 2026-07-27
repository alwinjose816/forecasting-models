# ==========================================================
# PHASE HAZARD MODEL
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from load_supabase import supabase

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
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

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    print()

    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    print()

    print(df.columns.tolist())

    return df
# ==========================================================
# CREATE TIME TO TRANSITION TARGET
# ==========================================================

def create_time_to_transition(df):

    print("\n" + "="*70)
    print("CREATING TIME TO TRANSITION")
    print("="*70)

    df = df.copy()

    df["phase_change"] = (

        df["demand_phase"]

        !=

        df["demand_phase"].shift()

    )

    transition_rows = df.index[df["phase_change"]].tolist()

    remaining = []

    for i in range(len(df)):

        future = [

            t

            for t in transition_rows

            if t > i

        ]

        if len(future) == 0:

            remaining.append(np.nan)

        else:

            remaining.append(

                future[0] - i

            )

    df["time_to_transition"] = remaining

    # Drop rows only where the target is missing
    df = df.dropna(subset=["time_to_transition"])

    df["time_to_transition"] = (
        df["time_to_transition"]
        .astype(int)
    )

    print()

    print("Rows after target creation :", len(df))

    print()

    print(
        df[
            [
                "sales_date",
                "demand_phase",
                "time_to_transition"
            ]
        ].head(30)
    )

    return df
# ==========================================================
# PREPARE FEATURES
# ==========================================================

def prepare_features(df):

    print("\n" + "="*70)
    print("PREPARING FEATURES")
    print("="*70)

    exclude = [

        "id",
        "sales_date",
        "product",

        "time_to_transition",

        # Remove leakage
        "phase_change",

        "prediction",
        "absolute_error",
        "squared_error",
        "percentage_error",
        "model_name",

        "role",
        "ptri_level",

        "created_at",

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

    y = df["time_to_transition"]

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
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# ==========================================================
# TRAIN TIME TO TRANSITION MODEL
# ==========================================================

def train_time_to_transition_model(X_train, y_train):

    print("\n" + "="*70)
    print("TRAINING TIME TO TRANSITION MODEL")
    print("="*70)

    model = XGBRegressor(

        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,

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
# PREDICT TIME TO TRANSITION
# ==========================================================

def predict_time_to_transition(
    model,
    X_test,
    y_test
):

    print("\n" + "="*70)
    print("PREDICTING TIME TO TRANSITION")
    print("="*70)

    prediction = model.predict(X_test)

    results = pd.DataFrame({

        "actual_days": y_test.values,
        "predicted_days": prediction

    })

    print()
    print(results.head(20))

    return prediction
# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

def evaluate_model(
    y_test,
    prediction
):

    print("\n" + "="*70)
    print("TIME TO TRANSITION RESULTS")
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
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")
# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    df = load_dataset()

    df = create_time_to_transition(df)

    X, y, feature_columns = prepare_features(df)

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y
    )

    model = train_time_to_transition_model(
        X_train,
        y_train
    )

    prediction = predict_time_to_transition(
        model,
        X_test,
        y_test
    )

    evaluate_model(
        y_test,
        prediction
    )