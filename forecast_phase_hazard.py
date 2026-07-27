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
# CREATE PHASE HAZARD TARGET
# ==========================================================

def create_hazard_target(df):

    print("\n" + "="*70)
    print("CREATING PHASE HAZARD TARGET")
    print("="*70)

    df = df.copy()

    df["next_phase"] = df["demand_phase"].shift(-1)

    df = df.dropna(subset=["next_phase"])

    df["phase_hazard"] = (

        df["demand_phase"]

        !=

        df["next_phase"]

    ).astype(int)

    print()

    print(

        df[[
            "sales_date",
            "demand_phase",
            "next_phase",
            "phase_hazard"
        ]].head(20)

    )

    print()

    print("Hazard Distribution")

    print(df["phase_hazard"].value_counts())

    return df
# ==========================================================
# PREPARE HAZARD FEATURES
# ==========================================================

def prepare_features(df):

    print("\n" + "="*70)
    print("PREPARING HAZARD FEATURES")
    print("="*70)

    exclude = [

        "id",
        "sales_date",
        "product",

        "next_phase",
        "phase_hazard",

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
        "demand_phase",
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

    y = df["phase_hazard"]

    X = X.apply(

        pd.to_numeric,

        errors="coerce"

    )

    X = X.fillna(0).astype(float)

    return (

        X,
        y,
        feature_columns

    )
# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def split_train_test(

    X,
    y

):

    print("\n" + "="*70)
    print("TRAIN TEST SPLIT")
    print("="*70)

    split = int(

        len(X)

        * TRAIN_RATIO

    )

    X_train = X.iloc[:split]

    X_test = X.iloc[split:]

    y_train = y.iloc[:split]

    y_test = y.iloc[split:]

    print()

    print("Train :", len(X_train))

    print("Test  :", len(X_test))

    return (

        X_train,
        X_test,
        y_train,
        y_test

    )
# ==========================================================
# TRAIN PHASE HAZARD MODEL
# ==========================================================

def train_phase_hazard_model(

    X_train,
    y_train

):

    print("\n" + "="*70)
    print("TRAINING PHASE HAZARD MODEL")
    print("="*70)

    model = XGBClassifier(

        objective="binary:logistic",

        n_estimators=300,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.80,

        colsample_bytree=0.80,

        eval_metric="logloss",

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
# PREDICT PHASE HAZARD
# ==========================================================

def predict_phase_hazard(

    model,
    X_test,
    y_test

):

    print("\n" + "="*70)
    print("PREDICTING PHASE HAZARD")
    print("="*70)

    probability = model.predict_proba(X_test)[:,1]

    prediction = (

        probability >= 0.50

    ).astype(int)

    results = pd.DataFrame({

        "actual_hazard": y_test.values,

        "predicted_hazard": prediction,

        "hazard_probability": probability

    })

    print()

    print(results.head(20))

    return results
# ==========================================================
# EVALUATE HAZARD MODEL
# ==========================================================

def evaluate_hazard_model(results):

    print("\n" + "="*70)
    print("PHASE HAZARD MODEL RESULTS")
    print("="*70)

    accuracy = accuracy_score(

        results["actual_hazard"],

        results["predicted_hazard"]

    )

    precision = precision_score(

        results["actual_hazard"],

        results["predicted_hazard"]

    )

    recall = recall_score(

        results["actual_hazard"],

        results["predicted_hazard"]

    )

    f1 = f1_score(

        results["actual_hazard"],

        results["predicted_hazard"]

    )

    print()

    print("Accuracy :", round(accuracy,4))
    print("Precision:", round(precision,4))
    print("Recall   :", round(recall,4))
    print("F1 Score :", round(f1,4))

    print()

    print("Classification Report")

    print(

        classification_report(

            results["actual_hazard"],

            results["predicted_hazard"]

        )

    )

    print()

    print("Confusion Matrix")

    print(

        confusion_matrix(

            results["actual_hazard"],

            results["predicted_hazard"]

        )

    )
if __name__ == "__main__":

    df = load_dataset()

    df = create_hazard_target(df)

    X, y, feature_columns = prepare_features(df)

    X_train, X_test, y_train, y_test = split_train_test(

        X,
        y

    )

    model = train_phase_hazard_model(

        X_train,
        y_train

    )

    results = predict_phase_hazard(

        model,
        X_test,
        y_test

    )

    evaluate_hazard_model(

        results

    )