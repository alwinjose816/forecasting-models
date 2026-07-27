# ==========================================================
# GENOME → PHASE GENERATOR
# HDGEN Stage 4
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import joblib

import numpy as np
import pandas as pd

from load_supabase import supabase

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier

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
# LOAD HISTORICAL DATA
# ==========================================================

def load_history(product="Overall"):

    print("\n" + "="*70)
    print("LOADING HISTORICAL DATA")
    print("="*70)

    response = (

        supabase

        .table("final_training_data")

        .select(

            "sales_date,"
            "product,"
            "demand_phase"

        )

        .eq("product", product)

        .order("sales_date")

        .execute()

    )

    df = pd.DataFrame(response.data)

    df["sales_date"] = pd.to_datetime(

        df["sales_date"]

    )

    print()

    print("Rows :", len(df))

    print(df.head())

    return df
# ==========================================================
# LOAD PREDICTED GENOME
# ==========================================================

def load_predicted_genome(product="Overall"):

    print("\n" + "="*70)
    print("LOADING PREDICTED GENOME")
    print("="*70)

    response = (

        supabase

        .table("predicted_genome")

        .select("*")

        .eq("product", product)

        .order("sales_date")

        .execute()

    )

    genome = pd.DataFrame(response.data)

    genome["sales_date"] = pd.to_datetime(

        genome["sales_date"]

    )

    print()

    print("Rows :", len(genome))

    print("Columns :", len(genome.columns))

    return genome
# ==========================================================
# CREATE FUTURE PHASE
# ==========================================================

def create_future_phase(df):

    print("\n" + "="*70)
    print("CREATING FUTURE PHASE TARGET")
    print("="*70)

    df = df.copy()

    df["future_phase"] = (

        df["demand_phase"]

        .shift(-1)

    )

    df = df.iloc[:-1].reset_index(drop=True)

    print()

    print(

        df[[

            "sales_date",

            "demand_phase",

            "future_phase"

        ]].head()

    )

    return df
# ==========================================================
# MERGE DATASETS
# ==========================================================

def merge_dataset(

    history,

    genome

):

    print("\n" + "="*70)
    print("MERGING DATASETS")
    print("="*70)

    merged = history.merge(

        genome,

        on=[

            "sales_date",

            "product"

        ],

        how="inner"

    )

    print()

    print("Rows :", len(merged))

    print("Columns :", len(merged.columns))

    return merged

# ==========================================================
# PREPARE RELIABILITY-WEIGHTED FEATURES
# ==========================================================

def prepare_features(df, product="Overall"):

    print("\n" + "="*70)
    print("PREPARING RELIABILITY-WEIGHTED FEATURES")
    print("="*70)

    # ---------------------------------------
    # Load Gene Reliability
    # ---------------------------------------

    response = (

        supabase

        .table("gene_reliability")

        .select(

            "gene,reliability"

        )

        .eq(

            "product",

            product

        )

        .execute()

    )

    reliability = pd.DataFrame(response.data)

    reliability = dict(

        zip(

            reliability["gene"],

            reliability["reliability"]

        )

    )

    # ---------------------------------------
    # Genome Features
    # ---------------------------------------

    weighted_features = []

    velocity_features = []

    acceleration_features = []

    genome_velocity = []

    genome_acceleration = []

    genome_direction = []

    for gene, weight in reliability.items():

        if gene not in df.columns:
            continue

        # -------------------------------
        # Genome
        # -------------------------------

        weighted_col = f"weighted_{gene}"

        df[weighted_col] = (

            df[gene]

            * weight

        )

        weighted_features.append(weighted_col)

        # -------------------------------
        # Velocity
        # -------------------------------

        velocity_col = f"velocity_{gene}"

        df[velocity_col] = (

            df[gene]

            -

            df[gene].shift(1)

        )

        velocity_features.append(velocity_col)

        # -------------------------------
        # Acceleration
        # -------------------------------

        acceleration_col = f"acceleration_{gene}"

        df[acceleration_col] = (

            df[velocity_col]

            -

            df[velocity_col].shift(1)

        )

        acceleration_features.append(acceleration_col)

    # ---------------------------------------
    # Remove NaNs created by shifts
    # ---------------------------------------

    df = df.dropna().reset_index(drop=True)
    duplicates = df.duplicated(
        subset=["sales_date", "product"]
    ).sum()

    print("Duplicate rows:", duplicates)

    if duplicates > 0:
        raise ValueError(
            "Duplicate sales_date/product rows found."
        )

    # ---------------------------------------
    # Final Feature Set
    # ---------------------------------------

    # ---------------------------------------
    # Genome Evolution Velocity
    # ---------------------------------------

    df["genome_velocity"] = np.sqrt(

        np.sum(

            np.square(

                df[velocity_features]

            ),

            axis=1

        )

    )

    # ---------------------------------------
    # Genome Evolution Acceleration
    # ---------------------------------------

    df["genome_acceleration"] = np.sqrt(

        np.sum(

            np.square(

                df[acceleration_features]

            ),

            axis=1

        )

    )

    # ---------------------------------------
    # Genome Evolution Direction
    # ---------------------------------------

    previous = df[weighted_features].shift(1)

    current = df[weighted_features]

    dot = (current * previous).sum(axis=1)

    norm1 = np.sqrt(

        (current ** 2).sum(axis=1)

    )

    norm2 = np.sqrt(

        (previous ** 2).sum(axis=1)

    )

    df["genome_direction"] = dot / (

        norm1 * norm2 + 1e-9

    )

    df = df.dropna().reset_index(drop=True)

    feature_columns = (

        weighted_features

        +

        [

            "genome_velocity",

            "genome_acceleration",

            "genome_direction"

        ]

    )
    X = df[feature_columns]

    y = df["future_phase"]
    print("\nMissing values")

    print(X.isna().sum())

    if X.isna().sum().sum() > 0:
        raise ValueError(
            "Missing values found in feature matrix."
        )

   

    print()

    print("Genome Features :", len(weighted_features))

    print("Evolution Features : 3")

    print("Total Features :", len(feature_columns))
    print()

    print(X.head())

    return (

        X,

        y,

        feature_columns

    )

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def train_test_dataset(X, y):

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

    return (

        X_train,
        X_test,

        y_train,
        y_test

    )
# ==========================================================
# TRAIN GENOME → PHASE GENERATOR
# ==========================================================

def train_phase_generator(

    X_train,
    y_train

):

    print("\n" + "="*70)
    print("TRAINING GENOME → PHASE GENERATOR")
    print("="*70)

    os.makedirs(

        "models",

        exist_ok=True

    )


    model = CatBoostClassifier(

        iterations=500,

        depth=6,

        learning_rate=0.03,

        loss_function="MultiClass",

        eval_metric="Accuracy",

        random_seed=RANDOM_STATE,

        verbose=False

    )

    model.fit(
        X_train,
        y_train
    )
    print("\nCatBoost Classes")
    print(model.classes_)

    joblib.dump(

        model,

        "models/genome_phase_generator.pkl"

    )

    print()

    print("Training Complete")

    return model
# ==========================================================
# PREDICT FUTURE PHASE
# ==========================================================

def predict_phase(
    model,
    X_test,
    y_test,
    y_train
):

    print("\n" + "="*70)
    print("PREDICTING FUTURE PHASE")
    print("="*70)

    prediction = model.predict(X_test)

    prediction = prediction.flatten()
    print("\nUnique predictions:")
    print(np.unique(prediction))

    print("\nPrediction dtype:")
    print(prediction.dtype)

    print("\nFirst 20 predictions:")
    print(prediction[:20])

    print("\nUnique y_train:")
    print(np.unique(y_train))

    probability = model.predict_proba(X_test)

    confidence = probability.max(axis=1)

    result = pd.DataFrame({

        "actual_phase": y_test.values,

        "predicted_phase": prediction,

        "confidence": confidence

    })

    print()

    print(result.head(20))

    return (

        prediction,

        confidence,

        result

    )
# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

def evaluate(

    y_test,

    prediction

):

    print("\n" + "="*70)
    print("PHASE GENERATOR RESULTS")
    print("="*70)

    accuracy = accuracy_score(

        y_test,

        prediction

    )

    precision = precision_score(

        y_test,

        prediction,

        average="weighted"

    )

    recall = recall_score(

        y_test,

        prediction,

        average="weighted"

    )

    f1 = f1_score(

        y_test,

        prediction,

        average="weighted"

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

            y_test,

            prediction

        )

    )

    print()

    print("Confusion Matrix")

    print(

        confusion_matrix(

            y_test,

            prediction

        )

    )
# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_importance(

    model,

    features

):

    print("\n" + "="*70)
    print("GENOME FEATURE IMPORTANCE")
    print("="*70)

    importance = pd.DataFrame({

        "feature": features,

        "importance": model.feature_importances_

    })

    importance = (

        importance

        .sort_values(

            "importance",

            ascending=False

        )

    )

    print()

    print(importance)

    return importance

# ==========================================================
# SAVE PREDICTED PHASE
# ==========================================================

def save_predicted_phase(

        dataset,

        prediction,

        confidence

):

    print("\n" + "="*70)
    print("SAVING PREDICTED PHASE")
    print("="*70)

    split = int(len(dataset) * TRAIN_RATIO)

    
    result = dataset.iloc[split:].reset_index(drop=True)

    rows = pd.DataFrame({

        "sales_date":
            result["sales_date"].dt.strftime("%Y-%m-%d"),

        "product":
            result["product"],

        "actual_phase":
            result["future_phase"].astype(int),

        "predicted_phase":
            prediction.astype(int),

        "confidence":
            confidence,

        "model_name":
            "Genome Phase Generator"

    })

    supabase.table(

        "predicted_phase"

    ).upsert(

        rows.to_dict("records"),

        on_conflict="sales_date,product"

    ).execute()

    print()

    print("Saved :", len(rows))
# ==========================================================
# SAVE FEATURE IMPORTANCE
# ==========================================================

def save_feature_importance(

        importance

):

    rows = importance.copy()

    rows["model_name"] = "Genome Phase Generator"

    supabase.table(

        "genome_phase_importance"

    ).upsert(

        rows.to_dict("records"),

        on_conflict="feature"

    ).execute()

    print()

    print(

        "Feature Importance Saved :", len(rows)

    )
# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    history = load_history()

    history = create_future_phase(

        history

    )

    genome = load_predicted_genome()

    dataset = merge_dataset(

        history,

        genome

    )

    X, y, features = prepare_features(

        dataset,

        "Overall"

    )

    (

        X_train,

        X_test,

        y_train,

        y_test

    ) = train_test_dataset(

        X,

        y

    )

    model = train_phase_generator(

        X_train,

        y_train

    )

    (
        prediction,
        confidence,
        result
    ) = predict_phase(

        model,

        X_test,

        y_test,

        y_train

    )

    evaluate(

        y_test,

        prediction

    )

    importance = feature_importance(

        model,

        features

    )

    save_feature_importance(

        importance

    )

    save_predicted_phase(

        dataset,

        prediction,

        confidence

    )
    