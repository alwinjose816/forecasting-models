# ==========================================================
# DYNAMIC PHASE PREDICTION
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report

)

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

        raise ValueError(

            "Dataset not found."

        )

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
# CREATE NEXT PHASE TARGET
# ==========================================================

def create_phase_target(df):

    print("\n" + "="*70)
    print("CREATING NEXT PHASE TARGET")
    print("="*70)

    df = df.copy()

    # Tomorrow's phase
    df["target_phase"] = (

        df["demand_phase"]

        .shift(-1)

    )

    # Remove last row
    df = df.dropna(

        subset=["target_phase"]

    )

    df["target_phase"] = (

        df["target_phase"]

        .astype(int)

    )

    print()

    print(

        df[[

            "sales_date",

            "demand_phase",

            "target_phase"

        ]].head(20)

    )

    print()

    print("Target Phase Distribution")

    print()

    print(

        df["target_phase"]

        .value_counts()

        .sort_index()

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

    train_df = (

        df.iloc[:split]

        .copy()

    )

    test_df = (

        df.iloc[split:]

        .copy()

    )

    print()

    print("Train :", len(train_df))

    print("Test  :", len(test_df))

    print()

    print(

        train_df["sales_date"].min(),

        "->",

        train_df["sales_date"].max()

    )

    print()

    print(

        test_df["sales_date"].min(),

        "->",

        test_df["sales_date"].max()

    )

    return train_df, test_df
# ==========================================================
# PREPARE FEATURES
# ==========================================================

def prepare_features(

    train_df,

    test_df

):

    print("\n" + "="*70)
    print("PREPARING FEATURES")
    print("="*70)

    exclude = [

        # identifiers
        "id",
        "sales_date",
        "product",

        # target
        "target_phase",

        # outputs
        "prediction",
        "absolute_error",
        "squared_error",
        "percentage_error",
        "model_name",

        # text columns
        "role",
        "ptri_level",

        # timestamp
        "created_at",

        # remove derived recursive variables
        "phase_energy",
        "dynamic_alpha"

    ]

    feature_columns = [

        c

        for c in train_df.columns

        if c not in exclude

    ]

    print()

    print("Number of Features :", len(feature_columns))

    print()

    print(feature_columns)

    X_train = train_df[feature_columns].copy()

    X_test = test_df[feature_columns].copy()

    y_train = train_df["target_phase"]

    y_test = test_df["target_phase"]

    # convert to numeric

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

    X_train = X_train.astype(float)

    X_test = X_test.astype(float)

    y_train = y_train.astype(int)

    y_test = y_test.astype(int)

    return (

        X_train,

        y_train,

        X_test,

        y_test,

        feature_columns

    )
# ==========================================================
# TRAIN DYNAMIC PHASE PREDICTION MODEL
# ==========================================================

from xgboost import XGBClassifier


def train_phase_prediction_model(

    X_train,

    y_train

):

    print("\n" + "="*70)
    print("TRAINING DYNAMIC PHASE PREDICTION MODEL")
    print("="*70)

    model = XGBClassifier(

        objective="multi:softprob",

        n_estimators=300,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.80,

        colsample_bytree=0.80,

        random_state=42,

        eval_metric="mlogloss"

    )

    model.fit(

        X_train,

        y_train

    )

    print()

    print("Training Complete")

    return model
# ==========================================================
# PREDICT NEXT PHASE
# ==========================================================

def predict_next_phase(

    model,

    X_test,

    test_df

):

    print("\n" + "="*70)
    print("PREDICTING NEXT DEMAND PHASE")
    print("="*70)

    predicted_phase = model.predict(

        X_test

    )

    probabilities = model.predict_proba(

        X_test

    )

    results = test_df.copy()

    results["predicted_phase"] = predicted_phase

    results["prediction_confidence"] = (

        probabilities.max(axis=1)

    )

    print()

    print(

        results[[

            "sales_date",

            "demand_phase",

            "target_phase",

            "predicted_phase",

            "prediction_confidence"

        ]].head(20)

    )

    return (

        results,

        probabilities

    )
# ==========================================================
# EVALUATE PHASE PREDICTION
# ==========================================================

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    classification_report,

    confusion_matrix

)


def evaluate_phase_prediction(

    results

):

    print("\n" + "="*70)
    print("PHASE PREDICTION RESULTS")
    print("="*70)

    y_true = results["target_phase"]

    y_pred = results["predicted_phase"]

    accuracy = accuracy_score(

        y_true,

        y_pred

    )

    precision = precision_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    print()

    print("Accuracy :", round(accuracy,4))

    print("Precision:", round(precision,4))

    print("Recall   :", round(recall,4))

    print("F1 Score :", round(f1,4))

    print()

    print("Classification Report\n")

    print(

        classification_report(

            y_true,

            y_pred,

            zero_division=0

        )

    )

    print()

    print("Confusion Matrix\n")

    print(

        confusion_matrix(

            y_true,

            y_pred

        )

    )
# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    data = load_dataset()

    data = create_phase_target(data)

    train_df, test_df = split_train_test(data)

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

    model = train_phase_prediction_model(

        X_train,

        y_train

    )

    results, probabilities = predict_next_phase(

        model,

        X_test,

        test_df

    )

    evaluate_phase_prediction(

        results

    )