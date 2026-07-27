# ==========================================================
# DYNAMIC TRANSITION PROBABILITY
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from load_supabase import supabase

from xgboost import XGBClassifier
# ==========================================================
# CONFIGURATION
# ==========================================================

TRAIN_RATIO = 0.80
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

    return df
# ==========================================================
# NEXT PHASE TARGET
# ==========================================================

def create_target(df):

    print("\n" + "="*70)
    print("CREATING NEXT PHASE TARGET")
    print("="*70)

    df = df.copy()

    df["target_phase"] = (

        df["demand_phase"]

        .shift(-1)

    )

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

        ]].head(15)

    )

    return df
# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def split_train_test(df):

    split = int(

        len(df)

        * TRAIN_RATIO

    )

    train = df.iloc[:split].copy()

    test = df.iloc[split:].copy()

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

        "target_phase",

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

        for c in train_df.columns

        if c not in exclude

    ]

    print()

    print("Features :", len(feature_columns))

    X_train = train_df[feature_columns].copy()

    X_test = test_df[feature_columns].copy()

    y_train = train_df["target_phase"]

    y_test = test_df["target_phase"]

    X_train = X_train.apply(

        pd.to_numeric,

        errors="coerce"

    )

    X_test = X_test.apply(

        pd.to_numeric,

        errors="coerce"

    )

    X_train = X_train.fillna(0).astype(float)

    X_test = X_test.fillna(0).astype(float)

    return (

        X_train,

        y_train,

        X_test,

        y_test,

        feature_columns

    )
# ==========================================================
# TRAIN DYNAMIC TRANSITION MODEL
# ==========================================================

def train_transition_model(

    X_train,

    y_train

):

    print("\n" + "="*70)
    print("TRAINING TRANSITION MODEL")
    print("="*70)

    model = XGBClassifier(

        objective="multi:softprob",

        n_estimators=300,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.80,

        colsample_bytree=0.80,

        eval_metric="mlogloss",

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    print()

    print("Training Complete")

    return model

# ==========================================================
# DYNAMIC TRANSITION PROBABILITY
# ==========================================================

def compute_transition_probability(

    model,

    X_test,

    test_df

):

    print("\n" + "="*70)
    print("COMPUTING DYNAMIC TRANSITION PROBABILITIES")
    print("="*70)

    probability = model.predict_proba(

        X_test

    )

    classes = model.classes_

    results = test_df.copy()

    for i, phase in enumerate(classes):

        results[

            f"prob_phase_{phase}"

        ] = probability[:, i]

    results["current_phase"] = results["demand_phase"]

    results["predicted_phase"] = model.predict(X_test)

    results["transition_confidence"] = (

        probability.max(axis=1)

    )

    print()

    cols = [

        "sales_date",

        "demand_phase",

        "predicted_phase",

        "transition_confidence"

    ]

    cols += [

        f"prob_phase_{p}"

        for p in classes

    ]

    print(

        results[cols].head(10)

    )

    return results
# ==========================================================
# DYNAMIC TRANSITION RISK
# ==========================================================

from scipy.stats import entropy


def compute_transition_risk(results):

    print("\n" + "="*70)
    print("COMPUTING DYNAMIC TRANSITION RISK")
    print("="*70)

    prob_cols = [

        c

        for c in results.columns

        if c.startswith("prob_phase_")

    ]

    probabilities = results[prob_cols].values

    # ----------------------------------
    # Transition Entropy
    # ----------------------------------

    transition_entropy = np.array([

        entropy(p)

        for p in probabilities

    ])

    # ----------------------------------
    # Maximum Probability
    # ----------------------------------

    maximum_probability = probabilities.max(axis=1)

    # ----------------------------------
    # Uncertainty
    # ----------------------------------

    uncertainty = 1 - maximum_probability

    # ----------------------------------
    # Dynamic PTR
    # ----------------------------------

    dynamic_ptr = (

        0.60 * transition_entropy +

        0.40 * uncertainty

    )

    results["transition_entropy"] = transition_entropy

    results["transition_uncertainty"] = uncertainty

    results["dynamic_transition_risk"] = dynamic_ptr

    print()

    print(

        results[[

            "sales_date",

            "transition_entropy",

            "transition_uncertainty",

            "dynamic_transition_risk"

        ]].head(20)

    )

    return results
# ==========================================================
# TRANSITION RISK LEVEL
# ==========================================================

def assign_transition_level(results):

    print("\n" + "="*70)
    print("ASSIGNING RISK LEVEL")
    print("="*70)

    results["risk_level"] = pd.qcut(

        results["dynamic_transition_risk"],

        q=3,

        labels=[

            "Low",

            "Medium",

            "High"

        ]

    )

    print()

    print(

        results["risk_level"]

        .value_counts()

    )

    return results
# ==========================================================
# SAVE RESULTS
# ==========================================================

def save_transition_probability(results):

    print("\n" + "="*70)
    print("SAVING DYNAMIC TRANSITION RESULTS")
    print("="*70)

    save_columns = [

        "sales_date",

        "product",

        "current_phase",

        "predicted_phase",

        "transition_confidence",

        "transition_entropy",

        "transition_uncertainty",

        "dynamic_transition_risk",

        "risk_level"

    ]

    probability_columns = [

        c

        for c in results.columns

        if c.startswith("prob_phase_")

    ]

    save_columns.extend(probability_columns)

    data = results[save_columns].copy()

    data["sales_date"] = data["sales_date"].astype(str)

    supabase.table(
        "dynamic_transition_probability"
    ).upsert(
        data.to_dict("records")
    ).execute()

    print()

    print("Saved", len(data), "rows.")
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

    # Train classifier
    model = train_transition_model(

        X_train,

        y_train

    )

    # Predict probabilities
    results = compute_transition_probability(

        model,

        X_test,

        test_df

    )
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix
    )

    print("\n" + "="*70)
    print("MODEL PERFORMANCE")
    print("="*70)

    accuracy = accuracy_score(
        test_df["target_phase"],
        results["predicted_phase"]
    )

    print()
    print("Accuracy :", round(accuracy,4))

    print()
    print("Classification Report")
    print(
        classification_report(
            test_df["target_phase"],
            results["predicted_phase"]
        )
    )

    print()
    print("Confusion Matrix")
    print(
    confusion_matrix(
        test_df["target_phase"],
        results["predicted_phase"]
    )
)

    # Compute transition risk
    results = compute_transition_risk(

        results

    )

    # Assign risk levels
    results = assign_transition_level(

        results

    )

    # Save / display
    save_transition_probability(

        results

    )
