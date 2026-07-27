from forecast_demand_memory import (
    run_forecast_demand_memory
)

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


def run_forecastability_transition_model(
    product="Overall"
):

    # ==========================================
    # LOAD FORECASTABILITY FEATURES
    # ==========================================

    result = run_forecast_demand_memory(
        product
    )

    df = pd.DataFrame(
        result["features"]
    )

    # ==========================================
    # NEXT STATE TARGET
    # ==========================================

    df["next_state"] = (
        df["forecastability_state"]
        .shift(-1)
    )

    df = df.dropna(
        subset=["next_state"]
    )

    # ==========================================
    # FEATURES
    # ==========================================

    feature_cols = [

        "forecastability_index",

        "cv_30",

        "entropy_30",

        "shock_rate_30",

        "rolling_memory_strength",

        "seasonality_strength"

    ]

    X = df[
        feature_cols
    ]

    y = df[
        "next_state"
    ]

    # ==========================================
    # ENCODE STATES
    # ==========================================

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(
        y
    )

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    split_index = int(
        len(df) * 0.8
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y_encoded[:split_index]
    y_test = y_encoded[split_index:]

    # ==========================================
    # MODEL
    # ==========================================

    model = XGBClassifier(

        n_estimators=200,

        max_depth=3,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="multi:softmax",

        num_class=3,

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # ==========================================
    # PREDICTION
    # ==========================================

    pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        pred
    )

    print("\n")
    print("=" * 50)
    print("FORECASTABILITY TRANSITION MODEL")
    print("=" * 50)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClass Mapping")

    for i, c in enumerate(
        encoder.classes_
    ):
        print(i, "=", c)

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            pred,
            target_names=
            encoder.classes_
        )
    )

    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_test,
            pred
        )
    )

    # ==========================================
    # TRANSITION PROBABILITY MATRIX
    # ==========================================

    transition_matrix = pd.crosstab(

        df[
            "forecastability_state"
        ],

        df[
            "next_state"
        ],

        normalize="index"
    )

    print("\nTransition Matrix")

    print(
        transition_matrix.round(3)
    )

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    importance = pd.DataFrame({

        "feature":
        feature_cols,

        "importance":
        model.feature_importances_

    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print("\nFeature Importance")

    print(
        importance
    )

    return {

        "accuracy":
        float(accuracy),

        "transition_matrix":
        transition_matrix
        .reset_index()
        .to_dict(
            "records"
        ),

        "feature_importance":
        importance
        .to_dict(
            "records"
        )
    }


if __name__ == "__main__":

    run_forecastability_transition_model(
        "Overall"
    )