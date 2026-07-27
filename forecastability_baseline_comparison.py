from forecast_demand_memory import (
    run_forecast_demand_memory
)

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


def evaluate_horizon(df, horizon):

    temp = df.copy()

    temp["future_state"] = (
        temp["forecastability_state"]
        .shift(-horizon)
    )

    temp = temp.dropna()

    # --------------------------
    # Persistence Baseline
    # --------------------------

    baseline_pred = (
        temp["forecastability_state"]
    )

    baseline_actual = (
        temp["future_state"]
    )

    baseline_acc = accuracy_score(
        baseline_actual,
        baseline_pred
    )

    # --------------------------
    # XGBoost Model
    # --------------------------

    feature_cols = [

        "forecastability_index",
        "cv_30",
        "entropy_30",
        "shock_rate_30",
        "rolling_memory_strength",
        "seasonality_strength"

    ]

    X = temp[feature_cols]

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        temp["future_state"]
    )

    split_index = int(
        len(temp) * 0.8
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    model = XGBClassifier(

        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42

    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    xgb_acc = accuracy_score(
        y_test,
        pred
    )

    return {

        "horizon": horizon,

        "baseline_accuracy":
        round(
            baseline_acc * 100,
            2
        ),

        "xgb_accuracy":
        round(
            xgb_acc * 100,
            2
        )

    }


def run_forecastability_baseline_comparison():

    result = run_forecast_demand_memory(
        "Overall"
    )

    df = pd.DataFrame(
        result["features"]
    )

    horizons = [
        1,
        7,
        14,
        30
    ]

    results = []

    for h in horizons:

        results.append(
            evaluate_horizon(
                df,
                h
            )
        )

    results_df = pd.DataFrame(
        results
    )

    results_df[
        "improvement"
    ] = (

        results_df[
            "xgb_accuracy"
        ]

        -

        results_df[
            "baseline_accuracy"
        ]

    )

    print("\n")
    print("=" * 70)
    print(
        "FORECASTABILITY BASELINE COMPARISON"
    )
    print("=" * 70)

    print(
        results_df
    )

    return results_df


if __name__ == "__main__":

    run_forecastability_baseline_comparison()