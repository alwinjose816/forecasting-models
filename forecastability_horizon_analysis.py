from forecast_demand_memory import (
    run_forecast_demand_memory
)

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


def evaluate_horizon(
    df,
    horizon
):

    temp = df.copy()

    temp["future_state"] = (
        temp["forecastability_state"]
        .shift(-horizon)
    )

    temp = temp.dropna()

    feature_cols = [

        "forecastability_index",

        "cv_30",

        "entropy_30",

        "shock_rate_30",

        "rolling_memory_strength",

        "seasonality_strength"

    ]

    X = temp[
        feature_cols
    ]

    y = temp[
        "future_state"
    ]

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        y
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

    acc = accuracy_score(
        y_test,
        pred
    )

    return acc


def run_forecastability_horizon_analysis(
    product="Overall"
):

    result = run_forecast_demand_memory(
        product
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

        acc = evaluate_horizon(
            df,
            h
        )

        results.append({

            "horizon_days":
            h,

            "accuracy":
            round(
                acc * 100,
                2
            )

        })

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 60)
    print(
        "FORECASTABILITY HORIZON ANALYSIS"
    )
    print("=" * 60)

    print(
        results_df
    )

    return results_df


if __name__ == "__main__":

    run_forecastability_horizon_analysis(
        "Overall"
    )