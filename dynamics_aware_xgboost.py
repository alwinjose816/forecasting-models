from load_data import load_dealer_orders

from state_transitions import (
    run_state_transitions
)

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def run_dynamics_aware_xgboost(product):

    # ==========================================
    # LOAD DEMAND
    # ==========================================

    df = load_dealer_orders()

    if product == "Overall":
        product_df = df.copy()
    else:
        product_df = df[
            df["product_code"] == product
        ].copy()

    daily = (
        product_df
        .groupby(
            product_df["order_date"].dt.date
        )["total_weight_mt"]
        .sum()
        .reset_index()
    )

    daily.columns = [
        "date",
        "demand"
    ]

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily = daily.sort_values(
        "date"
    )

    # ==========================================
    # BASELINE FEATURES
    # ==========================================

    daily["lag1"] = (
        daily["demand"]
        .shift(1)
    )

    daily["lag7"] = (
        daily["demand"]
        .shift(7)
    )

    daily["lag14"] = (
        daily["demand"]
        .shift(14)
    )

    daily["lag30"] = (
        daily["demand"]
        .shift(30)
    )

    daily["rolling_mean_7"] = (
        daily["demand"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    daily["rolling_mean_30"] = (
        daily["demand"]
        .shift(1)
        .rolling(30)
        .mean()
    )

    daily["day_of_week"] = (
        daily["date"]
        .dt.dayofweek
    )

    daily["month"] = (
        daily["date"]
        .dt.month
    )

    # ==========================================
    # LOAD STATE TRANSITIONS
    # ==========================================

    transition_result = (
        run_state_transitions(
            product
        )
    )

    if "error" in transition_result:
        return transition_result

    # ==========================================
    # LOAD REGIME DATA
    # ==========================================

    if "regimes" not in transition_result:
        return {
            "error":
            "No regime data found"
        }

    regime_df = pd.DataFrame(
        transition_result[
            "regimes"
        ]
    )

    regime_df["date"] = pd.to_datetime(
        regime_df["date"]
    )

    # ==========================================
    # STABILITY FEATURES
    # ==========================================

    stability_map = (
        transition_result[
            "stability"
        ]
    )

    duration_map = (
        transition_result[
            "expected_duration"
        ]
    )

    regime_df[
        "state_stability"
    ] = regime_df["state"].map(
        stability_map
    )

    regime_df[
        "expected_duration"
    ] = regime_df["state"].map(
        duration_map
    )

    # ==========================================
    # MERGE
    # ==========================================

    daily = daily.merge(

        regime_df[
            [
                "date",
                "state",
                "state_stability",
                "expected_duration"
            ]
        ],

        on="date",
        how="left"
    )

    daily = daily.dropna()

    # ==========================================
    # ONE HOT ENCODE STATE
    # ==========================================

    daily = pd.get_dummies(
        daily,
        columns=["state"],
        prefix="state"
    )

    state_cols = [

        c
        for c in daily.columns
        if c.startswith(
            "state_"
        )
    ]

    # ==========================================
    # FEATURES
    # ==========================================

    feature_cols = [

        "lag1",
        "lag7",
        "lag14",
        "lag30",

        "rolling_mean_7",
        "rolling_mean_30",

        "day_of_week",
        "month",

        "state_stability",
        "expected_duration"

    ] + state_cols

    X = daily[
        feature_cols
    ]

    y = daily["demand"]

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    split_index = int(
        len(daily) * 0.8
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    # ==========================================
    # XGBOOST
    # ==========================================

    model = XGBRegressor(

        n_estimators=100,
        max_depth=3,

        learning_rate=0.1,

        subsample=0.8,

        colsample_bytree=0.8,

        objective=
        "reg:squarederror",

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    test_forecast = (
        model.predict(
            X_test
        )
    )

    # ==========================================
    # METRICS
    # ==========================================

    mae = (
        mean_absolute_error(
            y_test,
            test_forecast
        )
    )

    rmse = np.sqrt(

        mean_squared_error(
            y_test,
            test_forecast
        )

    )

    nonzero_mask = (
        y_test != 0
    )

    if nonzero_mask.sum() > 0:

        mape = (

            np.abs(

                (
                    y_test[
                        nonzero_mask
                    ]
                    -
                    test_forecast[
                        nonzero_mask
                    ]
                )

                /

                y_test[
                    nonzero_mask
                ]

            )

        ).mean() * 100

    else:

        mape = np.nan

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    importance = dict(

        zip(

            feature_cols,

            model
            .feature_importances_

        )

    )

    importance = dict(

        sorted(

            importance.items(),

            key=lambda x: x[1],

            reverse=True

        )

    )

    # ==========================================
    # RETURN
    # ==========================================

    return {

        "model":
        "Dynamics-Aware XGBoost",

        "product":
        product,

        "mae":
        round(mae, 2),

        "rmse":
        round(rmse, 2),

        "mape":
        round(mape, 2),

        "feature_importance":
        importance,

        "dates":

        daily.iloc[
            split_index:
        ]["date"]

        .astype(str)

        .tolist(),

        "actual":
        y_test.tolist(),

        "forecast":
        test_forecast.tolist()
    }


if __name__ == "__main__":

    result = (
        run_dynamics_aware_xgboost(
            "Overall"
        )
    )

    if "error" in result:

        print(
            result["error"]
        )

    else:

        print(
            "\nModel:",
            result["model"]
        )

        print(
            "\nMAE:",
            result["mae"]
        )

        print(
            "RMSE:",
            result["rmse"]
        )

        print(
            "MAPE:",
            result["mape"]
        )

        print(
            "\nTop Features:"
        )

        for k, v in list(
            result[
                "feature_importance"
            ].items()
        )[:10]:

            print(
                k,
                round(v, 4)
            )