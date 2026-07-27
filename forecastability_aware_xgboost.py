from forecast_demand_memory import (
    run_forecast_demand_memory
)

from load_data import load_dealer_orders

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def run_forecastability_aware_xgboost(
    product="Overall"
):

    # ==========================================
    # LOAD DEMAND DATA
    # ==========================================

    df = load_dealer_orders()

    if product == "Overall":
        product_df = df.copy()
    else:
        product_df = df[
            df["product_code"] == product
        ].copy()

    daily = (
        product_df.groupby(
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
    # FORECASTABILITY STATES
    # ==========================================

    fi_result = run_forecast_demand_memory(
        product
    )

    fi_df = pd.DataFrame(
        fi_result["features"]
    )

    fi_df["date"] = pd.to_datetime(
        fi_df["date"]
    )

    daily = daily.merge(

        fi_df[
            [
                "date",
                "forecastability_state"
            ]
        ],

        on="date",

        how="left"
    )

    # ==========================================
    # FEATURES
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

    daily = daily.dropna()

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    split_index = int(
        len(daily) * 0.8
    )

    train_df = daily.iloc[
        :split_index
    ].copy()

    test_df = daily.iloc[
        split_index:
    ].copy()

    feature_cols = [

        "lag1",
        "lag7",
        "lag14",
        "lag30",
        "rolling_mean_7",
        "rolling_mean_30",
        "day_of_week",
        "month"

    ]

    # ==========================================
    # EXPERT DATASETS
    # ==========================================

    train_H = train_df[
        train_df[
            "forecastability_state"
        ] == "H"
    ]

    train_M = train_df[
        train_df[
            "forecastability_state"
        ] == "M"
    ]

    train_L = train_df[
        train_df[
            "forecastability_state"
        ] == "L"
    ]

    print("\nTraining Samples")

    print(
        "H:",
        len(train_H)
    )

    print(
        "M:",
        len(train_M)
    )

    print(
        "L:",
        len(train_L)
    )

    # ==========================================
    # MODEL TEMPLATE
    # ==========================================

    def build_model():

        return XGBRegressor(

            n_estimators=100,

            max_depth=3,

            learning_rate=0.1,

            subsample=0.8,

            colsample_bytree=0.8,

            objective=
            "reg:squarederror",

            random_state=42

        )

    # ==========================================
    # TRAIN EXPERTS
    # ==========================================

    model_H = build_model()

    model_M = build_model()

    model_L = build_model()

    model_H.fit(

        train_H[
            feature_cols
        ],

        train_H[
            "demand"
        ]
    )

    model_M.fit(

        train_M[
            feature_cols
        ],

        train_M[
            "demand"
        ]
    )

    model_L.fit(

        train_L[
            feature_cols
        ],

        train_L[
            "demand"
        ]
    )

    # ==========================================
    # ROUTING
    # ==========================================

    predictions = []

    for _, row in test_df.iterrows():

        X_row = row[
            feature_cols
        ].values.reshape(
            1,
            -1
        )

        state = row[
            "forecastability_state"
        ]

        if state == "H":

            pred = model_H.predict(
                X_row
            )[0]

        elif state == "M":

            pred = model_M.predict(
                X_row
            )[0]

        else:

            pred = model_L.predict(
                X_row
            )[0]

        predictions.append(
            pred
        )

    # ==========================================
    # METRICS
    # ==========================================

    actual = test_df[
        "demand"
    ].values

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(

        mean_squared_error(

            actual,

            predictions

        )

    )

    nonzero = actual != 0

    mape = (

        np.abs(

            (
                actual[nonzero]
                -
                np.array(
                    predictions
                )[nonzero]
            )

            /

            actual[nonzero]

        )

    ).mean() * 100

    print("\n")
    print("=" * 60)
    print(
        "FORECASTABILITY AWARE XGBOOST"
    )
    print("=" * 60)

    print(
        "MAE:",
        round(mae, 2)
    )

    print(
        "RMSE:",
        round(rmse, 2)
    )

    print(
        "MAPE:",
        round(mape, 2)
    )

    return {

        "mae":
        round(mae, 2),

        "rmse":
        round(rmse, 2),

        "mape":
        round(mape, 2)

    }


if __name__ == "__main__":

    run_forecastability_aware_xgboost(
        "Overall"
    )