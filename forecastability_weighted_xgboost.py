from load_data import load_dealer_orders
from forecast_demand_memory import (
    run_forecast_demand_memory
)

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def run_forecastability_weighted_xgboost(
    product="Overall"
):

    # ====================================
    # LOAD DATA
    # ====================================

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

    # ====================================
    # FORECASTABILITY STATES
    # ====================================

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
                "forecastability_state",
                "forecastability_index"
            ]
        ],

        on="date",
        how="left"
    )

    # ====================================
    # FEATURES
    # ====================================

    daily["lag1"] = daily["demand"].shift(1)
    daily["lag7"] = daily["demand"].shift(7)
    daily["lag14"] = daily["demand"].shift(14)
    daily["lag30"] = daily["demand"].shift(30)

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

    # ====================================
    # SPLIT
    # ====================================

    split_index = int(
        len(daily) * 0.8
    )

    train_df = daily.iloc[
        :split_index
    ].copy()

    test_df = daily.iloc[
        split_index:
    ].copy()

    # ====================================
    # SAMPLE WEIGHTS
    # ====================================

    weight_map = {

        "H": 10,

        "M": 2,

        "L": 1

    }

    train_df["sample_weight"] = (
        train_df["forecastability_state"]
        .map(weight_map)
    )

    print("\nTraining State Counts")

    print(
        train_df[
            "forecastability_state"
        ].value_counts()
    )

    print("\nWeight Distribution")

    print(
        train_df[
            "sample_weight"
        ].value_counts()
    )

    # ====================================
    # FEATURES
    # ====================================

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

    X_train = train_df[
        feature_cols
    ]

    X_test = test_df[
        feature_cols
    ]

    y_train = train_df[
        "demand"
    ]

    y_test = test_df[
        "demand"
    ]

    weights = train_df[
        "sample_weight"
    ]

    # ====================================
    # MODEL
    # ====================================

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

        y_train,

        sample_weight=weights

    )

    forecast = model.predict(
        X_test
    )

    # ====================================
    # METRICS
    # ====================================

    mae = mean_absolute_error(
        y_test,
        forecast
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            forecast
        )
    )

    nonzero = y_test != 0

    mape = (

        np.abs(

            (
                y_test[nonzero]
                -
                forecast[nonzero]
            )

            /

            y_test[nonzero]

        )

    ).mean() * 100

    print("\n")
    print("=" * 60)
    print(
        "FORECASTABILITY WEIGHTED XGBOOST"
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

    run_forecastability_weighted_xgboost(
        "Overall"
    )