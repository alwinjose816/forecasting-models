from load_data import load_dealer_orders
from forecast_demand_regimes import (
    run_forecast_demand_regimes
)

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def run_regime_aware_xgboost(product):

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
    # BASE FEATURES
    # ==========================================

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

    # ==========================================
    # LOAD REGIMES
    # ==========================================

    # ==========================================
    # LOAD REGIMES
    # ==========================================

    regime_result = run_forecast_demand_regimes(
        product
    )

    if "error" in regime_result:
        return regime_result

    forecast_split = regime_result["split_index"]

    regime_df = pd.DataFrame(
        regime_result["regimes"]
    )
    regime_df["date"] = pd.to_datetime(
        regime_df["date"]
    )

    daily = daily.merge(
        regime_df[
            [
                "date",
                "state",

                "memory_state",
                "memory_residual",

                "rolling_memory_strength",

                "memory_half_life",

                "entropy_90",

                "trend_90",

                "shock_score",

                "zero_demand_ratio_90"
            ]
        ],
        on="date",
        how="left"
    )

    daily = (
        daily
        .dropna()
        .reset_index(drop=True)
    )

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
        if c.startswith("state_")
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
        "month"

    ] + state_cols

    X = daily[
        feature_cols
    ]

    y = daily["demand"]

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    split_index = min(
        forecast_split,
        len(daily)
    )
    print("Forecast Split:", forecast_split)
    print("Daily Length:", len(daily))
    print("Train Rows:", split_index)
    print("Test Rows:", len(daily) - split_index)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # ==========================================
    # MODEL
    # ==========================================

    model = XGBRegressor(

        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,

        subsample=0.8,
        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42

    )

    model.fit(
        X_train,
        y_train
    )

    test_forecast = model.predict(
        X_test
    )
    future_forecast = []

    history = daily["demand"].tolist()

    last_row = daily.iloc[-1].copy()

    for i in range(30):

        row = {}

        row["lag1"] = history[-1]
        row["lag7"] = history[-7]
        row["lag14"] = history[-14]
        row["lag30"] = history[-30]

        row["rolling_mean_7"] = np.mean(
            history[-7:]
        )

        row["rolling_mean_30"] = np.mean(
            history[-30:]
        )

        next_date = (
            daily["date"].max()
            + pd.Timedelta(days=i + 1)
        )

        row["day_of_week"] = (
            next_date.dayofweek
        )

        row["month"] = (
            next_date.month
        )

     

        for col in state_cols:
            row[col] = last_row[col]

        future_X = pd.DataFrame([row])

        future_X = future_X[
            feature_cols
        ]

        pred = float(
            model.predict(
                future_X
            )[0]
        )

        future_forecast.append(
            pred
        )

        history.append(pred)
    future_dates = pd.date_range(
        start=daily["date"].max()
            + pd.Timedelta(days=1),
        periods=30,
        freq="D"
    )

    # ==========================================
    # METRICS
    # ==========================================

    mae = mean_absolute_error(
        y_test,
        test_forecast
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

    mape = (
        np.abs(
            (
                y_test[nonzero_mask]
                - test_forecast[nonzero_mask]
            )
            /
            y_test[nonzero_mask]
        )
    ).mean() * 100

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    importance = {
        feature: float(score)
        for feature, score in zip(
            feature_cols,
            model.feature_importances_
        )
    }

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
        "Regime-Aware XGBoost",

        "product":
        product,

        "mae":
        float(round(mae, 2)),

        "rmse":
        float(round(rmse, 2)),

        "mape":
        float(round(mape, 2)),

        "feature_importance":
        importance,

        "dates":
        daily.iloc[
            split_index:
        ]["date"]
        .astype(str)
        .tolist(),

        "actual":
        [
            float(x)
            for x in y_test
        ],

        "forecast":
        [
            float(x)
            for x in test_forecast
        ],
        "future_dates":
        future_dates
        .astype(str)
        .tolist(),

        "future_forecast":
        [
            float(x)
            for x in future_forecast
        ],
    }


if __name__ == "__main__":

    result = run_regime_aware_xgboost(
        "Overall"
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