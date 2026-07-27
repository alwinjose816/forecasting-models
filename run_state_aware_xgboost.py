from load_data import load_dealer_orders
from forecast_demand_states import (
    run_forecast_demand_states
)

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def run_state_aware_xgboost(product):

    # ==========================================
    # BASE DEMAND DATA
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
    # LOAD LATENT STATES
    # ==========================================

    state_result = run_forecast_demand_states(
        product
    )

    if "error" in state_result:
        return state_result

    states_df = pd.concat(

        [
            pd.DataFrame(
                state_result["train_states"]
            ),

            pd.DataFrame(
                state_result["test_states"]
            )

        ],

        ignore_index=True

    )

    states_df["date"] = pd.to_datetime(
        states_df["date"]
    )

    state_cols = [
        c
        for c in states_df.columns
        if c.startswith("z")
    ]

    merge_cols = (
        ["date"]
        + state_cols
    )

    daily = daily.merge(
        states_df[merge_cols],
        on="date",
        how="left"
    )

    daily = daily.dropna()
    

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

    split_index = (
        state_result["split_index"]
    )
    print("Daily rows:", len(daily))
    print("Split index:", split_index)

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
    # ==========================================
    # FUTURE 30 DAY FORECAST
    # ==========================================

    future_dates = pd.date_range(
        start=daily["date"].max() + pd.Timedelta(days=1),
        periods=30,
        freq="D"
    )

    future_forecast = []

    history = daily["demand"].tolist()

    future_row = X.iloc[-1:].copy()
    for z in state_cols:
        future_row[z] = X.iloc[-1][z]

    for i in range(30):

        future_date = future_dates[i]

        pred = model.predict(
            future_row
        )[0]

        future_forecast.append(
            float(pred)
        )

        history.append(pred)

        future_row["lag1"] = history[-1]

        if len(history) >= 7:
            future_row["lag7"] = history[-7]

        if len(history) >= 14:
            future_row["lag14"] = history[-14]

        if len(history) >= 30:
            future_row["lag30"] = history[-30]

        future_row["rolling_mean_7"] = np.mean(
            history[-7:]
        )

        future_row["rolling_mean_30"] = np.mean(
            history[-30:]
        )       

        future_row["day_of_week"] = (
            future_date.dayofweek
        )

        future_row["month"] = (
            future_date.month
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

        "product": str(product),

        "mae": float(round(mae, 2)),
        "rmse": float(round(rmse, 2)),
        "mape": float(round(mape, 2)),

        "feature_importance":
        {
            k: float(v)
            for k, v in importance.items()
        },

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
        [
            str(d.date())
            for d in future_dates
        ],

        "future_forecast":
        future_forecast

    }


if __name__ == "__main__":

    result = run_state_aware_xgboost(
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