from load_data import load_dealer_orders

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from load_supabase import supabase
import uuid

def run_xgboost_forecast(product):

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

    daily = daily.sort_values("date")

    # Lag Features

        # Lag Features

    daily["lag1"] = daily["demand"].shift(1)
    daily["lag7"] = daily["demand"].shift(7)
    daily["lag14"] = daily["demand"].shift(14)
    daily["lag30"] = daily["demand"].shift(30)

    # Rolling Features

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

    # Calendar Features

    daily["date"] = pd.to_datetime(daily["date"])

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month

    daily = daily.dropna()

    X = daily[
        [
            "lag1",
            "lag7",
            "lag14",
            "lag30",
            "rolling_mean_7",
            "rolling_mean_30",
            "day_of_week",
            "month"
        ]
    ]

    y = daily["demand"]

    # 80-20 Split

    split_index = int(
        len(daily) * 0.8
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # XGBoost Model

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
    r2 = r2_score(
        y_test,
        test_forecast
    )

    nonzero_mask = y_test != 0

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
    experiment_id = str(uuid.uuid4())
    # Retrain using all available historical data
    model.fit(X, y)

    # Future 7 Days

    history = daily[
        "demand"
    ].tolist()

    future_forecast = []

    for i in range(30):

        lag1 = history[-1]
        lag7 = history[-7]
        lag14 = history[-14]
        lag30 = history[-30]

        future_date = (
            pd.to_datetime(daily["date"].max())
            + pd.Timedelta(days=i + 1)
        )

        day_of_week = future_date.dayofweek
        month = future_date.month

        pred = model.predict(
            [[
                lag1,
                lag7,
                lag14,
                lag30,
                np.mean(history[-7:]),
                np.mean(history[-30:]),
                day_of_week,
                month
            ]]
        )[0]

        future_forecast.append(float(pred))

        history.append(pred)

    future_dates = pd.date_range(
        start=pd.to_datetime(
            daily["date"].max()
        ) + pd.Timedelta(days=1),
        periods=30,
        freq="D"
    )
    records = []

    for date, actual, pred in zip(
        daily.iloc[split_index:]["date"],
        y_test,
        test_forecast
    ):

        records.append({

            "forecast_date": str(date.date()),
            "model_name": "XGBoost",

            "actual_demand": float(actual),
            "predicted_demand": float(pred),

            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2": float(r2),

            "experiment_id": experiment_id

        })
    for date, pred in zip(
        future_dates,
        future_forecast
    ):

        records.append({

            "forecast_date": str(date.date()),
            "model_name": "XGBoost",

            "actual_demand": None,
            "predicted_demand": float(pred),

            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2": float(r2),

            "experiment_id": experiment_id

        })
    supabase.table(
        "hdgen_baseline_forecasts"
    ).upsert(
        records,
        on_conflict="forecast_date,model_name,experiment_id"
    ).execute()

    return {

        "product": product,

        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),

        "dates":
        daily.iloc[split_index:]["date"]
        .astype(str)
        .tolist(),

        "actual":
        y_test.tolist(),

        "forecast":
        test_forecast.tolist(),

        "future_dates":
        future_dates
        .astype(str)
        .tolist(),
        "r2": round(r2, 4),

        "future_forecast":
        future_forecast
    }
if __name__ == "__main__":

    result = run_xgboost_forecast("Overall")

    print("\nXGBoost Forecast Results")
    print("------------------------")
    print(f"MAE  : {result['mae']}")
    print(f"RMSE : {result['rmse']}")
    print(f"MAPE : {result['mape']}")
    print(f"R²   : {result['r2']}")

    print("\nSaved successfully to hdgen_baseline_forecasts.")