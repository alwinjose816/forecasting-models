from load_data import load_dealer_orders

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from load_supabase import supabase

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
def save_baseline_forecasts(
    dates,
    actual,
    forecast,
    model_name,
    mae,
    rmse,
    mape,
    r2
):

    rows = []

    for d, a, f in zip(
        dates,
        actual,
        forecast
    ):

        rows.append({

            "forecast_date": str(pd.to_datetime(d).date()),

            "model_name": model_name,

            "actual_demand": float(a),

            "predicted_demand": float(f),

            "mae": float(mae),

            "rmse": float(rmse),

            "mape": float(mape),

            "r2": float(r2),

            "experiment_id": "BASELINE_V1"

        })

    supabase.table(
        "hdgen_baseline_forecasts"
    ).upsert(
        rows,
        on_conflict="forecast_date,model_name,experiment_id"
    ).execute()

    print(
        f"\nSaved {len(rows)} baseline forecasts."
    )


def run_catboost_forecast(product):

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

    daily["date"] = pd.to_datetime(daily["date"])

    # Create complete daily calendar
    calendar = pd.DataFrame({
        "date": pd.date_range(
            daily["date"].min(),
            daily["date"].max(),
            freq="D"
        )
    })

    daily = (
        calendar
        .merge(
            daily,
            on="date",
            how="left"
        )
    )

    daily["demand"] = daily["demand"].fillna(0)

    daily = daily.sort_values("date")

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

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily["day_of_week"] = (
        daily["date"].dt.dayofweek
    )

    daily["month"] = (
        daily["date"].dt.month
    )

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

    # Same evaluation period as HDGEN

    TEST_START = pd.Timestamp("2025-08-26")

    train_mask = daily["date"] < TEST_START
    test_mask = daily["date"] >= TEST_START
    print("\nDataset Rows :", len(daily))

    print(
        "Training Rows :",
        train_mask.sum()
    )

    print(
        "Testing Rows  :",
        test_mask.sum()
    )

    print(
        "Test Starts   :",
        daily.loc[test_mask, "date"].min()
    )

    print(
        "Test Ends     :",
        daily.loc[test_mask, "date"].max()
    )

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    # CatBoost Model

    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
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
            pd.to_datetime(
                daily["date"].max()
            )
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

        future_forecast.append(
            float(pred)
        )

        history.append(pred)

    future_dates = pd.date_range(
        start=pd.to_datetime(
            daily["date"].max()
        ) + pd.Timedelta(days=1),
        periods=30,
        freq="D"
    )
    save_baseline_forecasts(

        dates=daily.loc[test_mask, "date"],

        actual=y_test,

        forecast=test_forecast,

        model_name="CatBoost",

        mae=mae,

        rmse=rmse,

        mape=mape,

        r2=r2

    )

    return {

        "product": product,

        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),

        "dates":
        daily.loc[test_mask, "date"]
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

        "future_forecast":
        future_forecast
    }
if __name__ == "__main__":

    result = run_catboost_forecast("Overall")

    print("\n" + "=" * 60)
    print("CATBOOST FORECAST RESULTS")
    print("=" * 60)

    print(f"Product : {result['product']}")
    print(f"MAE     : {result['mae']}")
    print(f"RMSE    : {result['rmse']}")
    print(f"MAPE    : {result['mape']}%")
    print(f"R²      : {result['r2']}")

    print("\nTest Forecast")
    print("-" * 60)

    for date, actual, forecast in zip(
        result["dates"],
        result["actual"],
        result["forecast"]
    ):
        print(
            f"{date} | "
            f"Actual: {actual:.2f} | "
            f"Forecast: {forecast:.2f}"
        )

    print("\nFuture 7-Day Forecast")
    print("-" * 60)

    for date, forecast in zip(
        result["future_dates"],
        result["future_forecast"]
    ):
        print(
            f"{date} : {forecast:.2f}"
        )