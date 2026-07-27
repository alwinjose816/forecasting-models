from load_data import load_dealer_orders
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import pandas as pd

def run_naive_forecast(product):

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

    daily.columns = ["date", "demand"]

    split_index = int(len(daily) * 0.8)

    train = daily[:split_index]
    test = daily[split_index:].copy()

    last_train_value = train["demand"].iloc[-1]

    forecast = []

    previous = last_train_value

    for actual in test["demand"]:
        forecast.append(previous)
        previous = actual

    test["forecast"] = forecast
    last_demand = float(
        daily["demand"].iloc[-1]
    )

    future_dates = pd.date_range(
        start=pd.to_datetime(daily["date"].max()) + pd.Timedelta(days=1),
        periods=30,
        freq="D"
    )

    future_forecast = [last_demand] * 30

    mae = mean_absolute_error(
        test["demand"],
        test["forecast"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["demand"],
            test["forecast"]
        )
    )
    r2 = r2_score(
        test["demand"],
        test["forecast"]
    )

    test_nonzero = test[
    test["demand"] != 0
    ]

    mape = (
        np.abs(
            (
                test_nonzero["demand"]
                - test_nonzero["forecast"]
            )
            / test_nonzero["demand"]
        )
    ).mean() * 100

    return {
        "product": product,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),

        "dates": test["date"].astype(str).tolist(),
        "actual": test["demand"].tolist(),
        "forecast": test["forecast"].tolist(),

        "future_dates": future_dates.astype(str).tolist(),
        "future_forecast": future_forecast
    }