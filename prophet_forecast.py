from load_data import load_dealer_orders

import pandas as pd
import numpy as np

from prophet import Prophet

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def run_prophet_forecast(product):

    df = load_dealer_orders()

    if product == "Overall":

        product_df = df.copy()

    else:

        product_df = df[
            df["product_code"] == product
        ].copy()

    # Daily Demand

    daily = (
        product_df.groupby(
            product_df["order_date"].dt.date
        )["total_weight_mt"]
        .sum()
        .reset_index()
    )

    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])

    daily = daily.sort_values("ds")

    # Train-Test Split

    split_index = int(len(daily) * 0.8)

    train = daily[:split_index]
    test = daily[split_index:]

    # Prophet Model

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.add_country_holidays(
        country_name='IN'
    )

    model.fit(train)

    # Forecast Test Period

    future = model.make_future_dataframe(
        periods=len(test)
    )

    forecast = model.predict(future)

    # Take only the forecast period
    test_forecast = forecast.tail(len(test)).reset_index(drop=True)

    test = test.reset_index(drop=True)

    test["forecast"] = np.maximum(
        test_forecast["yhat"].values,
        0
    )
    print("Test length:", len(test))
    print("Forecast length:", len(test_forecast))

    print("First test date:", test["ds"].iloc[0])
    print("First forecast date:", test_forecast["ds"].iloc[0])

    print("Last test date:", test["ds"].iloc[-1])
    print("Last forecast date:", test_forecast["ds"].iloc[-1])

    # Metrics

    mae = mean_absolute_error(
        test["y"],
        test["forecast"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["y"],
            test["forecast"]
        )
    )
    r2 = r2_score(
        test["y"],
        test_forecast["yhat"]
    )

    test_nonzero = test[
        test["y"] != 0
    ]

    mape = (
        np.abs(
            (
                test_nonzero["y"]
                - test_nonzero["forecast"]
            )
            / test_nonzero["y"]
        )
    ).mean() * 100
    # Refit on Full Data for Future Forecast

    final_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    final_model.add_country_holidays(
        country_name="IN"
    )

    final_model.fit(daily)

    # Future 7 Days Forecast

    future_30 = final_model.make_future_dataframe(
        periods=30
    )

    future_forecast = final_model.predict(
        future_30
    )

    next30 = future_forecast.tail(30).copy()

    next30["yhat"] = np.maximum(
        next30["yhat"],
        0
    )

    return {
        "product": product,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),

        "dates":
        test["ds"].astype(str).tolist(),

        "actual":
        test["y"].tolist(),

        "forecast":
        test["forecast"].tolist(),

        "future_dates":
        next30["ds"].astype(str).tolist(),

        "future_forecast":
        next30["yhat"].tolist()
    }