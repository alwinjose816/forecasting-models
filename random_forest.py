from load_data import load_dealer_orders

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def run_random_forest_forecast(product):

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

    daily.columns = [
        "date",
        "demand"
    ]

    daily = daily.sort_values("date")

    # Lag Features

    daily["lag1"] = (
        daily["demand"].shift(1)
    )

    daily["lag7"] = (
        daily["demand"].shift(7)
    )

    daily["lag14"] = (
        daily["demand"].shift(14)
    )

    daily = daily.dropna()

    # Features

    X = daily[
        ["lag1", "lag7", "lag14"]
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

    # Random Forest

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # Test Forecast

    test_forecast = model.predict(
        X_test
    )

    # Store Forecasts

    daily["forecast"] = np.nan

    daily.loc[
        daily.index[split_index:],
        "forecast"
    ] = test_forecast

    # Metrics on Test Set

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

    for _ in range(30):

        lag1 = history[-1]
        lag7 = history[-7]
        lag14 = history[-14]

        pred = model.predict(
            [[lag1, lag7, lag14]]
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

    return {

        "product": product,

        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),

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

        "future_forecast":
        future_forecast
    }