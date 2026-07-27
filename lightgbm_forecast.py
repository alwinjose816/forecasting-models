from load_data import load_dealer_orders

import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
def run_lightgbm_forecast(product):

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
    daily["rolling_std_7"] = (
        daily["demand"]
        .shift(1)
        .rolling(7)
        .std()
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
            "rolling_std_7",
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

    # LightGBM Model

    model = LGBMRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    test_forecast = model.predict(
        X_test
    )
    test_forecast = np.maximum(
        test_forecast,
        0
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
            pd.to_datetime(daily["date"].max())
            + pd.Timedelta(days=i + 1)
        )

        day_of_week = future_date.dayofweek
        month = future_date.month

        future_row = pd.DataFrame({
            "lag1": [lag1],
            "lag7": [lag7],
            "lag14": [lag14],
            "lag30": [lag30],

            "rolling_mean_7": [
                np.mean(history[-7:])
            ],

            "rolling_std_7": [
                np.std(history[-7:])
            ],

            "rolling_mean_30": [
                np.mean(history[-30:])
            ],

            "day_of_week": [
                day_of_week
            ],

            "month": [
                month
            ]
        })

        pred = max(
            0,
            model.predict(future_row)[0]
        )

        future_forecast.append(float(pred))

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