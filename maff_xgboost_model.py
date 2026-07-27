from load_data import load_dealer_orders

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

def run_maff_xgboost(product):

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

 
    daily["lag7"] = daily["demand"].shift(7)
  

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
    from demand_memory import run_demand_memory

    memory = run_demand_memory(product)

    ML50 = memory["ml50"]
    EMH = memory["emh"]

    daily["memory_core"] = (
        daily["demand"]
        .shift(1)
        .rolling(ML50)
        .mean()
    )

    daily["memory_extended"] = (
        daily["demand"]
        .shift(1)
        .rolling(EMH)
        .mean()
    )
    acf_values = np.abs(
        np.array(
            memory["acf"][1:EMH+1]
        )
    )

    if acf_values.sum() == 0:

        weights = np.ones(
            len(acf_values)
        ) / len(acf_values)

    else:

        weights = (
            acf_values
            /
            acf_values.sum()
        )

    memory_weighted = 0

    for lag in range(1, EMH+1):

        memory_weighted += (

            daily["demand"]
            .shift(lag)

            *

            weights[lag-1]

        )

    daily["memory_weighted"] = (
        memory_weighted
    )

    # Calendar Features

    daily["date"] = pd.to_datetime(daily["date"])

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month

    daily = daily.dropna()

    X = daily[
        [
            "lag7",

            "rolling_mean_7",
            "rolling_mean_30",

            "memory_core",
            "memory_extended",
            "memory_weighted",

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

    for i in range(7):

        lag7 = history[-7]

        future_date = (
            pd.to_datetime(daily["date"].max())
            + pd.Timedelta(days=i + 1)
        )

        day_of_week = future_date.dayofweek
        month = future_date.month

        rolling_mean_7 = np.mean(
            history[-7:]
        )

        rolling_mean_30 = np.mean(
            history[-30:]
        )

        memory_core = np.mean(
            history[-ML50:]
        )

        memory_extended = np.mean(
            history[-EMH:]
        )

        memory_weighted = np.average(
            history[-EMH:],
            weights=weights
        )

        pred = model.predict([
            [
                lag7,

                rolling_mean_7,
                rolling_mean_30,

                memory_core,
                memory_extended,
                memory_weighted,

                day_of_week,
                month
            ]
        ])[0]
        future_forecast.append(float(pred))

        history.append(pred)

    future_dates = pd.date_range(
        start=pd.to_datetime(
            daily["date"].max()
        ) + pd.Timedelta(days=1),
        periods=7,
        freq="D"
    )

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

        "future_forecast":
        future_forecast
    }