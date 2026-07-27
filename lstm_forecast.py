from load_data import load_dealer_orders

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense
)

def run_lstm_forecast(product):

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

    values = (
        daily["demand"]
        .values
        .reshape(-1, 1)
    )

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        values
    )

    # Sequence Length

    LOOKBACK = 30

    X = []
    y = []

    for i in range(
        LOOKBACK,
        len(scaled)
    ):

        X.append(
            scaled[
                i-LOOKBACK:i
            ]
        )

        y.append(
            scaled[i]
        )

    X = np.array(X)
    y = np.array(y)

    # 80-20 Split

    split_index = int(
        len(X) * 0.8
    )

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    # LSTM Model

    model = Sequential()

    model.add(
        LSTM(
            50,
            input_shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        )
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        verbose=0
    )

    # Test Forecast

    test_forecast = model.predict(
        X_test,
        verbose=0
    )

    test_forecast = (
        scaler.inverse_transform(
            test_forecast
        )
        .flatten()
    )

    y_actual = (
        scaler.inverse_transform(
            y_test
        )
        .flatten()
    )

    test_forecast = np.maximum(
        test_forecast,
        0
    )

    # Metrics

    mae = mean_absolute_error(
        y_actual,
        test_forecast
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            test_forecast
        )
    )
    r2 = r2_score(
        y_test,
        test_forecast
    )

    nonzero_mask = (
        y_actual != 0
    )

    mape = (
        np.abs(
            (
                y_actual[
                    nonzero_mask
                ]
                -
                test_forecast[
                    nonzero_mask
                ]
            )
            /
            y_actual[
                nonzero_mask
            ]
        )
    ).mean() * 100

    # Future 7 Days

    history = scaled[
        -LOOKBACK:
    ].copy()

    future_forecast = []

    for _ in range(30):

        pred = model.predict(
            history.reshape(
                1,
                LOOKBACK,
                1
            ),
            verbose=0
        )

        value = scaler.inverse_transform(
            pred
        )[0][0]

        value = max(
            0,
            value
        )

        future_forecast.append(
            float(value)
        )

        history = np.vstack(
            [history[1:], pred]
        )

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
        daily.iloc[
            LOOKBACK + split_index:
        ]["date"]
        .astype(str)
        .tolist(),

        "actual":
        y_actual.tolist(),

        "forecast":
        test_forecast.tolist(),

        "future_dates":
        future_dates
        .astype(str)
        .tolist(),

        "future_forecast":
        future_forecast
    }