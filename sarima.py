from load_data import load_dealer_orders

import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
def run_sarima_forecast(product):

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

    daily = daily.sort_values("date")
    split_index = int(len(daily) * 0.8)

    train = daily["demand"][:split_index]
    test = daily["demand"][split_index:]

    model = SARIMAX(
        train,
        order=(1,1,1),
        seasonal_order=(1,1,1,7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(
        disp=False
    )

    forecast = fitted_model.forecast(
        steps=len(test)
    )

    mae = mean_absolute_error(
        test,
        forecast
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            forecast
        )
    )
    r2 = r2_score(
        test,
        forecast
    )

    test_nonzero = test[test != 0]

    mape = (
        np.abs(
            (
                test_nonzero
                - forecast.loc[test_nonzero.index]
            )
            / test_nonzero
        )
    ).mean() * 100

    future_forecast = (
        fitted_model.forecast(30)
    )

    future_dates = pd.date_range(
        start=pd.to_datetime(
            daily["date"].max()
        ) + pd.Timedelta(days=1),
        periods=30,
        freq="D"
    )
    test_dates = daily["date"][split_index:]

    return {
        "product": product,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),

       

        "dates": test_dates.astype(str).tolist(),
        "actual": test.tolist(),
        "forecast": forecast.tolist(),

        "future_dates": future_dates.astype(str).tolist(),
        "future_forecast": future_forecast.tolist()
    }