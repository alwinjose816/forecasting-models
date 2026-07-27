from load_data import load_dealer_orders

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from forecast_deep_demand_transitions import (
    run_forecast_deep_demand_transitions
)

from sklearn.preprocessing import MinMaxScaler

def run_xgboost_forecast_DRFI(product):

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
    transition_result = (
        run_forecast_deep_demand_transitions(
            product
        )
    )

    regime_df = pd.DataFrame(
        transition_result["regimes"]
    )

    regime_df["date"] = pd.to_datetime(
        regime_df["date"]
    )

    test_dates = daily.iloc[
        split_index:
    ]["date"]

    regime_test = regime_df[
        regime_df["date"].isin(
            test_dates
        )
    ].copy()

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
    forecast_error = np.abs(
        y_test.values
        -
        test_forecast
    )
    print(
        "forecast_error:",
        len(forecast_error)
    )

    print(
        "regime_test:",
        len(regime_test)
    )
    n = min(
        len(forecast_error),
        len(regime_test)
    )

    forecast_error = (
        forecast_error[:n]
    )

    regime_test = (
        regime_test.iloc[:n]
    )
    drfi_df = pd.DataFrame({

        "forecast_error":
        forecast_error,

        "memory_strength":
        regime_test[
            "rolling_memory_strength"
        ].values,

        "state_stability":
        regime_test[
            "state_stability"
        ].values,

        "expected_duration":
        regime_test[
            "expected_duration"
        ].values,

        "transition_entropy":
        regime_test[
            "transition_entropy"
        ].values
    })
    analysis_df = pd.DataFrame({

        "error": forecast_error,

        "memory_strength":
        regime_test[
            "rolling_memory_strength"
        ],

        "memory_half_life":
        regime_test[
            "memory_half_life"
        ],

        "transition_strength":
        regime_test[
            "transition_strength"
        ],

        "state":
        regime_test[
            "state"
        ],

        "z1":
        regime_test["z1"],

        "z2":
        regime_test["z2"],

        "z3":
        regime_test["z3"],

        "z4":
        regime_test["z4"],

        "z5":
        regime_test["z5"],

        "z6":
        regime_test["z6"],

        "z7":
        regime_test["z7"],

        "z8":
        regime_test["z8"]
    })
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    X_risk = analysis_df[[
        "z4",
        "z6",
        "z7",
        "z8",
        "transition_strength",
        "memory_strength"
    ]]

    y_risk = analysis_df["error"]

    risk_model = LinearRegression()

    risk_model.fit(
        X_risk,
        y_risk
    )

    risk_score = risk_model.predict(
        X_risk
    )

    r2 = r2_score(
        y_risk,
        risk_score
    )

    print("\nForecastability Model R2:", r2)

    print("\nRisk Weights")

    for f, w in zip(
        X_risk.columns,
        risk_model.coef_
    ):
        print(
            f"{f}: {w:.4f}"
        )

    print(
        analysis_df.corr()["error"]
        .sort_values()
    )
    
    drfi_df["drfi_raw"] = (

        0.30 *
        drfi_df[
            "memory_strength"
        ]

        +

        0.30 *
        drfi_df[
            "state_stability"
        ]

        +

        0.20 *
        drfi_df[
            "expected_duration"
        ]

        -

        0.20 *
        drfi_df[
            "transition_entropy"
        ]
    )

    drfi_df["drfi"] = (
        MinMaxScaler()
        .fit_transform(
            drfi_df[
                ["drfi_raw"]
            ]
        )
    )
    drfi_correlation = (

        drfi_df[
            [
                "drfi",
                "forecast_error"
            ]
        ]
        .corr()
        .iloc[0,1]

    )

    print(
        "\nDRFI Correlation:",
        drfi_correlation
    )
    drfi_df["zone"] = pd.qcut(

        drfi_df["drfi"],

        3,

        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    zone_error = (

        drfi_df
        .groupby("zone")
        [
            "forecast_error"
        ]
        .mean()
    )

    print(
        "\nZone Error:"
    )

    print(
        zone_error
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
        future_forecast,
        "drfi_correlation":
        float(drfi_correlation),

        "zone_error":
        {
            str(k): float(v)
            for k, v in zone_error.items()
        },
    }
if __name__ == "__main__":

    result = run_xgboost_forecast_DRFI(
        "Overall"
    )

    print(
        "\nMAE:",
        result["mae"]
    )

    print(
        "\nRMSE:",
        result["rmse"]
    )

    print(
        "\nMAPE:",
        result["mape"]
    )