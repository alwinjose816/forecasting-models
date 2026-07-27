from forecast_demand_transitions import (
    run_forecast_demand_transitions
)

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from sklearn.linear_model import Ridge


def run_drff_forecastv2(
    product="Overall"
):

    # ==========================================
    # LOAD TRANSITION DATA
    # ==========================================

    result = run_forecast_demand_transitions(
        product
    )

    if "error" in result:
        return result

    df = pd.DataFrame(
        result["regimes"]
    )

    split_index = result[
        "split_index"
    ]

    # ==========================================
    # CLEAN
    # ==========================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    # ==========================================
    # TRAIN STATE PROFILES
    # ==========================================

    train_df = df.iloc[
        :split_index
    ].copy()

    state_mean_map = (

        train_df

        .groupby("state")["demand"]

        .mean()

        .to_dict()
    )
    train_df["state_mean"] = (
        train_df["state"]
        .map(state_mean_map)
    )

    train_df["transition_adjustment"] = (
        train_df["state_stability"]
        *
        train_df["expected_duration"]
    )
    test_df = df.iloc[
        split_index:
    ].copy()

    test_df["state_mean"] = (
        test_df["state"]
        .map(state_mean_map)
    )

    test_df["transition_adjustment"] = (
        test_df["state_stability"]
        *
        test_df["expected_duration"]
    )

    # ==========================================
    # DRFF FORECAST
    # ==========================================

    feature_cols = [

        "memory_state",

        "memory_residual",

        "state_mean",

        "transition_adjustment"

    ]

    X_train = train_df[
        feature_cols
    ]

    y_train = train_df[
        "demand"
    ]

    X_test = test_df[
        feature_cols
    ]

    ridge = Ridge(
        alpha=1.0
    )

    ridge.fit(
        X_train,
        y_train
    )

    forecasts = ridge.predict(
        X_test
    )

    y_test = test_df[
        "demand"
    ].values

    forecasts = np.array(
        forecasts
    )
    weights = dict(

        zip(
            feature_cols,
            ridge.coef_
        )

    )
    weights["intercept"] = float(
        ridge.intercept_
    )

    # ==========================================
    # METRICS
    # ==========================================

    mae = mean_absolute_error(
        y_test,
        forecasts
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            forecasts
        )
    )

    nonzero_mask = (
        y_test != 0
    )

    mape = (

        np.abs(

            (
                y_test[
                    nonzero_mask
                ]
                -
                forecasts[
                    nonzero_mask
                ]
            )

            /

            y_test[
                nonzero_mask
            ]

        )

    ).mean() * 100

    # ==========================================
    # FUTURE 30 DAYS
    # ==========================================

    history = (
        df["demand"]
        .tolist()
    )

    future_forecast = []

    current_state = int(
        df["state"].iloc[-1]
    )

    state_mean = state_mean_map.get(
        current_state,
        np.mean(history[-30:])
    )

    stability = result[
        "stability_map"
    ].get(
        str(current_state),
        0.5
    )

    duration = result[
        "expected_duration_map"
    ].get(
        str(current_state),
        1
    )

    for i in range(30):

        memory_state = state_mean

        memory_residual = 0

        transition_adjustment = (
            stability * duration
        )

        future_X = pd.DataFrame([{

            "memory_state":
            memory_state,

            "memory_residual":
            memory_residual,

            "state_mean":
            state_mean,

            "transition_adjustment":
            transition_adjustment

        }])

        pred = ridge.predict(
            future_X
        )[0]

        future_forecast.append(
            float(pred)
        )

        history.append(pred)

    future_dates = pd.date_range(

        start=(
            pd.to_datetime(
                df["date"].max()
            )
            +
            pd.Timedelta(days=1)
        ),

        periods=30,

        freq="D"
    )

    return {

        "model":
        "DRFF",
        "drff_weights":
        weights,

        "mae":
        round(mae, 2),

        "rmse":
        round(rmse, 2),

        "mape":
        round(mape, 2),

        "dates":
        test_df["date"]
        .astype(str)
        .tolist(),

        "actual":
        y_test.tolist(),

        "forecast":
        forecasts.tolist(),

        "future_dates":
        future_dates
        .astype(str)
        .tolist(),

        "future_forecast":
        future_forecast
    }


if __name__ == "__main__":

    result = run_drff_forecastv2(
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
    print(
    "\nDRFF Weights"
    )

    for k, v in result[
        "drff_weights"
    ].items():

        print(
            f"{k}: {v:.4f}"
        )