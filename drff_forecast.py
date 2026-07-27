from forecast_demand_transitions import (
    run_forecast_demand_transitions
)

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def run_drff_forecast(
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

    # ==========================================
    # DRFF FORECAST
    # ==========================================

    forecasts = []

    test_df = df.iloc[
        split_index:
    ].copy()

    for _, row in test_df.iterrows():

        state_mean = (
            state_mean_map.get(
                row["state"],
                row["rolling_mean_30"]
            )
        )

        transition_adjustment = (

            row["state_stability"]

            *

            row["expected_duration"]
        )

        forecast = (

            0.40 * row["memory_state"]

            +

            0.20 * row["memory_residual"]

            +

            0.20 * state_mean

            +

            0.20 * transition_adjustment

        )

        forecasts.append(
            forecast
        )

    y_test = test_df[
        "demand"
    ].values

    forecasts = np.array(
        forecasts
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

        pred = (

            0.40 * memory_state

            +

            0.20 * memory_residual

            +

            0.20 * state_mean

            +

            0.20 * transition_adjustment

        )

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

    result = run_drff_forecast(
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