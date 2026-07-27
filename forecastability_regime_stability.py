from forecast_demand_memory import (
    run_forecast_demand_memory
)

import pandas as pd
import numpy as np


def run_forecastability_regime_stability(
    product="Overall"
):

    result = run_forecast_demand_memory(
        product
    )

    df = pd.DataFrame(
        result["features"]
    )

    states = df[
        "forecastability_state"
    ].tolist()

    # =====================================
    # REGIME DURATIONS
    # =====================================

    durations = []

    current_state = states[0]
    current_length = 1

    for state in states[1:]:

        if state == current_state:

            current_length += 1

        else:

            durations.append({

                "state":
                current_state,

                "duration":
                current_length

            })

            current_state = state
            current_length = 1

    durations.append({

        "state":
        current_state,

        "duration":
        current_length

    })

    duration_df = pd.DataFrame(
        durations
    )

    # =====================================
    # AVERAGE REGIME LENGTH
    # =====================================

    avg_duration = (

        duration_df

        .groupby("state")

        ["duration"]

        .agg(
            ["count", "mean", "max"]
        )

        .round(2)

    )

    print("\n")
    print("=" * 70)
    print(
        "AVERAGE REGIME DURATION"
    )
    print("=" * 70)

    print(
        avg_duration
    )

    # =====================================
    # TRANSITION MATRIX
    # =====================================

    transition_matrix = pd.crosstab(

        df[
            "forecastability_state"
        ].shift(1),

        df[
            "forecastability_state"
        ],

        normalize="index"

    )

    print("\n")
    print("=" * 70)
    print(
        "TRANSITION MATRIX"
    )
    print("=" * 70)

    print(
        transition_matrix.round(3)
    )

    # =====================================
    # EXPECTED STABILITY
    # =====================================

    stability = {}

    for state in ["H", "M", "L"]:

        if state in transition_matrix.index:

            stability[state] = round(

                transition_matrix.loc[
                    state,
                    state
                ],

                4

            )

    print("\n")
    print("=" * 70)
    print(
        "STATE PERSISTENCE"
    )
    print("=" * 70)

    for k, v in stability.items():

        print(
            f"{k}: {v}"
        )

    # =====================================
    # SURVIVAL TABLE
    # =====================================

    survival = (

        duration_df

        .groupby("state")

        ["duration"]

        .describe()

        .round(2)

    )

    print("\n")
    print("=" * 70)
    print(
        "REGIME SURVIVAL PROFILE"
    )
    print("=" * 70)

    print(
        survival
    )

    return {

        "duration":
        duration_df,

        "avg_duration":
        avg_duration,

        "transition_matrix":
        transition_matrix,

        "survival":
        survival

    }


if __name__ == "__main__":

    run_forecastability_regime_stability(
        "Overall"
    )