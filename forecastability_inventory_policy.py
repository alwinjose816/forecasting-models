from forecast_demand_memory import (
    run_forecast_demand_memory
)

import pandas as pd
import numpy as np


def run_forecastability_inventory_policy(
    product="Overall"
):

    result = run_forecast_demand_memory(
        product
    )

    df = pd.DataFrame(
        result["features"]
    )

    # =====================================
    # DEMAND UNCERTAINTY
    # =====================================

    df["rolling_std_30"] = (
        df["demand"]
        .shift(1)
        .rolling(30)
        .std()
    )

    # =====================================
    # TRADITIONAL SAFETY STOCK
    # =====================================

    z = 1.65
    lead_time = 7

    df["traditional_ss"] = (

        z

        *

        df["rolling_std_30"]

        *

        np.sqrt(
            lead_time
        )

    )

    # =====================================
    # FORECASTABILITY AWARE
    # =====================================

    df["forecastability_ss"] = (

        df["traditional_ss"]

        *

        (
            1
            +
            (
                1
                -
                df[
                    "forecastability_index"
                ]
            )
        )

    )

    # =====================================
    # REGIME ANALYSIS
    # =====================================

    policy = (

        df.groupby(
            "forecastability_state"
        )

        .agg({

            "forecastability_index":
            "mean",

            "rolling_std_30":
            "mean",

            "traditional_ss":
            "mean",

            "forecastability_ss":
            "mean"

        })

        .round(2)

    )

    policy[
        "additional_buffer"
    ] = (

        policy[
            "forecastability_ss"
        ]

        -

        policy[
            "traditional_ss"
        ]

    ).round(2)

    print("\n")
    print("=" * 70)
    print(
        "FORECASTABILITY INVENTORY POLICY"
    )
    print("=" * 70)

    print(policy)

    print("\n")
    print("=" * 70)
    print(
        "BUFFER INCREASE BY STATE"
    )
    print("=" * 70)

    print(

        policy[
            [
                "additional_buffer"
            ]
        ]

    )

    return {

        "policy":
        policy,

        "data":
        df

    }


if __name__ == "__main__":

    run_forecastability_inventory_policy(
        "Overall"
    )