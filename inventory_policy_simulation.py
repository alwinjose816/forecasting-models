from forecast_demand_memory import (
    run_forecast_demand_memory
)

import pandas as pd
import numpy as np


def run_inventory_policy_simulation(
    product="Overall"
):

    result = run_forecast_demand_memory(
        product
    )

    df = pd.DataFrame(
        result["features"]
    )

    # ====================================
    # DEMAND VOLATILITY
    # ====================================

    df["rolling_std_30"] = (

        df["demand"]

        .shift(1)

        .rolling(30)

        .std()

    )

    z = 1.65
    lead_time = 7

    # ====================================
    # POLICY A
    # TRADITIONAL
    # ====================================

    df["traditional_ss"] = (

        z

        *

        df["rolling_std_30"]

        *

        np.sqrt(
            lead_time
        )

    )

    # ====================================
    # POLICY B
    # FORECASTABILITY AWARE
    # ====================================

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

    # ====================================
    # NEXT DAY DEMAND
    # ====================================

    df["future_demand"] = (
        df["demand"]
        .shift(-1)
    )

    # ====================================
    # STOCKOUT CHECK
    # ====================================

    df["traditional_stockout"] = (

        df["future_demand"]

        >

        df["traditional_ss"]

    )

    df["forecastability_stockout"] = (

        df["future_demand"]

        >

        df["forecastability_ss"]

    )

    # ====================================
    # DROP NAN
    # ====================================

    df = df.dropna()

    # ====================================
    # RESULTS
    # ====================================

    traditional_stockouts = (

        df[
            "traditional_stockout"
        ]

        .sum()

    )

    fi_stockouts = (

        df[
            "forecastability_stockout"
        ]

        .sum()

    )

    traditional_inventory = (

        df[
            "traditional_ss"
        ]

        .mean()

    )

    fi_inventory = (

        df[
            "forecastability_ss"
        ]

        .mean()

    )

    service_level_traditional = (

        1

        -

        traditional_stockouts
        /
        len(df)

    ) * 100

    service_level_fi = (

        1

        -

        fi_stockouts
        /
        len(df)

    ) * 100

    # ====================================
    # PRINT
    # ====================================

    print("\n")
    print("=" * 70)
    print(
        "INVENTORY POLICY SIMULATION"
    )
    print("=" * 70)

    comparison = pd.DataFrame({

        "Policy": [

            "Traditional",

            "Forecastability"

        ],

        "Avg Inventory": [

            round(
                traditional_inventory,
                2
            ),

            round(
                fi_inventory,
                2
            )

        ],

        "Stockouts": [

            int(
                traditional_stockouts
            ),

            int(
                fi_stockouts
            )

        ],

        "Service Level %": [

            round(
                service_level_traditional,
                2
            ),

            round(
                service_level_fi,
                2
            )

        ]

    })

    print(
        comparison
    )

    return comparison


if __name__ == "__main__":

    run_inventory_policy_simulation(
        "Overall"
    )