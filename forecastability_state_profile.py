from forecast_demand_memory import (
    run_forecast_demand_memory
)

from xgboost_forecast_validation import (
    run_xgboost_forecast_validation
)

import pandas as pd
import numpy as np

from scipy.stats import kruskal


def run_forecastability_state_profile(
    product="Overall"
):

    # =====================================
    # FORECASTABILITY DATA
    # =====================================

    fi_result = run_forecast_demand_memory(
        product
    )

    fi_df = pd.DataFrame(
        fi_result["features"]
    )

    fi_df["date"] = pd.to_datetime(
        fi_df["date"]
    )

    # =====================================
    # FORECAST ERRORS
    # =====================================

    forecast_result = (
        run_xgboost_forecast_validation(
            product
        )
    )

    error_df = pd.DataFrame({

        "date":
        pd.to_datetime(
            forecast_result["test_dates"]
        ),

        "forecast_error":
        forecast_result[
            "prediction_error"
        ]

    })

    # =====================================
    # MERGE
    # =====================================

    merged = fi_df.merge(

        error_df,

        on="date",

        how="left"

    )

    # =====================================
    # STATE PROFILE
    # =====================================

    profile = (

        merged.groupby(
            "forecastability_state"
        )

        .agg({

            "cv_30": "mean",

            "entropy_30": "mean",

            "shock_rate_30": "mean",

            "rolling_memory_strength":
            "mean",

            "seasonality_strength":
            "mean",

            "forecastability_index":
            "mean",

            "forecast_error":
            "mean"

        })

        .round(4)

    )

    print("\n")
    print("=" * 70)
    print(
        "FORECASTABILITY STATE PROFILE"
    )
    print("=" * 70)

    print(profile)

    # =====================================
    # KRUSKAL TEST
    # =====================================

    H = merged[
        merged[
            "forecastability_state"
        ] == "H"
    ][
        "forecast_error"
    ].dropna()

    M = merged[
        merged[
            "forecastability_state"
        ] == "M"
    ][
        "forecast_error"
    ].dropna()

    L = merged[
        merged[
            "forecastability_state"
        ] == "L"
    ][
        "forecast_error"
    ].dropna()

    stat, p = kruskal(
        H,
        M,
        L
    )

    print("\n")
    print("=" * 70)
    print(
        "KRUSKAL-WALLIS TEST"
    )
    print("=" * 70)

    print(
        "Statistic:",
        round(stat, 4)
    )

    print(
        "P-value:",
        p
    )

    # =====================================
    # STATE COUNTS
    # =====================================

    print("\n")
    print("=" * 70)
    print(
        "STATE COUNTS"
    )
    print("=" * 70)

    print(

        merged[
            "forecastability_state"
        ]

        .value_counts()

    )

    return {

        "profile":
        profile,

        "merged":
        merged,

        "kruskal_stat":
        stat,

        "kruskal_p":
        p

    }


if __name__ == "__main__":

    run_forecastability_state_profile(
        "Overall"
    )