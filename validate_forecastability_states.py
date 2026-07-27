from forecast_demand_memory import (
    run_forecast_demand_memory
)

from xgboost_forecast_validation import (
    run_xgboost_forecast_validation
)

import pandas as pd
import numpy as np


fi_result = run_forecast_demand_memory(
    "Overall"
)

xgb_result = run_xgboost_forecast_validation(
    "Overall"
)

fi_df = pd.DataFrame(
    fi_result["features"]
)

fi_df["date"] = pd.to_datetime(
    fi_df["date"]
)

xgb_df = pd.DataFrame({

    "date": pd.to_datetime(
        xgb_result["test_dates"]
    ),

    "actual":
    xgb_result["actual"],

    "forecast":
    xgb_result["forecast"],

    "error":
    xgb_result["prediction_error"]
})

merged = xgb_df.merge(

    fi_df[
        [
            "date",
            "forecastability_state",
            "forecastability_index"
        ]
    ],

    on="date",
    how="left"
)

print(
    merged.groupby(
        "forecastability_state"
    )["error"]
    .agg(
        ["count", "mean", "std"]
    )
)
print("\nAverage Demand By State")

print(
    merged.groupby(
        "forecastability_state"
    )["actual"]
    .mean()
)
merged["ape"] = (

    np.abs(
        merged["actual"]
        -
        merged["forecast"]
    )

    /

    (
        merged["actual"]
        + 1e-6
    )

) * 100
print("\nMAPE By State")

print(
    merged.groupby(
        "forecastability_state"
    )["ape"]
    .mean()
)
from scipy.stats import kruskal

error_H = merged[
    merged["forecastability_state"] == "H"
]["ape"]

error_M = merged[
    merged["forecastability_state"] == "M"
]["ape"]

error_L = merged[
    merged["forecastability_state"] == "L"
]["ape"]

stat, p = kruskal(
    error_H,
    error_M,
    error_L
)

print("\nKruskal-Wallis Test")
print("Statistic:", stat)
print("P-value:", p)
print("\nTransition Matrix")

transition = pd.crosstab(

    fi_df["forecastability_state"].shift(1),

    fi_df["forecastability_state"],

    normalize="index"
)

print(
    transition.round(3)
)
print("\nState Characteristics")

state_profile = (

    fi_df.groupby(
        "forecastability_state"
    )[

        [
            "cv_30",
            "entropy_30",
            "shock_rate_30",
            "forecastability_index"
        ]

    ]

    .mean()

)

print(
    state_profile.round(3)
)