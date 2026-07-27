from forecast_demand_memory import (
    run_forecast_demand_memory
)

import pandas as pd
import numpy as np


def calculate_future_cv(series):

    mean_val = series.mean()

    if mean_val == 0:
        return np.nan

    return series.std() / mean_val


def build_dynamic_forecastability_surface(
    product="Overall"
):

    result = run_forecast_demand_memory(
        product
    )

    df = pd.DataFrame(
        result["features"]
    )

    df = df.sort_values(
        "date"
    )

    horizons = [
        1,
        7,
        14,
        30
    ]

    # --------------------------------
    # Future Forecastability Surface
    # --------------------------------

    for h in horizons:

        future_cv = []

        for i in range(len(df)):

            future_window = df[
                "demand"
            ].iloc[
                i + 1:
                i + h + 1
            ]

            if len(future_window) < h:

                future_cv.append(
                    np.nan
                )

                continue

            cv = calculate_future_cv(
                future_window
            )

            future_cv.append(
                cv
            )

        df[
            f"future_cv_{h}"
        ] = future_cv

        # Convert CV to forecastability

        max_cv = np.nanmax(
            future_cv
        )

        df[
            f"DFS_{h}"
        ] = (

            1

            -

            (
                df[
                    f"future_cv_{h}"
                ]

                /

                max_cv
            )

        )

    # --------------------------------
    # Correlation Analysis
    # --------------------------------

    print("\n")
    print("=" * 70)
    print(
        "DYNAMIC FORECASTABILITY SURFACE"
    )
    print("=" * 70)

    corr_results = []

    for h in horizons:

        corr = df[
            "forecastability_index"
        ].corr(

            df[
                f"DFS_{h}"
            ]

        )

        corr_results.append({

            "horizon": h,

            "correlation":
            round(
                corr,
                4
            )

        })

    corr_df = pd.DataFrame(
        corr_results
    )

    print(
        "\nForecastability Surface Correlation"
    )

    print(
        corr_df
    )

    print(
        "\nSample Surface"
    )

    print(

        df[
            [
                "date",
                "forecastability_index",
                "DFS_1",
                "DFS_7",
                "DFS_14",
                "DFS_30"
            ]
        ]

        .tail(10)

    )

    return {

        "surface": df,

        "correlations":
        corr_df

    }


if __name__ == "__main__":

    build_dynamic_forecastability_surface(
        "Overall"
    )