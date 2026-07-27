from forecast_deep_demand_transitions import (
    run_forecast_deep_demand_transitions
)

import pandas as pd
import numpy as np


def validate_drfi_features(product="Overall"):

    result = run_forecast_deep_demand_transitions(
        product
    )

    df = pd.DataFrame(
        result["regimes"]
    )

    print(
        "\nAvailable Columns:\n"
    )

    print(df.columns.tolist())

    # ==========================
    # Forecast Error Proxy
    # ==========================

    df["forecast_error"] = np.abs(

        df["demand"]

        -

        df["rolling_mean_30"]

    )

    features = [

        "transition_strength",

        "transition_entropy",

        "shock_score",

        "z_score",

        "cv_30",

        "cv_90",

        "entropy_30",

        "entropy_90",

        "state_stability",

        "expected_duration"

    ]

    rows = []

    print(
        "\nFeature Correlations\n"
    )

    for col in features:

        if col not in df.columns:
            continue

        corr = abs(

            df["forecast_error"]

            .corr(

                df[col]

            )

        )

        rows.append({

            "feature": col,

            "correlation": corr

        })

        print(
            f"{col}: {corr:.4f}"
        )

    ranking = pd.DataFrame(
        rows
    )

    ranking = ranking.sort_values(

        "correlation",

        ascending=False

    )

    print(
        "\n=========================="
    )

    print(
        "FEATURE IMPORTANCE RANKING"
    )

    print(
        "=========================="
    )

    print(ranking)

    return ranking


if __name__ == "__main__":

    ranking = validate_drfi_features(
        "Overall"
    )

    print(
        "\nTop DRFI Features"
    )

    print(
        ranking.head(5)
    )