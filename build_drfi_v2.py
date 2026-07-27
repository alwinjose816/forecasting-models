from load_supabase import supabase

import pandas as pd
import numpy as np


def build_drfi_v2():

    print(
        "Loading forecastability dataset..."
    )

    result = (
        supabase
        .table(
            "forecastability_dataset"
        )
        .select("*")
        .execute()
    )

    df = pd.DataFrame(
        result.data
    )

    print(
        "Rows:",
        len(df)
    )

    # ==========================
    # Normalize Features
    # ==========================

    def normalize(x):

        denom = x.max() - x.min()

        if denom == 0:
            return pd.Series(
                np.zeros(len(x)),
                index=x.index
            )

        return (
            x - x.min()
        ) / denom
   

    df["transition_entropy_n"] = normalize(
        df["transition_entropy"]
    )

    df["transition_strength_n"] = normalize(
        df["transition_strength"]
    )

    df["shock_score_n"] = normalize(
        df["shock_score"]
    )

    df["z_score_n"] = normalize(
        np.abs(df["z_score"])
    )

    df["state_stability_n"] = normalize(
        df["state_stability"]
    )

    # ==========================
    # Correlation-based Weights
    # ==========================

    w_entropy = 0.267
    w_transition = 0.230
    w_stability = 0.185
    w_zscore = 0.160
    w_shock = 0.158

    # ==========================
    # Risk Score
    # ==========================

    df["risk_score"] = (

        w_entropy
        * df["transition_entropy_n"]

        +

        w_transition
        * df["transition_strength_n"]

        +

        w_stability
        * (1 - df["state_stability_n"])

        +

        w_zscore
        * df["z_score_n"]

        +

        w_shock
        * df["shock_score_n"]

    )
    # ==========================
    # Normalize
    # ==========================

    risk_min = (
        df["risk_score"]
        .min()
    )

    risk_max = (
        df["risk_score"]
        .max()
    )

    df["drfi"] = (

        1 -

        (
            (
                df["risk_score"]
                - risk_min
            )
            /
            (
                risk_max
                - risk_min
            )
        )

    )
    print(
        "\nDRFI Range"
    )

    print(
        "Min:",
        round(df["drfi"].min(), 4)
    )

    print(
        "Max:",
        round(df["drfi"].max(), 4)
    )

    # ==========================
    # Zones
    # ==========================

    df["zone"] = np.where(
        df["drfi"] >= 0.80,
        "High",
        np.where(
            df["drfi"] >= 0.60,
            "Medium",
            "Low"
        )
    )

    print(
        "\nZone Distribution"
    )

    print(
        df["zone"]
        .value_counts()
    )
    print(
        df[
            [
                "product_code",
                "date",
                "drfi",
                "zone"
            ]
        ].head()
    )
    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.strftime("%Y-%m-%d")
    )

    # ==========================
    # Update Supabase
    # ==========================

    batch_size = 500

    records = df[
        [
            "product_code",
            "date",
            "drfi",
            "zone"
        ]
    ].to_dict(
        "records"
    )

    for row in records:

        (
            supabase
            .table(
                "forecastability_dataset"
            )
            .update(
                {
                    "drfi":
                    float(
                        row["drfi"]
                    ),

                    "zone":
                    str(
                        row["zone"]
                    )
                }
            )
            .eq(
                "product_code",
                row["product_code"]
            )
            .eq(
                "date",
                row["date"]
            )
            .execute()
        )
    print(
        "\nHighest Risk Days"
    )

    print(

        df[
            [
                "date",
                "demand",
                "drfi",
                "shock_score",
                "transition_strength",
                "transition_entropy"
            ]
        ]

        .sort_values(
            "drfi"
        )

        .head(20)
    )

    print(
        "\nDRFI Updated Successfully"
    )


if __name__ == "__main__":

    build_drfi_v2()