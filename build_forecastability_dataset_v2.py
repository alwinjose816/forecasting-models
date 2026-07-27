from load_data import load_dealer_orders
from load_supabase import supabase

from forecast_deep_demand_transitions import (
    run_forecast_deep_demand_transitions
)

import pandas as pd
import numpy as np


def build_forecastability_dataset_v2():

    df = load_dealer_orders()

    products = (
        df["product_code"]
        .dropna()
        .unique()
        .tolist()
    )

    products.insert(
        0,
        "Overall"
    )

    all_rows = []

    for product in products:

        try:

            print(
                f"\nProcessing {product}"
            )

            transition_result = (
                run_forecast_deep_demand_transitions(
                    product
                )
            )

            regime_df = pd.DataFrame(
                transition_result["regimes"]
            )

            regime_df["date"] = pd.to_datetime(
                regime_df["date"]
            )

            if product == "Overall":

                product_df = df.copy()

            else:

                product_df = df[
                    df["product_code"]
                    == product
                ].copy()

            daily = (
                product_df.groupby(
                    product_df["order_date"].dt.date
                )["total_weight_mt"]
                .sum()
                .reset_index()
            )

            daily.columns = [
                "date",
                "demand"
            ]

            daily["date"] = pd.to_datetime(
                daily["date"]
            )

            full_df = daily.merge(

                regime_df[
                    [
                        "date",
                        "state",
                        "rolling_memory_strength",
                        "transition_strength",
                        "transition_entropy",
                        "state_stability",
                        "shock_score",
                        "z_score"
                    ]
                ],

                on="date",

                how="inner"
            )

            full_df["product_code"] = product

            full_df = full_df.rename(
                columns={
                    "rolling_memory_strength":
                    "memory_strength"
                }
            )

            full_df["forecast_error"] = None
            full_df["drfi"] = None
            full_df["zone"] = None

            all_rows.extend(
                full_df[
                    [
                        "date",
                        "demand",
                        "forecast_error",
                        "drfi",
                        "zone",
                        "product_code",
                        "state",
                        "memory_strength",
                        "transition_strength",
                        "transition_entropy",
                        "state_stability",
                        "shock_score",
                        "z_score"
                    ]
                ]
                .to_dict("records")
            )

            print(
                "Rows:",
                len(full_df)
            )

        except Exception as e:

            print(
                f"Failed {product}"
            )

            print(e)

    final_df = pd.DataFrame(
        all_rows
    )

    print(
        "\nTotal Rows:",
        len(final_df)
    )

    # ==========================
    # Fix Timestamp issue
    # ==========================

    final_df["date"] = (
        pd.to_datetime(
            final_df["date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    final_df = final_df.replace(
        [np.inf, -np.inf],
        None
    )

    final_df = final_df.where(
        pd.notnull(final_df),
        None
    )

    print(final_df.dtypes)
    print(final_df.head())

    print(
        "Date type:",
        type(
            final_df.iloc[0]["date"]
        )
    )

    supabase.table(
        "forecastability_dataset"
    ).delete().neq(
        "date",
        "1900-01-01"
    ).execute()

    batch_size = 500

    records = (
        final_df
        .to_dict("records")
    )

    print(
        type(
            final_df.iloc[0]["date"]
        )
    )
    for i in range(
        0,
        len(records),
        batch_size
    ):

        batch = records[
            i:i + batch_size
        ]

        supabase.table(
            "forecastability_dataset"
        ).insert(
            batch
        ).execute()

        print(
            f"Inserted {i + len(batch)} rows"
        )

    print(
        "\nUpload Complete"
    )


if __name__ == "__main__":

    build_forecastability_dataset_v2()