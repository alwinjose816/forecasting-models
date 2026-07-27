from xgboost_forecast_DRFIv2 import (
    run_xgboost_forecast_DRFIv2
)

from load_data import load_dealer_orders

import pandas as pd

from supabase import create_client


SUPABASE_URL = "https://yvuxjdpvvtpbngoubqgq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl2dXhqZHB2dnRwYm5nb3VicWdxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg1OTY1MTEsImV4cCI6MjA5NDE3MjUxMX0.kI3mBen9jhQ4avBkL93xkAZg27dVvyE7NJACPECrKsE"

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


def build_forecastability_dataset():

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

            result = (
                run_xgboost_forecast_DRFIv2(
                    product
                )
            )

            rows = result[
                "forecastability_dataset"
            ]

            all_rows.extend(
                rows
            )

        except Exception as e:

            print(
                f"Failed: {product}"
            )

            print(e)

    final_df = pd.DataFrame(
        all_rows
    )

    print(
        "\nTotal Rows:",
        len(final_df)
    )

    print(
        final_df.head()
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
        "\nForecastability Dataset Uploaded"
    )


if __name__ == "__main__":

    build_forecastability_dataset()