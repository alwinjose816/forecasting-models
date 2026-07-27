from supabase import create_client
import pandas as pd
from datetime import datetime, date

SUPABASE_URL = (
    "https://yvuxjdpvvtpbngoubqgq.supabase.co"
)

SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl2dXhqZHB2dnRwYm5nb3VicWdxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg1OTY1MTEsImV4cCI6MjA5NDE3MjUxMX0.kI3mBen9jhQ4avBkL93xkAZg27dVvyE7NJACPECrKsE"
)

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)
# ==========================================================
# SAVE DEMAND PHASE FEATURES
# ==========================================================

def save_phase_features(df):

    if isinstance(df, pd.DataFrame):

        df = df.copy()

        for col in df.columns:

            df[col] = df[col].apply(
                lambda x: x.isoformat()
                if isinstance(x, (pd.Timestamp, datetime, date))
                else x
            )

        # ==========================================================
        # KEEP ONLY COLUMNS THAT EXIST IN SUPABASE
        # ==========================================================

        columns_to_save = [

            "sales_date",
            "product",
            "demand",

            "memory_short",
            "memory_weekly",
            "memory_biweekly",
            "memory_monthly",

            "rolling_memory_strength",
            "memory_half_life",
            "memory_state",
            "memory_residual",
            "memory_ratio",
            "memory_drift",

            "local_mean_7",
            "local_mean_14",
            "local_mean_30",
            "local_mean_90",

            "local_std_7",
            "local_std_14",
            "local_std_30",
            "local_std_90",

            "local_cv_7",
            "local_cv_14",
            "local_cv_30",
            "local_cv_90",

            "trend_7",
            "trend_14",
            "trend_30",
            "trend_90",

            "momentum_7",
            "momentum_14",
            "momentum_30",

            "acceleration_7",
            "acceleration_14",
            "acceleration_30",

            "demand_entropy_7",
            "demand_entropy_14",
            "demand_entropy_30",
            "demand_entropy_90",

            "skewness_30",
            "kurtosis_30",

            "shock_score",
            "absolute_shock",
            "shock_rate_30",
            "shock_rate_90",

            "transition_strength",
            "transition_direction",
            "transition_entropy_30",
            "transition_entropy_90",
            "transition_volatility",

            "state_stability",
            "behaviour_persistence",

            "demand_density",
            "days_since_last_order",

            "weekly_similarity",

            "day_of_week",
            "week_of_year",
            "month",
            "quarter",

            "drfi_memory",
            "drfi_stability",
            "drfi_shock",
            "drfi_seasonality",

            "complexity_score",

            "behaviour_dimension",

            "phase_potential",
            "phase_velocity",
            "phase_curvature",

            "created_at"
        ]

        df = df[
            [c for c in columns_to_save if c in df.columns]
        ]

        records = df.to_dict(orient="records")

    else:

        records = df

    batch_size = 500

    for i in range(0, len(records), batch_size):

        batch = records[i:i + batch_size]

        try:
            supabase.table(
                "demand_phase_features"
            ).upsert(
                batch,
                on_conflict="product,sales_date"
            ).execute()

        except Exception as e:
            print(f"Batch {i // batch_size + 1} failed: {e}")
            raise

    print(f"Saved {len(records)} rows to Supabase.")
# ==========================================================
# SAVE BEHAVIOUR EMBEDDINGS
# ==========================================================

from datetime import datetime, date
import pandas as pd


def save_behaviour_embeddings(df):

    if isinstance(df, pd.DataFrame):

        df = df.copy()

        # ----------------------------------------------
        # Convert dates
        # ----------------------------------------------

        for col in df.columns:

            df[col] = df[col].apply(
                lambda x: x.isoformat()
                if isinstance(x, (pd.Timestamp, datetime, date))
                else x
            )

        # ----------------------------------------------
        # Keep only table columns
        # ----------------------------------------------

        columns_to_save = [

            "sales_date",
            "product",

            "embedding_1",
            "embedding_2",
            "embedding_3",
            "embedding_4",
            "embedding_5",
            "embedding_6",
            "embedding_7",
            "embedding_8",
            "embedding_9",
            "embedding_10",
            "embedding_11",
            "embedding_12",

            "reconstruction_error"

        ]

        df = df[
            [c for c in columns_to_save if c in df.columns]
        ]

        records = df.to_dict(
            orient="records"
        )

    else:

        records = df

    batch_size = 500

    for i in range(0, len(records), batch_size):

        batch = records[i:i + batch_size]

        try:

            supabase.table(
                "behaviour_embeddings"
            ).upsert(

                batch,

                on_conflict="product,sales_date"

            ).execute()

        except Exception as e:

            print(
                f"Batch {i//batch_size+1} failed:"
            )

            print(e)

            raise

    print()

    print(
        f"Saved {len(records)} Behaviour Embeddings."
    )