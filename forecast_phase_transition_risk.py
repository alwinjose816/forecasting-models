# ==========================================================
# PHASE TRANSITION RISK INDEX (PTRI)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from load_supabase import supabase
# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(product="Overall"):

    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)

    response = (

        supabase

        .table("final_training_data")

        .select("*")

        .eq("product", product)

        .order("sales_date")

        .execute()

    )

    df = pd.DataFrame(response.data)

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    print()

    print("Rows :", len(df))

    return df
# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize(series):

    return (

        series - series.min()

    ) / (

        series.max() - series.min() + 1e-8

    )
# ==========================================================
# PHASE TRANSITION RISK INDEX
# ==========================================================

def compute_ptri(df):

    print("\n" + "="*70)
    print("COMPUTING PTRI")
    print("="*70)

    entropy = normalize(

        df["phase_entropy"]

    )

    transition = normalize(

        df["transition_strength"]

    )

    shock = normalize(

        df["absolute_shock"]

    )

    instability = normalize(

        1 - df["phase_stability"]

    )

    velocity = normalize(

        np.abs(

            df["phase_velocity"]

        )

    )

    # -----------------------------------
    # PTRI
    # -----------------------------------

    df["PTRI"] = (

        0.25 * entropy +

        0.25 * transition +

        0.20 * shock +

        0.20 * instability +

        0.10 * velocity

    )

    print()

    print(

        df[[
            "sales_date",
            "demand_phase",
            "PTRI"
        ]].head(20)

    )

    print()

    print(

        df["PTRI"].describe()

    )

    return df
# ==========================================================
# RISK LEVEL
# ==========================================================

def classify_risk(df):

    print("\n" + "="*70)
    print("CLASSIFYING RISK")
    print("="*70)

    df["PTRI_Level"] = pd.cut(

        df["PTRI"],

        bins=[

            0,

            0.33,

            0.66,

            1.01

        ],

        labels=[

            "Low",

            "Medium",

            "High"

        ]

    )

    print()

    print(

        df["PTRI_Level"]

        .value_counts()

    )

    return df
# ==========================================================
# SAVE PTRI
# ==========================================================

def save_ptri(df):

    print("\n" + "="*70)
    print("SAVING PTRI")
    print("="*70)

    save_df = df[[
        "sales_date",
        "product",
        "PTRI",
        "PTRI_Level"
    ]].copy()

    save_df = save_df.rename(columns={
        "PTRI": "ptri",
        "PTRI_Level": "ptri_level"
    })
    # Convert datetime
    save_df["sales_date"] = save_df["sales_date"].dt.strftime("%Y-%m-%d")

    # Convert categorical to string
    save_df["ptri_level"] = save_df["ptri_level"].astype(str)

    # Convert numpy float to Python float
    save_df["ptri"] = save_df["ptri"].astype(float)

    records = save_df.to_dict("records")
    print()

    print(
        save_df.head()
    )

    batch_size = 500

    for i in range(0, len(records), batch_size):

        batch = records[i:i+batch_size]

        supabase.table(
            "final_training_data"
        ).upsert(
            batch,
            on_conflict="product,sales_date"
        ).execute()

    print(f"Saved {len(records)} rows.")
if __name__ == "__main__":

    data = load_dataset()

    data = compute_ptri(data)

    data = classify_risk(data)

    save_ptri(data)