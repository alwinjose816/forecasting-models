# ==========================================================
# LOAD DATASET
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from load_supabase import supabase
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
# NORMALIZE
# ==========================================================

def normalize(series):

    return (

        series - series.min()

    ) / (

        series.max() - series.min() + 1e-8

    )
# ==========================================================
# PHASE ENERGY
# ==========================================================

def compute_phase_energy(df):

    print("\n" + "="*70)
    print("COMPUTING PHASE ENERGY")
    print("="*70)

    entropy = normalize(df["phase_entropy"])

    transition = normalize(df["transition_strength"])

    shock = normalize(df["absolute_shock"])

    instability = normalize(
        1 - df["phase_stability"]
    )

    ptri = normalize(df["ptri"])

    df["phase_energy"] = (

        0.30 * entropy +

        0.25 * transition +

        0.20 * shock +

        0.15 * instability +

        0.10 * ptri

    )

    print(df[[
        "sales_date",
        "phase_energy"
    ]].head())

    return df
# ==========================================================
# DYNAMIC ALPHA
# ==========================================================

def compute_dynamic_alpha(df):

    memory = normalize(

        df["rolling_memory_strength"]

    )

    df["dynamic_alpha"] = (

        0.60 +

        0.35 * memory

    )

    print("\nDynamic Alpha")

    print(

        df[[
            "sales_date",
            "dynamic_alpha"
        ]].head()

    )

    return df
def compute_phase_potential(df):

    print("\n" + "="*70)
    print("COMPUTING LATENT PHASE POTENTIAL")
    print("="*70)

    phi = []

    previous = df["phase_energy"].iloc[0]

    for _, row in df.iterrows():

        alpha = row["dynamic_alpha"]

        energy = row["phase_energy"]

        current = (

            alpha * previous +

            (1 - alpha) * energy

        )

        phi.append(current)

        previous = current

    df["latent_phase_potential"] = phi

    print()

    print(

        df[[
            "sales_date",
            "latent_phase_potential"
        ]].head(20)

    )

    return df
# ==========================================================
# PHASE VELOCITY
# ==========================================================

def compute_phase_velocity(df):

    print("\n" + "="*70)
    print("COMPUTING LATENT PHASE VELOCITY")
    print("="*70)

    df["latent_phase_velocity"] = (

        df["latent_phase_potential"]

        .diff()

        .fillna(0)

    )

    print()

    print(

        df[[

            "sales_date",

            "latent_phase_velocity"

        ]].head(20)

    )

    return df
# ==========================================================
# PHASE CURVATURE
# ==========================================================

def compute_phase_curvature(df):

    print("\n" + "="*70)
    print("COMPUTING LATENT PHASE CURVATURE")
    print("="*70)

    df["latent_phase_curvature"] = (

        df["latent_phase_velocity"]

        .diff()

        .fillna(0)

    )

    print()

    print(

        df[[

            "sales_date",

            "latent_phase_curvature"

        ]].head(20)

    )

    return df
# ==========================================================
# SAVE LATENT PHASE VARIABLES
# ==========================================================

from datetime import datetime, date

def save_phase_potential(df):

    print("\n" + "="*70)
    print("SAVING LATENT PHASE VARIABLES")
    print("="*70)

    save_df = df[[
        "sales_date",
        "product",
        "phase_energy",
        "dynamic_alpha",
        "latent_phase_potential",
        "latent_phase_velocity",
        "latent_phase_curvature"
    ]].copy()

    # --------------------------------------------
    # Convert Timestamp to ISO string
    # --------------------------------------------

    for col in save_df.columns:

        save_df[col] = save_df[col].apply(

            lambda x: x.isoformat()

            if isinstance(
                x,
                (pd.Timestamp, datetime, date)
            )

            else x

        )

    print()

    print(save_df.head())

    records = save_df.to_dict("records")

    batch_size = 500

    for i in range(0, len(records), batch_size):

        batch = records[i:i+batch_size]

        supabase.table(
            "final_training_data"
        ).upsert(
            batch,
            on_conflict="product,sales_date"
        ).execute()

    print()

    print(f"Saved {len(records)} rows.")
# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    data = load_dataset()

    data = compute_phase_energy(data)

    data = compute_dynamic_alpha(data)

    data = compute_phase_potential(data)

    data = compute_phase_velocity(data)

    data = compute_phase_curvature(data)

    save_phase_potential(data)
