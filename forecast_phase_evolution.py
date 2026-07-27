# ==========================================================
# PHASE EVOLUTION ENGINE
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from load_supabase import supabase


# ==========================================================
# CONFIGURATION
# ==========================================================

TRAIN_RATIO = 0.80


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(product="Overall"):

    print("\n" + "="*70)
    print("LOADING PHASE DATASET")
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

    if df.empty:

        raise ValueError("Dataset not found.")

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    print()

    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    print()

    print(df.columns.tolist())

    return df


# ==========================================================
# CREATE PHASE EPISODES
# ==========================================================

def create_phase_episodes(df):

    print("\n" + "="*70)
    print("CREATING PHASE EPISODES")
    print("="*70)

    df = df.copy()

    # ------------------------------------------------------
    # Detect every time phase changes
    # ------------------------------------------------------

    df["phase_change"] = (

        df["demand_phase"]

        !=

        df["demand_phase"].shift()

    )

    # ------------------------------------------------------
    # Episode ID
    # ------------------------------------------------------

    df["episode_id"] = (

        df["phase_change"]

        .cumsum()

    )

    print()

    print(

        df[[

            "sales_date",

            "demand_phase",

            "phase_change",

            "episode_id"

        ]].head(40)

    )

    print()

    print(

        "Total Episodes :",

        df["episode_id"].nunique()

    )

    return df
# ==========================================================
# EPISODE LENGTH
# ==========================================================

def compute_episode_length(df):

    print("\n" + "="*70)
    print("EPISODE LENGTH")
    print("="*70)

    episode = (

        df

        .groupby("episode_id")

        .agg(

            phase=("demand_phase", "first"),

            start_date=("sales_date", "min"),

            end_date=("sales_date", "max"),

            episode_length=("sales_date", "count")

        )

        .reset_index()

    )

    print()

    print(episode.head(20))

    print()

    print("Total Episodes :", len(episode))

    return episode
# ==========================================================
# PHASE AGE
# ==========================================================

def compute_phase_age(df):

    print("\n" + "="*70)
    print("COMPUTING PHASE AGE")
    print("="*70)

    df = df.copy()

    df["phase_age"] = (

        df

        .groupby("episode_id")

        .cumcount()

        + 1

    )

    print()

    print(

        df[[

            "sales_date",

            "demand_phase",

            "episode_id",

            "phase_age"

        ]].head(30)

    )

    return df
# ==========================================================
# REMAINING LIFE
# ==========================================================

def compute_remaining_life(df):

    print("\n" + "="*70)
    print("COMPUTING REMAINING LIFE")
    print("="*70)

    df = df.copy()

    episode_length = (

        df

        .groupby("episode_id")["phase_age"]

        .transform("max")

    )

    df["remaining_life"] = (

        episode_length

        -

        df["phase_age"]

    )

    print()

    print(

        df[[

            "sales_date",

            "demand_phase",

            "phase_age",

            "remaining_life"

        ]].head(30)

    )

    return df
# ==========================================================
# PHASE LIFETIME STATISTICS
# ==========================================================

def compute_phase_statistics(episode):

    print("\n" + "="*70)
    print("PHASE LIFETIME STATISTICS")
    print("="*70)

    stats = (

        episode

        .groupby("phase")

        .agg(

            number_of_episodes=("episode_id","count"),

            mean_lifetime=("episode_length","mean"),

            median_lifetime=("episode_length","median"),

            std_lifetime=("episode_length","std"),

            min_lifetime=("episode_length","min"),

            max_lifetime=("episode_length","max")

        )

        .reset_index()

    )

    stats["std_lifetime"] = (

        stats["std_lifetime"]

        .fillna(0)

    )

    print()

    print(stats)

    return stats
# ==========================================================
# SAVE RESULTS
# ==========================================================

def save_phase_evolution(df, stats):

    print("\n" + "="*70)
    print("SAVING PHASE EVOLUTION")
    print("="*70)

    print()

    print(df.head())

    print()

    print(stats)

    print()

    print("Ready to save.")
if __name__ == "__main__":

    df = load_dataset()

    df = create_phase_episodes(df)

    episode = compute_episode_length(df)

    df = compute_phase_age(df)

    df = compute_remaining_life(df)

    stats = compute_phase_statistics(episode)

    save_phase_evolution(df, stats)
