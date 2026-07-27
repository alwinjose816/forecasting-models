# ==========================================================
# PHASE TRANSITION EXPLANATION
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

    if df.empty:
        raise ValueError("Dataset not found.")

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    print()

    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    return df
# ==========================================================
# DETECT PHASE TRANSITIONS
# ==========================================================

def detect_phase_transitions(df):

    print("\n" + "="*70)
    print("DETECTING PHASE TRANSITIONS")
    print("="*70)

    df = df.copy()

    df["previous_phase"] = (

        df["demand_phase"]

        .shift(1)

    )

    transitions = (

        df[

            df["previous_phase"] != df["demand_phase"]

        ]

        .copy()

    )

    transitions = transitions.iloc[1:]

    print()

    print("Transitions Found :", len(transitions))

    print()

    print(

        transitions[[

            "sales_date",

            "previous_phase",

            "demand_phase"

        ]].head(20)

    )

    return transitions
# ==========================================================
# EXTRACT TRANSITION WINDOWS
# ==========================================================

def extract_transition_windows(df, transitions, window=10):

    print("\n" + "="*70)
    print("EXTRACTING TRANSITION WINDOWS")
    print("="*70)

    all_windows = []

    for _, transition in transitions.iterrows():

        idx = transition.name

        start = idx - window
        end = idx + window

        if start < 0 or end >= len(df):
            continue

        segment = df.iloc[start:end+1].copy()

        segment["relative_day"] = np.arange(-window, window+1)

        segment["transition_date"] = transition["sales_date"]

        segment["from_phase"] = transition["previous_phase"]

        segment["to_phase"] = transition["demand_phase"]

        all_windows.append(segment)

    windows = pd.concat(all_windows, ignore_index=True)

    print()

    print("Windows :", windows["transition_date"].nunique())

    print("Rows :", len(windows))

    print()

    print(

        windows[[
            "transition_date",
            "relative_day",
            "sales_date",
            "demand_phase"
        ]].head(30)

    )

    return windows
# ==========================================================
# AVERAGE TRANSITION PROFILE
# ==========================================================

def build_transition_profile(windows):

    print("\n" + "="*70)
    print("BUILDING TRANSITION PROFILE")
    print("="*70)

    profile = (

        windows

        .groupby("relative_day")

        .agg(

            entropy=("phase_entropy","mean"),

            stability=("phase_stability","mean"),

            shock=("absolute_shock","mean"),

            transition=("transition_strength","mean"),

            ptri=("ptri","mean"),

            latent_phi=("latent_phase_potential","mean")

        )

        .reset_index()

    )

    profile = profile.round(4)

    print()

    print(profile)

    return profile
# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    data = load_dataset()

    transitions = detect_phase_transitions(
        data
    )

    windows = extract_transition_windows(
        data,
        transitions
    )

    profile = build_transition_profile(
        windows
    )