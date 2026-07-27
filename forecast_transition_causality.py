# ==========================================================
# TRANSITION CAUSALITY ANALYSIS
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

    df["sales_date"] = pd.to_datetime(df["sales_date"])

    print()

    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    return df
# ==========================================================
# DETECT TRANSITIONS
# ==========================================================

def detect_transitions(df):

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

    print("Transitions :", len(transitions))

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

def extract_transition_windows(df, transitions):

    print("\n" + "="*70)
    print("EXTRACTING TRANSITION WINDOWS")
    print("="*70)

    windows = []

    for idx in transitions.index:

        start = idx - 10
        end = idx + 10

        if start < 0:
            continue

        if end >= len(df):
            continue

        temp = df.iloc[start:end+1].copy()

        temp["transition_date"] = df.loc[idx, "sales_date"]

        temp["relative_day"] = np.arange(-10,11)

        windows.append(temp)

    windows = pd.concat(
        windows,
        ignore_index=True
    )

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

        ]].head(25)

    )

    return windows
# ==========================================================
# CAUSAL FEATURE EVOLUTION
# ==========================================================

def analyse_causality(windows):

    print("\n" + "="*70)
    print("ANALYSING CAUSAL SIGNALS")
    print("="*70)

    features = [

        "transition_strength",

        "absolute_shock",

        "phase_stability",

        "phase_entropy",

        "ptri"

    ]

    summary = (

        windows

        .groupby("relative_day")[features]

        .mean()

        .reset_index()

    )

    print()

    print(summary)

    return summary
if __name__ == "__main__":

    df = load_dataset()

    transitions = detect_transitions(df)

    windows = extract_transition_windows(

        df,

        transitions

    )

    summary = analyse_causality(

        windows

    )