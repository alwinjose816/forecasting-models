# ==========================================================
# PHASE TRANSITION PROBABILITY
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from load_supabase import supabase
# ==========================================================
# LOAD PHASE DATASET
# ==========================================================

def load_phase_dataset(product="Overall"):

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

    return df
# ==========================================================
# BUILD PHASE TRANSITIONS
# ==========================================================

def build_phase_transitions(df):

    print("\n" + "="*70)
    print("BUILDING PHASE TRANSITIONS")
    print("="*70)

    transitions = df[[
        "sales_date",
        "demand_phase"
    ]].copy()

    transitions["next_phase"] = (

        transitions["demand_phase"]

        .shift(-1)

    )

    transitions = transitions.dropna()

    transitions["next_phase"] = (

        transitions["next_phase"]

        .astype(int)

    )

    print()

    print(transitions.head(15))

    return transitions
# ==========================================================
# TRANSITION PROBABILITY MATRIX
# ==========================================================

def build_transition_matrix(transitions):

    print("\n" + "=" * 70)
    print("BUILDING TRANSITION MATRIX")
    print("=" * 70)

    transition_counts = pd.crosstab(

        transitions["demand_phase"],

        transitions["next_phase"]

    )

    transition_probability = (

        transition_counts

        .div(

            transition_counts.sum(axis=1),

            axis=0

        )

    )

    print("\nTransition Counts\n")
    print(transition_counts)

    print("\nTransition Probability\n")
    print(

        transition_probability.round(3)

    )

    return (

        transition_counts,

        transition_probability

    )
if __name__ == "__main__":

    phase_data = load_phase_dataset()

    transitions = build_phase_transitions(
        phase_data
    )

    transition_counts, transition_probability = (

        build_transition_matrix(

            transitions

        )

    )
