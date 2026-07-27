# ==========================================================
# PHASE GRAPH
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from load_supabase import supabase
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

    return df
# ==========================================================
# CREATE PHASE TRANSITIONS
# ==========================================================

def create_phase_pairs(df):

    print("\n" + "="*70)
    print("CREATING PHASE PAIRS")
    print("="*70)

    df = df.copy()

    df["next_phase"] = (

        df["demand_phase"]

        .shift(-1)

    )

    df = df.dropna(subset=["next_phase"])

    df["next_phase"] = (

        df["next_phase"]

        .astype(int)

    )

    print()

    print(

        df[[

            "sales_date",

            "demand_phase",

            "next_phase"

        ]].head(20)

    )

    return df
# ==========================================================
# BUILD TRANSITION MATRIX
# ==========================================================

def build_transition_matrix(df):

    print("\n" + "="*70)
    print("BUILDING TRANSITION MATRIX")
    print("="*70)

    matrix = (

        df

        .groupby(

            [

                "demand_phase",

                "next_phase"

            ]

        )

        .size()

        .reset_index(

            name="transition_count"

        )

    )

    print()

    print(matrix)

    return matrix
if __name__ == "__main__":

    df = load_dataset()

    df = create_phase_pairs(df)

    matrix = build_transition_matrix(df)
# ==========================================================
# COMPUTE TRANSITION PROBABILITIES
# ==========================================================

def compute_transition_probabilities(matrix):

    print("\n" + "="*70)
    print("COMPUTING TRANSITION PROBABILITIES")
    print("="*70)

    matrix = matrix.copy()

    matrix["total"] = (

        matrix

        .groupby("demand_phase")["transition_count"]

        .transform("sum")

    )

    matrix["transition_probability"] = (

        matrix["transition_count"]

        /

        matrix["total"]

    )

    print()

    print(

        matrix[[

            "demand_phase",

            "next_phase",

            "transition_count",

            "transition_probability"

        ]]

    )

    return matrix
if __name__ == "__main__":

    df = load_dataset()

    df = create_phase_pairs(df)

    matrix = build_transition_matrix(df)

    matrix = compute_transition_probabilities(
        matrix
    )