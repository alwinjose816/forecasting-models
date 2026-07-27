# ==========================================================
# PHASE 5
# DEMAND PHASE FINGERPRINTS
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from load_supabase import supabase
# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42
# ==========================================================
# LOAD FINAL TRAINING DATA
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
        raise ValueError(
            "No data found."
        )

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    print()

    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    return df
# ==========================================================
# BUILD PHASE FINGERPRINTS
# ==========================================================

def build_phase_fingerprints(df):

    print("\n" + "=" * 70)
    print("BUILDING PHASE FINGERPRINTS")
    print("=" * 70)

    fingerprint = (

        df

        .groupby("demand_phase")

        .agg(

            observations=("demand","count"),

            demand_mean=("demand","mean"),
            demand_std=("demand","std"),
            demand_min=("demand","min"),
            demand_max=("demand","max"),

            entropy_mean=("phase_entropy","mean"),
            entropy_std=("phase_entropy","std"),

            shock_mean=("absolute_shock","mean"),
            shock_std=("absolute_shock","std"),

            stability_mean=("phase_stability","mean"),
            stability_std=("phase_stability","std"),

            transition_mean=("transition_strength","mean"),
            transition_std=("transition_strength","std"),

            memory_mean=("rolling_memory_strength","mean"),

            confidence_mean=("phase_confidence","mean"),

            pagerank=("pagerank","mean"),

            behaviour=("behaviour_persistence","mean")

        )

    ).reset_index()
    fingerprint["demand_cv"] = (

        fingerprint["demand_std"]

        /

        fingerprint["demand_mean"]

    )

    fingerprint = fingerprint.round(3)

    print()

    print(fingerprint)

    return fingerprint
# ==========================================================
# ASSIGN SCIENTIFIC PHASE NAMES
# ==========================================================

def assign_phase_names(fingerprint):

    print("\n" + "=" * 70)
    print("ASSIGNING SCIENTIFIC PHASE NAMES")
    print("=" * 70)

    fp = fingerprint.copy()

    fp["phase_name"] = "Behaviour"

    # ------------------------------------------
    # Stable
    # ------------------------------------------

    stable_idx = (

        fp["stability_mean"].idxmax()

    )

    fp.loc[stable_idx, "phase_name"] = "Stable"

    # ------------------------------------------
    # Shock
    # ------------------------------------------

    remaining = fp[fp.phase_name == "Behaviour"]

    shock_idx = remaining["shock_mean"].idxmax()

    fp.loc[shock_idx, "phase_name"] = "Shock"

    # ------------------------------------------
    # Transition
    # ------------------------------------------

    remaining = fp[fp.phase_name == "Behaviour"]

    transition_idx = remaining["transition_mean"].idxmax()

    fp.loc[transition_idx, "phase_name"] = "Transition"

    # ------------------------------------------
    # Growth
    # ------------------------------------------

    remaining = fp[fp.phase_name == "Behaviour"]

    growth_idx = remaining["behaviour"].idxmax()

    fp.loc[growth_idx, "phase_name"] = "Growth"

    # ------------------------------------------
    # Recovery
    # ------------------------------------------

    remaining = fp[fp.phase_name == "Behaviour"]

    recovery_idx = remaining["memory_mean"].idxmax()

    fp.loc[recovery_idx, "phase_name"] = "Recovery"

    # ------------------------------------------
    # Remaining
    # ------------------------------------------

    fp.loc[
        fp.phase_name == "Behaviour",
        "phase_name"
    ] = "Volatile"

    print()

    print(

        fp[[

            "demand_phase",

            "phase_name",

            "stability_mean",

            "entropy_mean",

            "shock_mean",

            "transition_mean"

        ]]

    )
    print("\n" + "=" * 70)
    print("PHASE INTERPRETATION")
    print("=" * 70)

    for _, row in fp.iterrows():

        print()

        print(f"Phase {int(row['demand_phase'])}")

        print(f"Name : {row['phase_name']}")

        print(f"Demand Mean      : {row['demand_mean']:.2f}")

        print(f"Demand CV        : {row['demand_cv']:.3f}")

        print(f"Entropy          : {row['entropy_mean']:.3f}")

        print(f"Shock            : {row['shock_mean']:.3f}")

        print(f"Transition       : {row['transition_mean']:.3f}")

        print(f"Stability        : {row['stability_mean']:.3f}")

        print(f"Memory           : {row['memory_mean']:.3f}")

    return fp
if __name__ == "__main__":

    phase_df = load_phase_dataset()

    fingerprints = build_phase_fingerprints(
            phase_df
        )

    fingerprints = assign_phase_names(
        fingerprints
    )