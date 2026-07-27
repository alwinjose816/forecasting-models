import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from load_supabase import supabase

MAX_HORIZON = 30
# ==========================================================
# LOAD PREDICTED PHASES
# ==========================================================

def load_predicted_phases(product="Overall"):

    print("\n" + "="*70)
    print("LOADING PREDICTED PHASES")
    print("="*70)

    response = (

        supabase

        .table("predicted_phase")

        .select("*")

        .eq("product", product)

        .order("sales_date")

        .execute()

    )

    df = pd.DataFrame(response.data)

    if df.empty:
        raise ValueError("Predicted phases not found.")

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    print()

    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    print()

    print(df[[
        "sales_date",
        "predicted_phase",
        "confidence"
    ]].head())

    return df
# ==========================================================
# BUILD TRANSITION DYNAMICS
# ==========================================================

def build_transition_matrix(df):

    print("\n" + "="*70)
    print("BUILDING TRANSITION MATRIX")
    print("="*70)

    # ------------------------------------------------------
    # Unique Phases
    # ------------------------------------------------------

    phases = sorted(
        df["predicted_phase"].unique()
    )

    # ------------------------------------------------------
    # Transition Counts
    # ------------------------------------------------------

    transition = pd.DataFrame(

        0,

        index=phases,

        columns=phases,

        dtype=float

    )

    current = df["predicted_phase"].values[:-1]
    future = df["predicted_phase"].values[1:]

    for c, f in zip(current, future):

        transition.loc[c, f] += 1

    print("\nTransition Counts\n")
    print(transition)

    # ------------------------------------------------------
    # Transition Probabilities
    # ------------------------------------------------------

    probability = transition.div(

        transition.sum(axis=1),

        axis=0

    ).fillna(0)
    print("\n" + "="*70)
    print("30-DAY PHASE EVOLUTION")
    print("="*70)

    P = probability.values
    phases = probability.index.tolist()

    # Start from the latest predicted phase
    current_phase = df["predicted_phase"].iloc[-1]

    state = np.zeros(len(phases))
    state[phases.index(current_phase)] = 1

    forecast = []

    for day in range(1, 31):

        state = state @ P

        predicted = phases[np.argmax(state)]

        forecast.append({

            "day": day,

            "predicted_phase": predicted,

            "confidence": float(np.max(state)),

            "P1": state[0],
            "P2": state[1],
            "P3": state[2],
            "P4": state[3],
            "P5": state[4]

        })

    forecast = pd.DataFrame(forecast)

    print(forecast)

    print("\n" + "="*70)
    print("ONE STEP TRANSITION PROBABILITIES")
    print("="*70)

    print()
    print(probability.round(3))

    # ------------------------------------------------------
    # Multi Horizon
    # ------------------------------------------------------

    print("\n" + "="*70)
    print("MULTI-HORIZON TRANSITIONS")
    print("="*70)

    horizons = [3, 7, 14, 30]

    probability_matrix = probability.values

    for h in horizons:

        future_matrix = np.linalg.matrix_power(

            probability_matrix,

            h

        )

        future_df = pd.DataFrame(

            future_matrix,

            index=phases,

            columns=phases

        )

        print(f"\n{h}-Day Transition")
        print(future_df.round(3))

    # ------------------------------------------------------
    # Phase Persistence
    # ------------------------------------------------------

    print("\n" + "="*70)
    print("PHASE PERSISTENCE")
    print("="*70)

    persistence = []

    for phase in phases:

        p = probability.loc[phase, phase]

        if p >= 0.999:

            duration = np.inf

        else:

            duration = 1 / (1 - p)

        persistence.append({

            "phase": phase,

            "stay_probability": p,

            "expected_duration": duration

        })

    persistence = pd.DataFrame(persistence)

    print()
    print(persistence)

    # ------------------------------------------------------
    # Transition Entropy
    # ------------------------------------------------------

    print("\n" + "="*70)
    print("TRANSITION ENTROPY")
    print("="*70)

    entropy = -(

        probability.replace(0, np.nan)

        * np.log2(

            probability.replace(0, np.nan)

        )

    ).sum(axis=1).fillna(0)

    entropy = pd.DataFrame({

        "phase": phases,

        "transition_entropy": entropy.values

    })

    print()
    print(entropy)

    return {

        "counts": transition,

        "probability": probability,

        "persistence": persistence,

        "entropy": entropy

    }
# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    predicted_phase = load_predicted_phases()

    transition_results = build_transition_matrix(
        predicted_phase
    )