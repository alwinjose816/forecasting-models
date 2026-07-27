import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np

from load_supabase import supabase

PHASE_MODEL = "models/genome_phase_generator.pkl"
PRODUCT = "Overall"
# ==========================================================
# LOAD FUTURE GENOME
# ==========================================================

def load_future_genome(product=PRODUCT):

    print("\n" + "="*70)
    print("LOADING FUTURE GENOME")
    print("="*70)

    response = (

        supabase

        .table("future_genome")

        .select("*")

        .eq("product", product)

        .order("sales_date")

        .execute()

    )

    df = pd.DataFrame(response.data)

    print()

    print("Rows :", len(df))

    return df
# ==========================================================
# LOAD GENE RELIABILITY
# ==========================================================

def load_reliability(product=PRODUCT):

    response = (

        supabase

        .table("gene_reliability")

        .select("*")

        .eq("product", product)

        .execute()

    )

    reliability = pd.DataFrame(response.data)

    return dict(

        zip(

            reliability.gene,

            reliability.reliability

        )

    )
# ==========================================================
# APPLY RELIABILITY WEIGHTS
# ==========================================================

def weight_genome(df, reliability):

    print("\n" + "="*70)
    print("WEIGHTING GENOME")
    print("="*70)

    weighted = pd.DataFrame()

    weighted["sales_date"] = df["sales_date"]

    weighted["product"] = df["product"]

    for gene in reliability:

        weighted[f"weighted_{gene}"] = (

            df[gene]

            *

            reliability[gene]

        )

    print()

    print(weighted.head())

    return weighted
# ==========================================================
# LOAD PHASE MODEL
# ==========================================================

def load_phase_model():

    print("\n" + "="*70)
    print("LOADING PHASE MODEL")
    print("="*70)

    model = joblib.load(

        PHASE_MODEL

    )

    print()

    print("Model Loaded")

    return model
# ==========================================================
# PREDICT FUTURE PHASES
# ==========================================================

def predict_future_phases(

        weighted_df,
        phase_model

):

    print("\n" + "="*70)
    print("PREDICTING FUTURE PHASES")
    print("="*70)

    feature_columns = [

        c for c in weighted_df.columns

        if c.startswith("weighted_")

    ]

    X = weighted_df[feature_columns]

    phases = phase_model.predict(X)

    probabilities = phase_model.predict_proba(X)

    results = weighted_df[

        [

            "sales_date",

            "product"

        ]

    ].copy()

    results["forecast_day"] = np.arange(

        1,

        len(results)+1

    )

    results["predicted_phase"] = phases.astype(int)

    results["confidence"] = probabilities.max(axis=1)

    for i, cls in enumerate(phase_model.classes_):

        col = f"p{int(cls)}"      # lowercase

        results[col] = probabilities[:, i]

    print()

    print(results.head())

    return results
# ==========================================================
# SAVE FUTURE PHASES
# ==========================================================

def save_future_phases(results):

    print("\n" + "="*70)
    print("SAVING FUTURE PHASES")
    print("="*70)

    df = results.copy()

    df["sales_date"] = pd.to_datetime(

        df["sales_date"]

    ).dt.strftime(

        "%Y-%m-%d"

    )

    supabase.table(

        "future_phase_prediction"

    ).upsert(

        df.to_dict("records"),

        on_conflict="product,sales_date"

    ).execute()

    print()

    print("Saved :", len(df))
if __name__ == "__main__":

    future_genome = load_future_genome()

    reliability = load_reliability()

    weighted_genome = weight_genome(

        future_genome,

        reliability

    )

    phase_model = load_phase_model()

    results = predict_future_phases(

        weighted_genome,

        phase_model

    )

    save_future_phases(

        results

    )

    print("\n")
    print("="*70)
    print("30 DAY PHASE FORECAST COMPLETE")
    print("="*70)

    print(results)

