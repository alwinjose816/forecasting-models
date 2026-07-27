# ==========================================================
# RECURSIVE PHASE EVOLUTION MODEL (RGPEM)
# Hierarchical Demand Genome Evolution Network (HDGEN)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import joblib

import numpy as np
import pandas as pd

from datetime import timedelta

from load_supabase import supabase

# ==========================================================
# CONFIGURATION
# ==========================================================

FORECAST_DAYS = 30

PRODUCT = "Overall"

RANDOM_STATE = 42

MODEL_FOLDER = "models/gene_models"

PHASE_MODEL = "models/genome_phase_generator.pkl"

# ==========================================================
# DEMAND GENOME
# ==========================================================

GENE_GROUPS = {

    "memory":[

        "memory_short",
        "memory_weekly",
        "memory_biweekly",
        "memory_monthly",
        "rolling_memory_strength",
        "memory_half_life",
        "memory_ratio",
        "memory_drift"

    ],

    "trend":[

        "local_mean_30",
        "local_std_30",
        "local_cv_30",
        "momentum_30",
        "acceleration_30"

    ],

    "complexity":[

        "demand_entropy_30",
        "transition_entropy_30"

    ],

    "shock":[

        "absolute_shock",
        "transition_strength"

    ],

    "drfi":[

        "drfi_memory",
        "drfi_stability",
        "drfi_shock",
        "drfi_seasonality"

    ],

    "behaviour":[

        "weekly_similarity",
        "behaviour_persistence",

        "phase_potential",
        "phase_velocity",
        "phase_curvature",

        "phase_confidence",
        "phase_stability",
        "phase_entropy",

        "rare_phase_score"

    ],

    "graph":[

        "degree_centrality",
        "betweenness",
        "closeness",
        "pagerank"

    ]

}

# ==========================================================
# CREATE GENE LIST
# ==========================================================

def get_gene_list():

    genes = []

    for chromosome in GENE_GROUPS.values():

        genes.extend(chromosome)

    return genes

# ==========================================================
# LOAD LATEST GENOME
# ==========================================================

def load_latest_genome(product=PRODUCT):

    print("\n"+"="*70)
    print("LOADING LATEST GENOME")
    print("="*70)

    response = (

        supabase

        .table("final_training_data")

        .select("*")

        .eq("product",product)

        .order("sales_date",desc=True)

        .limit(1)

        .execute()

    )

    df = pd.DataFrame(response.data)

    if df.empty:

        raise ValueError(
            "Latest genome not found."
        )

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    genes = get_gene_list()

    genome = df[genes].copy()

    print()

    print("Latest Date")

    print(df["sales_date"].iloc[0])

    print()

    print("Genome Shape")

    print(genome.shape)

    return df, genome
# ==========================================================
# LOAD GENE EVOLUTION MODELS
# ==========================================================

def load_gene_models():

    print("\n" + "="*70)
    print("LOADING GENE EVOLUTION MODELS")
    print("="*70)

    genes = get_gene_list()

    models = {}

    missing = []

    for gene in genes:

        model_path = os.path.join(

            MODEL_FOLDER,

            f"{gene}.pkl"

        )

        if not os.path.exists(model_path):

            missing.append(gene)

            continue

        models[gene] = joblib.load(model_path)

    if len(missing) > 0:

        print("\nMissing Models")

        print(missing)

        raise FileNotFoundError(
            "Some gene models are missing."
        )

    print()

    print("Loaded Models :", len(models))

    return models


# ==========================================================
# PREDICT NEXT GENOME
# ==========================================================

def predict_next_genome(

        current_genome,

        gene_models

):

    print("\n" + "="*70)
    print("PREDICTING NEXT GENOME")
    print("="*70)

    genes = get_gene_list()

    next_genome = current_genome.copy()

    # --------------------------------------------
    # Predict every gene
    # --------------------------------------------

    for gene in genes:

        model = gene_models[gene]

        prediction = model.predict(

            current_genome[genes]

        )[0]

        next_genome.loc[
            next_genome.index[0],
            gene
        ] = prediction

    print()

    print("Predicted Genome")

    print(next_genome.T)

    return next_genome


# ==========================================================
# EVOLVE GENOME
# ==========================================================

def evolve_one_day(

        genome,

        gene_models

):

    next_genome = predict_next_genome(

        genome,

        gene_models

    )

    return next_genome
# ==========================================================
# LOAD GENOME → PHASE MODEL
# ==========================================================

def load_phase_model():

    print("\n" + "="*70)
    print("LOADING GENOME → PHASE MODEL")
    print("="*70)

    if not os.path.exists(PHASE_MODEL):

        raise FileNotFoundError(
            PHASE_MODEL
        )

    model = joblib.load(PHASE_MODEL)

    print()

    print("Model Loaded Successfully")

    return model
# ==========================================================
# LOAD GENE RELIABILITY
# ==========================================================

def load_gene_reliability(product=PRODUCT):

    response = (

        supabase

        .table("gene_reliability")

        .select("gene,reliability")

        .eq("product", product)

        .execute()

    )

    reliability = pd.DataFrame(response.data)

    reliability = dict(

        zip(

            reliability["gene"],

            reliability["reliability"]

        )

    )

    return reliability
# ==========================================================
# PREDICT PHASE FROM GENOME
# ==========================================================

def predict_phase(

        genome,

        phase_model,

        reliability

):

    print("\n" + "="*70)
    print("PREDICTING NEXT PHASE")
    print("="*70)

    genes = get_gene_list()

    weighted = {}

    for gene in genes:

        weighted[f"weighted_{gene}"] = (

            genome.iloc[0][gene]

            *

            reliability[gene]

        )

    X = pd.DataFrame([weighted])

    # IMPORTANT
    X = X[phase_model.feature_names_]

    phase = phase_model.predict(X)

    phase = int(phase.flatten()[0])

    probabilities = phase_model.predict_proba(X)[0]

    confidence = probabilities.max()

    print()

    print("Predicted Phase")

    print(phase)

    print()

    print("Confidence")

    print(round(confidence,4))

    print()

    print("Probabilities")

    for i,p in enumerate(probabilities):

        print(

            f"Phase {phase_model.classes_[i]} :",

            round(float(p),4)

        )

    return {

        "phase": phase,

        "confidence": float(confidence),

        "probabilities": probabilities

    }
# ==========================================================
# RECURSIVE 30-DAY EVOLUTION
# ==========================================================

def recursive_phase_evolution(

        latest_row,
        genome,
        gene_models,
        phase_model,
        reliability

):

    print("\n" + "="*70)
    print("RECURSIVE GENOME PHASE EVOLUTION")
    print("="*70)

    current_genome = genome.copy()

    current_date = latest_row["sales_date"].iloc[0]

    rows = []

    for day in range(1, FORECAST_DAYS + 1):

        print(f"\nForecast Day {day}")

        current_date += timedelta(days=1)

        # -----------------------------
        # Predict next genome
        # -----------------------------

        current_genome = evolve_one_day(

            current_genome,

            gene_models

        )

        # -----------------------------
        # Predict phase
        # -----------------------------

        phase = predict_phase(

            current_genome,

            phase_model,

            reliability

        )

        row = {

            "forecast_day": day,

            "sales_date": current_date,

            "predicted_phase": phase["phase"],

            "confidence": phase["confidence"]

        }

        probs = phase["probabilities"]

        for i, p in enumerate(probs):

            row[f"P{phase_model.classes_[i]}"] = float(p)

        rows.append(row)

    forecast = pd.DataFrame(rows)

    print()

    print(forecast)

    return forecast
# ==========================================================
# SAVE RECURSIVE FORECAST
# ==========================================================

def save_recursive_forecast(

        forecast

):

    rows = forecast.copy()

    rows["sales_date"] = rows["sales_date"].dt.strftime("%Y-%m-%d")

    supabase.table(

        "recursive_phase_forecast"

    ).upsert(

        rows.to_dict("records"),

        on_conflict="sales_date"

    ).execute()

    print()

    print("Saved :", len(rows))
if __name__ == "__main__":

    latest_row, genome = load_latest_genome()

    gene_models = load_gene_models()

    phase_model = load_phase_model()

    reliability = load_gene_reliability()

    forecast = recursive_phase_evolution(

        latest_row,

        genome,

        gene_models,

        phase_model,

        reliability

    )

    save_recursive_forecast(

        forecast

    )
