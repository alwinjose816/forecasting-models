# ==========================================================
# RECURSIVE GENOME EVOLUTION
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

PRODUCT = "Overall"

FORECAST_DAYS = 30

MODEL_FOLDER = "models/gene_models"
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

    print("\n" + "="*70)
    print("LOADING LATEST GENOME")
    print("="*70)

    response = (

        supabase

        .table("final_training_data")

        .select("*")

        .eq("product", product)

        .order("sales_date", desc=True)

        .limit(1)

        .execute()

    )

    df = pd.DataFrame(response.data)

    if df.empty:

        raise ValueError("No data found.")

    df["sales_date"] = pd.to_datetime(df["sales_date"])

    genes = get_gene_list()

    genome = df[genes].copy()

    print()

    print("Latest Date")

    print(df.sales_date.iloc[0])

    print()

    print("Genome Shape")

    print(genome.shape)

    return df, genome
# ==========================================================
# LOAD GENE MODELS
# ==========================================================

def load_gene_models():

    print("\n" + "="*70)
    print("LOADING GENE MODELS")
    print("="*70)

    models = {}

    for gene in get_gene_list():

        path = os.path.join(

            MODEL_FOLDER,

            f"{gene}.pkl"

        )

        models[gene] = joblib.load(path)

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

    genes = get_gene_list()

    next_genome = current_genome.copy()

    X = current_genome[genes]

    for gene in genes:

        model = gene_models[gene]

        prediction = model.predict(X)[0]

        next_genome.at[
            next_genome.index[0],
            gene
        ] = prediction

    return next_genome
# ==========================================================
# RECURSIVE GENOME EVOLUTION
# ==========================================================

def recursive_genome_evolution(

        latest_row,
        genome,
        gene_models

):

    print("\n" + "="*70)
    print("RECURSIVE GENOME EVOLUTION")
    print("="*70)

    current_genome = genome.copy()

    current_date = latest_row["sales_date"].iloc[0]

    forecasts = []

    genes = get_gene_list()

    for day in range(1, FORECAST_DAYS + 1):

        print(f"\nForecast Day {day}")

        current_date = current_date + timedelta(days=1)

        current_genome = predict_next_genome(

            current_genome,

            gene_models

        )

        row = {

            "forecast_day": day,

            "sales_date": current_date,

            "product": PRODUCT

        }

        for gene in genes:

            row[gene] = current_genome.iloc[0][gene]

        forecasts.append(row)

    forecast_df = pd.DataFrame(forecasts)

    print()

    print(forecast_df.head())

    return forecast_df
# ==========================================================
# SAVE FUTURE GENOME
# ==========================================================

def save_future_genome(df):

    print("\n" + "="*70)
    print("SAVING FUTURE GENOME")
    print("="*70)

    rows = df.copy()

    rows["sales_date"] = rows["sales_date"].dt.strftime("%Y-%m-%d")

    supabase.table(

        "future_genome"

    ).upsert(

        rows.to_dict("records"),

        on_conflict="sales_date,product"

    ).execute()

    print()

    print("Saved :", len(rows))
if __name__ == "__main__":

    latest_row, genome = load_latest_genome()

    gene_models = load_gene_models()

    future_genome = recursive_genome_evolution(

        latest_row,

        genome,

        gene_models

    )

    save_future_genome(

        future_genome

    )

    print("\n")
    print("="*70)
    print("30-DAY GENOME FORECAST COMPLETE")
    print("="*70)

    print(future_genome.head())