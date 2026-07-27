# ==========================================================
# GENE EVOLUTION ENGINE
# Hierarchical Demand Genome Evolution Network (HDGEN)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import joblib

import numpy as np
import pandas as pd

from load_supabase import supabase

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
   
    r2_score
)

TRAIN_RATIO = 0.80
RANDOM_STATE = 42
# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(product="Overall"):

    print("\n" + "=" * 70)
    print("LOADING DEMAND GENOME")
    print("=" * 70)

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
# DEMAND GENOME
# ==========================================================

GENE_GROUPS = {

    "memory": [

        "memory_short",
        "memory_weekly",
        "memory_biweekly",
        "memory_monthly",
        "rolling_memory_strength",
        "memory_half_life",
        "memory_ratio",
        "memory_drift"

    ],

    "trend": [

        "local_mean_30",
        "local_std_30",
        "local_cv_30",
        "momentum_30",
        "acceleration_30"

    ],

    "complexity": [

        "demand_entropy_30",
        "transition_entropy_30"

    ],

    "shock": [

        "absolute_shock",
        "transition_strength"

    ],

    "drfi": [

        "drfi_memory",
        "drfi_stability",
        "drfi_shock",
        "drfi_seasonality"

    ],

    "behaviour": [

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

    "graph": [

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
# RECONSTRUCT FUTURE GENOME
# ==========================================================

def reconstruct_next_day_genome(df, genes):

    print("\n" + "="*70)
    print("RECONSTRUCTING NEXT-DAY GENOME")
    print("="*70)

    predicted = df[
        ["sales_date", "product"]
    ].copy()

    X = df[genes]      # Prepare once

    for gene in genes:

        print(f"\nPredicting : {gene}")

        model = joblib.load(

            f"models/gene_models/delta_{gene}.pkl"

        )

        delta = model.predict(X)

        predicted[gene] = (

            df[gene]

            +

            delta

        )

    print()

    print(predicted.head())

    print()

    print("Rows :", len(predicted))

    print("Predicted Genes :", len(predicted.columns) - 2)

    return predicted
# ==========================================================
# SAVE RECONSTRUCTED GENOME
# ==========================================================

def save_predicted_genome(predicted):

    print("\n" + "=" * 70)
    print("SAVING RECONSTRUCTED GENOME")
    print("=" * 70)

    rows = predicted.copy()

    rows["sales_date"] = rows["sales_date"].dt.strftime("%Y-%m-%d")

    data = rows.to_dict("records")

    supabase.table(

        "predicted_genome"

    ).upsert(

        data,

        on_conflict="sales_date,product"

    ).execute()

    print()

    print("Saved :", len(rows))
# ==========================================================
# EVALUATE GENOME RECONSTRUCTION
# ==========================================================

def evaluate_reconstruction(
        df,
        predicted,
        genes
):

    print("\n" + "="*70)
    print("GENOME RECONSTRUCTION PERFORMANCE")
    print("="*70)

    results = []

    for gene in genes:

        actual = df[gene].shift(-1).iloc[:-1]

        pred = predicted[
            gene
        ].iloc[:-1]

        mae = mean_absolute_error(
            actual,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                pred
            )
        )

       

        r2 = r2_score(
            actual,
            pred
        )

        results.append({

            "gene": gene,

            "mae": mae,

            "rmse": rmse,

        

            "r2": r2

        })

    results = pd.DataFrame(results)

    print()
    print(results)

    print()
    print("Average Metrics")
    print("----------------")

    print(
        "MAE :",
        round(results["mae"].mean(),4)
    )

    print(
        "RMSE:",
        round(results["rmse"].mean(),4)
    )


    print(
        "R²  :",
        round(results["r2"].mean(),4)
    )

    return results
# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    df = load_dataset()

    genes = get_gene_list()

    print("\n" + "=" * 70)
    print("GENOME SUMMARY")
    print("=" * 70)

    print()

    print("Rows :", len(df))

    print("Genes :", len(genes))

    split = int(len(df) * TRAIN_RATIO)

    test_df = df.iloc[split:].reset_index(drop=True)

    predicted_genome = reconstruct_next_day_genome(
        test_df,
        genes
    )

    metrics = evaluate_reconstruction(
        test_df,
        predicted_genome,
        genes
    )

    predicted_genome["sales_date"] = (

        predicted_genome["sales_date"]

        + pd.Timedelta(days=1)

    )

    save_predicted_genome(

        predicted_genome

    )

 