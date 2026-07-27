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

from catboost import CatBoostRegressor

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

    # ======================================================
    # Memory Chromosome
    # ======================================================

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

    # ======================================================
    # Trend Chromosome
    # ======================================================

    "trend": [

        "local_mean_30",
        "local_std_30",
        "local_cv_30",

        "momentum_30",
        "acceleration_30"

    ],

    # ======================================================
    # Complexity Chromosome
    # ======================================================

    "complexity": [

        "demand_entropy_30",

        "transition_entropy_30"

    ],

    # ======================================================
    # Shock Chromosome
    # ======================================================

    "shock": [

        "absolute_shock",

        "transition_strength"

    ],

    # ======================================================
    # Forecastability Chromosome (DRFI)
    # ======================================================

    "drfi": [

        "drfi_memory",

        "drfi_stability",

        "drfi_shock",

        "drfi_seasonality"

    ],

    # ======================================================
    # Behaviour Chromosome
    # ======================================================

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

    # ======================================================
    # Graph Chromosome
    # ======================================================

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

    print("\n" + "=" * 70)
    print("DEMAND GENOME")
    print("=" * 70)

    print()

    print("Chromosomes :", len(GENE_GROUPS))

    print("Genes       :", len(genes))

    print()

    print(genes)

    return genes
# ==========================================================
# PREPARE GENE TARGETS
# ==========================================================

def prepare_gene_targets(df, genes):

    print("\n" + "=" * 70)
    print("PREPARING GENE EVOLUTION TARGETS")
    print("=" * 70)

    df = df.copy()

    target_columns = []

    for gene in genes:

        target = f"delta_{gene}"

        df[target] = (

            df[gene].shift(-1)

            -

            df[gene]

        )

        target_columns.append(target)

    # Remove only the final row because shift(-1) creates one NaN there
    df = df.iloc[:-1].reset_index(drop=True)

    print()
    print("Rows :", len(df))
    print("Targets :", len(target_columns))

    print("\nMissing values in future targets")
    print(df[target_columns].isna().sum())

    print()

    cols = ["sales_date"] + target_columns[:5]

    print(df[cols].head())

    return df, target_columns
# ==========================================================
# TRAIN GENE EVOLUTION MODELS
# ==========================================================

def train_gene_models(df, genes):

    print("\n" + "=" * 70)
    print("TRAINING GENE EVOLUTION MODELS")
    print("=" * 70)

    os.makedirs("models/gene_models", exist_ok=True)

    feature_columns = [
        g for g in genes
        if df[g].notna().any() and df[g].nunique() > 1
    ]

    results = []

    for gene in genes:

        print(f"\nTraining : {gene}")

        target = f"delta_{gene}"

        train_df = df.dropna(subset=[target]).reset_index(drop=True)
        # ------------------------------------------------------
        # Skip genes with insufficient training data
        # ------------------------------------------------------

        if len(train_df) < 30:

            print(f"\nSkipping {gene}")
            print("Reason : insufficient training samples")
            continue

        split = int(len(train_df) * TRAIN_RATIO)

        train = train_df.iloc[:split]
        test = train_df.iloc[split:]

        X_train = train[feature_columns]
        y_train = train[target]

        X_test = test[feature_columns]
        y_test = test[target]
        if len(y_train) == 0 or len(y_test) == 0:

            print(f"\nSkipping {gene}")
            print("Reason : empty training or testing labels")
            continue

        model = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=False,
            random_seed=RANDOM_STATE
        )
        # Skip if training features contain missing values
        if X_train.isna().sum().sum() > 0:
            print(f"\nSkipping {gene}")
            print("Reason : missing values in training features")
            continue

        # Skip if testing features contain missing values
        if X_test.isna().sum().sum() > 0:
            print(f"\nSkipping {gene}")
            print("Reason : missing values in testing features")
            continue
        

        model.fit(

            X_train,
            y_train

        )

        pred = model.predict(X_test)

        mae = mean_absolute_error(

            y_test,
            pred

        )

        rmse = np.sqrt(

            mean_squared_error(

                y_test,
                pred

            )

        )

        r2 = r2_score(

            y_test,
            pred

        )

        joblib.dump(

            model,

            f"models/gene_models/delta_{gene}.pkl"

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

    # ==========================================================
    # GENE QUALITY
    # ==========================================================

    STABLE_R2 = 0.70
    DYNAMIC_R2 = 0.30

    results["quality"] = np.where(

        results["r2"] >= STABLE_R2,

        "Stable",

        np.where(

            results["r2"] >= DYNAMIC_R2,

            "Dynamic",

            "Chaotic"

        )

    )

    results = results.sort_values(

        "r2",

        ascending=False

    )

    print()
    print(results)

    return results

# ==========================================================
# SAVE GENE CATALOG
# ==========================================================

def save_gene_catalog(results, product="Overall"):

    print("\n" + "=" * 70)
    print("SAVING GENE CATALOG")
    print("=" * 70)

    rows = []

    for _, row in results.iterrows():

        gene = row["gene"]

        chromosome = None

        for group, genes in GENE_GROUPS.items():

            if gene in genes:
                chromosome = group
                break

        rows.append({

            "product": product,

            "gene": gene,

            "chromosome": chromosome,

            "mae": float(row["mae"]),

            "rmse": float(row["rmse"]),

            "r2": float(row["r2"]),

            "quality": row["quality"],

            "predictable": row["quality"] != "Chaotic",

            "model_name": "CatBoost",

            "model_path": f"models/gene_models/delta_{gene}.pkl"

        })

    supabase.table("gene_catalog").upsert(
        rows,
        on_conflict="product,gene"
    ).execute()

    print()
    print("Saved :", len(rows))
# ==========================================================
# SAVE GENE RELIABILITY
# ==========================================================

def save_gene_reliability(
        results,
        product="Overall"
):

    print("\n" + "="*70)
    print("SAVING GENE RELIABILITY")
    print("="*70)

    reliability = results.copy()

    # -------------------------------
    # Negative R² should have zero reliability
    # -------------------------------

    reliability["r2_positive"] = reliability["r2"].clip(lower=0)

    max_r2 = reliability["r2_positive"].max()

    if max_r2 == 0:

        reliability["reliability"] = 0

    else:

        reliability["reliability"] = (

            reliability["r2_positive"]

            / max_r2

        )

    rows = []

    for _, row in reliability.iterrows():

        chromosome = None

        for group, genes in GENE_GROUPS.items():

            if row["gene"] in genes:

                chromosome = group

                break

        rows.append({

            "product": product,

            "gene": row["gene"],

            "chromosome": chromosome,

            "r2": float(row["r2"]),

            "reliability": float(row["reliability"]),

            "quality": row["quality"],

            "model_name": "HDGEN"

        })

    supabase.table(

        "gene_reliability"

    ).upsert(

        rows,

        on_conflict="product,gene"

    ).execute()

    print()

    print("Saved :", len(rows))
if __name__ == "__main__":

    df = load_dataset()

    genes = get_gene_list()

    df, target_columns = prepare_gene_targets(
        df,
        genes
    )

    print("\n" + "=" * 70)
    print("GENOME SUMMARY")
    print("=" * 70)

    print()

    print("Rows :", len(df))
    print("Genes :", len(genes))
    print("Targets :", len(target_columns))

    results = train_gene_models(

        df,
        genes

    )

    print("\n" + "=" * 70)
    print("GENE EVOLUTION SUMMARY")
    print("=" * 70)

    save_gene_catalog(results)
    save_gene_reliability(results)

    print(results)