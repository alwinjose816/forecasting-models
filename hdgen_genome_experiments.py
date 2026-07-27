# ============================================================
# HDGEN V2
# Hierarchical Demand Genome Evolution Network
# Decoder + Gene Optimization
# ============================================================

from load_data import load_dealer_orders
from load_supabase import supabase

import pandas as pd
import numpy as np
import joblib
import random
import copy
import warnings

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from build_hdgen_dashboard import build_dashboard

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

EXPERIMENT_ID = "AGFE_V1"

TRAIN_RATIO = 0.80

FUTURE_DAYS = 30

MINIMUM_GENES = 5

EARLY_STOPPING = 10

# Fitness Weights
FITNESS_WEIGHTS = {

    "mae": 0.30,

    "rmse": 0.30,

    "mape": 0.20,

    "r2": 0.20

}

# ============================================================
# DATABASE TABLES
# ============================================================

GENOME_TABLE = "hdgen_genome_experiments"

GENE_TABLE = "hdgen_genome_genes"



FORECAST_TABLE = "hdgen_best_forecast"

BEHAVIOUR_TABLE = "hdgen_behaviour_genome"

GENE_FORECAST_TABLE = "hdgen_gene_forecasts"
BEST_MODEL_TABLE = "hdgen_best_model"



# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

POPULATION_SIZE = 200

MAX_GENERATIONS = 50

ELITE_PERCENT = 0.10

MUTATION_RATE = 0.15

CROSSOVER_RATE = 0.80

# ============================================================
# GLOBAL VARIABLES
# ============================================================

BEST_MODEL = None

BEST_SCALER = None

BEST_GENES = None

BEST_METRICS = None

BEST_PREDICTION = None

BEST_FITNESS = np.inf
# ==========================================
# BULK INSERT BUFFERS
# ==========================================

GENOME_BUFFER = []
GENE_BUFFER = []

BATCH_SIZE = 500
# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(actual, prediction):

    mae = mean_absolute_error(
        actual,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction
        )
    )

    mask = actual != 0

    if mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (actual[mask] - prediction[mask])
                /
                actual[mask]
            )
        ) * 100

    else:

        mape = np.nan

    r2 = r2_score(
        actual,
        prediction
    )

    return {

        "mae": float(mae),

        "rmse": float(rmse),

        "mape": float(mape),

        "r2": float(r2)

    }


# ============================================================
# FITNESS FUNCTION
# ============================================================

def calculate_fitness(metrics):

    """
    Lower fitness is better.
    """

    return (

        FITNESS_WEIGHTS["mae"] * metrics["mae"]

        +

        FITNESS_WEIGHTS["rmse"] * metrics["rmse"]

        +

        FITNESS_WEIGHTS["mape"] * metrics["mape"]

        -

        FITNESS_WEIGHTS["r2"] * metrics["r2"]

    )

def save_genome(

    generation,
    genome_id,
    genes,
    model_name,
    metrics,
    fitness,
    all_genes

):

    chromosome = build_chromosome(all_genes, genes)

    GENOME_BUFFER.append({

        "experiment_id": EXPERIMENT_ID,

        "generation": generation,

        "genome_id": genome_id,

        "chromosome": chromosome,

        "genes_selected": len(genes),

        "selected_gene_names": list(genes),

        "best_model": model_name,

        "mae": float(metrics["mae"]),

        "rmse": float(metrics["rmse"]),

        "mape": float(metrics["mape"]),

        "r2": float(metrics["r2"]),

        "fitness_score": float(fitness),

        "is_best": False

    })

    if len(GENOME_BUFFER) >= BATCH_SIZE:

        supabase.table(GENOME_TABLE).insert(
            GENOME_BUFFER.copy()
        ).execute()

        GENOME_BUFFER.clear()
def save_gene_details(

    generation,
    genome_id,
    all_genes,
    selected_genes

):

    for gene in all_genes:

        GENE_BUFFER.append({

            "experiment_id": EXPERIMENT_ID,

            "generation": generation,

            "genome_id": genome_id,

            "gene_name": gene,

            "selected": gene in selected_genes

        })

    if len(GENE_BUFFER) >= BATCH_SIZE:

        supabase.table(GENE_TABLE).insert(
            GENE_BUFFER.copy()
        ).execute()

        GENE_BUFFER.clear()
# ============================================================
# GENE UTILITIES
# ============================================================

def get_gene_columns(df):

    ignore = [

        "forecast_date",

        "sales_date",

        "product",

        "demand",

        "id",

        "created_at"

    ]

    genes = [

        col

        for col in df.columns

        if col not in ignore

    ]

    return sorted(genes)


def build_chromosome(all_genes, selected_genes):

    return [

        1 if gene in selected_genes else 0

        for gene in all_genes

    ]


def remove_gene(

    current_genes,

    gene

):

    genes = current_genes.copy()

    genes.remove(gene)

    return genes


# ============================================================
# PRINT UTILITIES
# ============================================================

def print_header(title):

    print()

    print("=" * 80)

    print(title)

    print("=" * 80)


def print_metrics(metrics):

    print(f"MAE  : {metrics['mae']:.4f}")

    print(f"RMSE : {metrics['rmse']:.4f}")

    print(f"MAPE : {metrics['mape']:.4f}")

    print(f"R²   : {metrics['r2']:.4f}")


def print_generation(

    generation,

    genes,

    model,

    fitness

):

    print()

    print("-" * 60)

    print(f"Generation : {generation}")

    print(f"Genes Used : {len(genes)}")

    print(f"Best Model : {model}")

    print(f"Fitness    : {fitness:.4f}")

    print("-" * 60)
# ============================================================
# LOAD GENE FORECASTS
# ============================================================

def load_gene_forecasts():

    response = (
        supabase
        .table(GENE_FORECAST_TABLE)
        .select("*")
        .eq("selected", True)
        .eq("experiment_id", EXPERIMENT_ID)
        .execute()
    )

    df = pd.DataFrame(response.data)

    if df.empty:

        raise Exception("No selected gene forecasts found.")

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    return df


# ============================================================
# LOAD ACTUAL DEMAND
# ============================================================

def load_actual_demand(product="Overall"):

    df = load_dealer_orders()

    if product != "Overall":

        df = df[

            df["product_code"] == product

        ].copy()

    demand = (

        df

        .groupby("order_date")["total_weight_mt"]

        .sum()

        .reset_index()

    )

    demand["order_date"] = pd.to_datetime(

        demand["order_date"]

    )

    demand = (

        demand

        .sort_values("order_date")

        .set_index("order_date")

        .asfreq(

            "D",

            fill_value=0

        )

        .reset_index()

    )

    demand.rename(

        columns={

            "order_date":"forecast_date",

            "total_weight_mt":"demand"

        },

        inplace=True

    )

    return demand


# ============================================================
# LOAD BEHAVIOUR GENOME
# ============================================================

def load_training_genome_dataset():

    response = (

        supabase

        .table(BEHAVIOUR_TABLE)

        .select("*")

        .order("sales_date")

        .execute()

    )

    genome = pd.DataFrame(response.data)

    if genome.empty:

        raise Exception("Behaviour genome is empty.")

    genome["sales_date"] = pd.to_datetime(

        genome["sales_date"]

    )

    genome.rename(

        columns={

            "sales_date":"forecast_date"

        },

        inplace=True

    )

    forecast = load_gene_forecasts()

    test_start = forecast[

        forecast["data_type"]=="TEST"

    ]["forecast_date"].min()

    train = genome[

        genome["forecast_date"] < test_start

    ].copy()

    return train
# ============================================================
# BUILD DECODER DATASET
# ============================================================

def build_decoder_dataset():

    forecasts = load_gene_forecasts()

    test = forecasts[
        forecasts["data_type"] == "TEST"
    ].copy()

    future = forecasts[
        forecasts["data_type"] == "FUTURE"
    ].copy()

    # -----------------------------
    # TEST MATRIX
    # -----------------------------

    test_matrix = (

        test

        .pivot(

            index="forecast_date",

            columns="gene_name",

            values="forecast_value"

        )

        .sort_index()

    )

    # -----------------------------
    # FUTURE MATRIX
    # -----------------------------

    future_matrix = (

        future

        .pivot(

            index="forecast_date",

            columns="gene_name",

            values="forecast_value"

        )

        .sort_index()

    )

    demand = load_actual_demand()

    dataset = (

        test_matrix

        .reset_index()

        .merge(

            demand,

            on="forecast_date",

            how="left"

        )

    )

    dataset = dataset.dropna().reset_index(drop=True)

    expected = len(

        test["gene_name"].unique()

    )

    if test_matrix.shape[1] != expected:

        raise Exception(

            "Missing TEST genes."

        )

    if future_matrix.shape[1] != expected:

        raise Exception(

            "Missing FUTURE genes."

        )

    return (

        dataset,

        future_matrix.reset_index()

    )


# ============================================================
# PREPARE TRAIN / TEST DATA
# ============================================================

def prepare_decoder_data():

    train = load_training_genome_dataset()

    test, future = build_decoder_dataset()

    X_train = train.drop(

        columns=[

            "forecast_date",

            "product",

            "demand",

            "id",

            "created_at"

        ],

        errors="ignore"

    )

    y_train = train["demand"]

    X_test = test.drop(

        columns=[

            "forecast_date",

            "demand"

        ]

    )

    # Preserve original order
    common_features = [

        g

        for g in X_train.columns

        if g in X_test.columns

    ]

    X_train = X_train[

        common_features

    ]

    X_test = X_test[

        common_features

    ]

    y_test = test["demand"]

    return (

        X_train,

        X_test,

        y_train,

        y_test,

        train,

        test,

        future,

        common_features

    )
def run_linear_decoder(

    X_train,
    X_test,
    y_train,
    y_test

):

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test.values,
        prediction
    )

    return {

        "model_name":"Linear Regression",

        **metrics,

        "prediction":prediction,

        "model":model

    }
def run_rf_decoder(

    X_train,
    X_test,
    y_train,
    y_test

):

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test.values,
        prediction
    )

    return {

        "model_name":"Random Forest",

        **metrics,

        "prediction":prediction,

        "model":model

    }
def run_xgb_decoder(

    X_train,
    X_test,
    y_train,
    y_test

):

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )


    model.fit(

        X_train,

        y_train

    )

    prediction = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test.values,
        prediction
    )

    return {

        "model_name":"XGBoost",

        **metrics,

        "prediction":prediction,

        "model":model

    }
def run_lgbm_decoder(

    X_train,
    X_test,
    y_train,
    y_test

):

    model = LGBMRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(

        X_train,

        y_train

    )

    prediction = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test.values,
        prediction
    )

    return {

        "model_name":"LightGBM",

        **metrics,

        "prediction":prediction,

        "model":model

    }
def run_catboost_decoder(

    X_train,
    X_test,
    y_train,
    y_test

):

    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(

        X_train,

        y_train

    )

    prediction = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test.values,
        prediction
    )

    return {

        "model_name":"CatBoost",

        **metrics,

        "prediction":prediction,

        "model":model

    }
def run_decoder_models(

    X_train,
    X_test,
    y_train,
    y_test

):

    results = [

        run_linear_decoder(
            X_train,
            X_test,
            y_train,
            y_test
        ),

        run_rf_decoder(
            X_train,
            X_test,
            y_train,
            y_test
        ),

        run_xgb_decoder(
            X_train,
            X_test,
            y_train,
            y_test
        ),

        run_lgbm_decoder(
            X_train,
            X_test,
            y_train,
            y_test
        ),

        run_catboost_decoder(
            X_train,
            X_test,
            y_train,
            y_test
        )

    ]

    return pd.DataFrame(results)
# ============================================================
# EVALUATE ONE GENE SUBSET
# ============================================================

def evaluate_subset(

    X_train,
    X_test,
    y_train,
    y_test,
    genes

):

    train_subset = X_train[genes].copy()
    test_subset = X_test[genes].copy()

    scaler = StandardScaler()

    train_subset = pd.DataFrame(
        scaler.fit_transform(train_subset),
        columns=genes,
        index=X_train.index
    )

    test_subset = pd.DataFrame(
        scaler.transform(test_subset),
        columns=genes,
        index=X_test.index
    )

    results = run_decoder_models(
        train_subset,
        test_subset,
        y_train,
        y_test
    )

    best, ranked = select_best_decoder(results)

    fitness = calculate_fitness({
        "mae": best["mae"],
        "rmse": best["rmse"],
        "mape": best["mape"],
        "r2": best["r2"]
    })

    return {

        "genes": genes,

        "fitness": fitness,

        "best": best,

        "ranked": ranked,

        "scaler": scaler

    }
def select_best_decoder(results):

    results = results.copy()

    results["score"] = (

        results["mae"].rank()

        +

        results["rmse"].rank()

        +

        results["mape"].rank()

        +

        results["r2"].rank(
            ascending=False
        )

    )

    results = results.sort_values(
        "score"
    )

    results["rank"] = np.arange(
        1,
        len(results)+1
    )

    best = results.iloc[0]

    return best, results
# ============================================================
# CREATE INITIAL POPULATION
# ============================================================

def generate_initial_population(

    all_genes

):

    population = []

    elite = [1] * len(all_genes)

    population.append(elite)

    while len(population) < POPULATION_SIZE:

        chromosome = []

        for _ in all_genes:

            chromosome.append(

                1 if random.random() < 0.70 else 0

            )

        if sum(chromosome) < MINIMUM_GENES:

            ones = random.sample(

                range(len(all_genes)),

                MINIMUM_GENES

            )

            chromosome = [0] * len(all_genes)

            for i in ones:

                chromosome[i] = 1

        if chromosome not in population:

            population.append(chromosome)

    print()

    print("=" * 80)

    print("INITIAL POPULATION CREATED")

    print("=" * 80)

    print("Population :", len(population))

    print("Genes      :", len(all_genes))

    return population

# ============================================================
# CHROMOSOME → GENE LIST
# ============================================================

def chromosome_to_genes(

    chromosome,

    all_genes

):

    genes = []

    for bit, gene in zip(

        chromosome,

        all_genes

    ):

        if bit == 1:

            genes.append(gene)

    return genes
# ============================================================
# EVALUATE ONE CHROMOSOME
# ============================================================

def evaluate_chromosome(

    chromosome,

    all_genes,

    X_train,

    X_test,

    y_train,

    y_test

):

    genes = chromosome_to_genes(

        chromosome,

        all_genes

    )

    if len(genes) < MINIMUM_GENES:

        return None

    result = evaluate_subset(

        X_train,

        X_test,

        y_train,

        y_test,

        genes

    )

    return {

        "chromosome": chromosome,

        "genes": genes,

        "fitness": result["fitness"],

        "best": result["best"],

        "scaler": result["scaler"],

        "genome_id": None

    }
# ============================================================
# EVALUATE POPULATION
# ============================================================

def evaluate_population(

    population,

    all_genes,

    X_train,

    X_test,

    y_train,

    y_test,

    generation

):

    evaluated = []

    genome = 1

    for chromosome in population:

        result = evaluate_chromosome(

            chromosome,

            all_genes,

            X_train,

            X_test,

            y_train,

            y_test

        )

        if result is None:

            continue

        save_genome(

            generation=generation,

            genome_id=f"G{generation}_{genome}",

            genes=result["genes"],

            model_name=result["best"]["model_name"],

            metrics=result["best"],

            fitness=result["fitness"],

            all_genes=all_genes

        )

        save_gene_details(

            generation,

            f"G{generation}_{genome}",

            all_genes,

            result["genes"]

        )

        result["genome_id"] = f"G{generation}_{genome}"

        evaluated.append(result)

        genome += 1

    evaluated = sorted(

        evaluated,

        key=lambda x: x["fitness"]

    )

    return evaluated
# ============================================================
# ELITE SELECTION
# ============================================================

def elite_selection(

    evaluated

):

    elite_size = max(

        2,

        int(

            POPULATION_SIZE * ELITE_PERCENT

        )

    )

    elites = evaluated[:elite_size]

    return elites
# ============================================================
# CROSSOVER
# ============================================================

def crossover(

    parent1,

    parent2

):

    if random.random() > CROSSOVER_RATE:

        return parent1.copy()

    point = random.randint(

        1,

        len(parent1)-2

    )

    child = (

        parent1[:point]

        +

        parent2[point:]

    )

    return child
# ============================================================
# MUTATION
# ============================================================

def mutation(

    chromosome

):

    child = chromosome.copy()

    for i in range(len(child)):

        if random.random() < MUTATION_RATE:

            child[i] = 1 - child[i]

    if sum(child) < MINIMUM_GENES:

        zeros = [

            i

            for i,v in enumerate(child)

            if v == 0

        ]

        random.shuffle(

            zeros

        )

        need = MINIMUM_GENES - sum(child)

        for idx in zeros[:need]:

            child[idx] = 1

    return child
# ============================================================
# CREATE NEXT GENERATION
# ============================================================

def create_next_generation(

    elites

):

    population = [

        e["chromosome"]

        for e in elites

    ]

    while len(population) < POPULATION_SIZE:

        parent1 = random.choice(

            elites

        )["chromosome"]

        parent2 = random.choice(

            elites

        )["chromosome"]

        child = crossover(

            parent1,

            parent2

        )

        child = mutation(

            child

        )

        population.append(

            child

        )

    return population
# ============================================================
# HDGEN EVOLUTION ENGINE
# ============================================================

def run_genome_evolution(

    X_train,

    X_test,

    y_train,

    y_test

):

    print_header(

        "HDGEN GENOME EVOLUTION"

    )

    all_genes = list(

        X_train.columns

    )

    population = generate_initial_population(

        all_genes

    )

    history = []

    best = None

    no_improvement = 0

    for generation in range(

        1,

        MAX_GENERATIONS + 1

    ):

        print()

        print(f"Generation {generation}")

        evaluated = evaluate_population(

            population,

            all_genes,

            X_train,

            X_test,

            y_train,

            y_test,

            generation

        )

        current_best = evaluated[0]

        print(

            "Fitness :",

            round(

                current_best["fitness"],

                4

            )

        )

        print(

            "Genes :",

            len(

                current_best["genes"]

            )

        )

        if (

            best is None

            or

            current_best["fitness"] < best["fitness"]

        ):
            

            best = current_best
            best["generation"] = generation
            best["genome_id"] = current_best["genome_id"]

            no_improvement = 0

        else:

            no_improvement += 1

        history.append({

            "generation": generation,

            "fitness": current_best["fitness"],

            "genes": len(current_best["genes"]),

            "model": current_best["best"]["model_name"]

        })

        if no_improvement >= EARLY_STOPPING:

            print()

            print("Early stopping reached.")

            break

        elites = elite_selection(

            evaluated

        )

        population = create_next_generation(

            elites

        )

    print_header(

        "EVOLUTION COMPLETED"

    )

    print(

        "Best Fitness :", round(best["fitness"],4)

    )

    print(

        "Selected Genes :", len(best["genes"])

    )

    print(

        "Best Decoder :", best["best"]["model_name"]

    )

    return {

        "selected_genes": best["genes"],

        "best_model": best["best"],

        "fitness": best["fitness"],

        "history": history,

        "scaler": best["scaler"],

        "generation": best["generation"],

        "genome_id": best["genome_id"]

    }
# ============================================================
# RETRAIN BEST DECODER USING SELECTED GENES
# ============================================================

def retrain_best_decoder(

    best_name,
    selected_genes

):

    response = (

        supabase

        .table("hdgen_behaviour_genome")

        .select("*")

        .order("sales_date")

        .execute()

    )

    genome = pd.DataFrame(response.data)

    genome.rename(

        columns={

            "sales_date":"forecast_date"

        },

        inplace=True

    )

    # -----------------------------------
    # Train only with selected genes
    # -----------------------------------

    X = genome[selected_genes].copy()

    y = genome["demand"]

    scaler = StandardScaler()

    X = pd.DataFrame(

        scaler.fit_transform(X),

        columns=selected_genes,

        index=X.index

    )

    # -----------------------------------
    # Create same decoder
    # -----------------------------------

    if best_name == "Linear Regression":

        model = LinearRegression()

    elif best_name == "Random Forest":

        model = RandomForestRegressor(

            n_estimators=200,

            max_depth=10,

            random_state=42,

            n_jobs=-1

        )

    elif best_name == "XGBoost":

        model = XGBRegressor(

            n_estimators=100,

            max_depth=3,

            learning_rate=0.1,

            subsample=0.8,

            colsample_bytree=0.8,

            objective="reg:squarederror",

            random_state=42

        )

    elif best_name == "LightGBM":

        model = LGBMRegressor(

            n_estimators=100,

            max_depth=3,

            learning_rate=0.1,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=42

        )

    elif best_name == "CatBoost":

        model = CatBoostRegressor(

            iterations=300,

            depth=6,

            learning_rate=0.05,

            loss_function="RMSE",

            random_seed=42,

            verbose=False

        )

    else:

        raise ValueError(best_name)

    model.fit(

        X,

        y

    )

    return (

        model,

        scaler,

        selected_genes

    )

# ============================================================
# SAVE FINAL FORECAST
# ============================================================

def save_final_forecast(

    test,
    y_test,
    test_prediction,
    future

):

    (
        supabase
        .table(FORECAST_TABLE)
        .delete()
        .eq("experiment_id", EXPERIMENT_ID)
        .execute()
    )

    rows = []

    # ==========================
    # TEST DATA
    # ==========================

    for date, actual, pred in zip(

        test["forecast_date"],
        y_test.values,
        test_prediction

    ):

        error = actual - pred

        rows.append({

            "experiment_id": EXPERIMENT_ID,

            "forecast_type": "TEST",

            "forecast_date": str(date.date()),

            "actual_demand": float(actual),

            "predicted_demand": float(pred),

            "forecast_error": float(error),

            "absolute_error": float(abs(error)),

            "percentage_error": (
                float(abs(error) / actual * 100)
                if actual != 0 else None
            )

        })

    # ==========================
    # FUTURE DATA
    # ==========================

    for _, row in future.iterrows():

        rows.append({

            "experiment_id": EXPERIMENT_ID,

            "forecast_type": "FUTURE",

            "forecast_date": str(row["forecast_date"].date()),

            "actual_demand": None,

            "predicted_demand": float(row["forecast"]),

            "forecast_error": None,

            "absolute_error": None,

            "percentage_error": None

        })

    (
        supabase
        .table(FORECAST_TABLE)
        .insert(rows)
        .execute()
    )

    print(f"Saved {len(rows)} rows.")
# ============================================================
# CLEAN PREVIOUS EXPERIMENT
# ============================================================

def clear_previous_experiment():

    print()
    print("=" * 80)
    print("CLEARING PREVIOUS EXPERIMENT DATA")
    print("=" * 80)

    tables = [

        GENOME_TABLE,
        GENE_TABLE,
       
        FORECAST_TABLE,
        BEST_MODEL_TABLE,
      
        

    ]

    for table in tables:

        try:

            response = (
                supabase
                .table(table)
                .delete()
                .eq("experiment_id", EXPERIMENT_ID)
                .execute()
            )

            print(f"✓ {table}")
            print(response.data)

        except Exception as e:

            print(f"✗ Failed to clear {table}: {e}")

    print("Cleanup completed.\n")

# ============================================================
# MAIN HDGEN PIPELINE
# ============================================================

def run_genome_decoder():
    # Clear previous experiment
    clear_previous_experiment()
    # Flush remaining genome rows
   
    print()
    print("="*80)
    print("HDGEN DECODER")
    print("="*80)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train,
        test,
        future,
        common_features

    )=prepare_decoder_data()

    # =====================================================
    # STEP 1
    # Gene Optimization
    # =====================================================

    optimization = run_genome_evolution(

        X_train,

        X_test,

        y_train,

        y_test

    )
    if GENOME_BUFFER:

        supabase.table(GENOME_TABLE).insert(
            GENOME_BUFFER
        ).execute()

        GENOME_BUFFER.clear()

    # Flush remaining gene rows

    if GENE_BUFFER:

        supabase.table(GENE_TABLE).insert(
            GENE_BUFFER
        ).execute()

        GENE_BUFFER.clear()


    selected_genes = optimization["selected_genes"]

    print()
    print("="*80)
    print("BEST GENE SUBSET")
    print("="*80)

    print(selected_genes)

    print()

    print("Total Selected :", len(selected_genes))

    # =====================================================
    # STEP 2
    # Evaluate Best Gene Subset
    # =====================================================

    subset_result = evaluate_subset(

        X_train,
        X_test,
        y_train,
        y_test,
        selected_genes

    )

    best = subset_result["best"]

    print()
    print("="*80)
    print("BEST DECODER")
    print("="*80)

    print(best["model_name"])

    print()

    print_metrics(best)
    (
    supabase
        .table(GENOME_TABLE)
        .update({"is_best": False})
        .eq("experiment_id", EXPERIMENT_ID)
        .execute()
    )
    (
    supabase
        .table(GENOME_TABLE)
        .update({"is_best": True})
        .eq("experiment_id", EXPERIMENT_ID)
        .eq("generation", optimization["generation"])
        .eq("genome_id", optimization["genome_id"])
        .execute()
    )
    
    

    # =====================================================
    # STEP 3
    # Retrain
    # =====================================================

    final_model, final_scaler, feature_order = retrain_best_decoder(

        best["model_name"],

        selected_genes

    )

    # =====================================================
    # STEP 4
    # Future Dataset
    # =====================================================

    future_input = future.copy()

    missing = set(feature_order) - set(future_input.columns)

    if missing:

        raise ValueError(
            f"Missing features : {missing}"
        )

    future_input = future_input[feature_order]

    future_scaled = pd.DataFrame(

        final_scaler.transform(

            future_input

        ),

        columns=feature_order,

        index=future.index

    )

    future_prediction = final_model.predict(

        future_scaled

    )

    future["forecast"] = future_prediction
    joblib.dump(final_model, "hdgen_best_decoder.pkl")

    joblib.dump(final_scaler, "hdgen_scaler.pkl")
    save_final_forecast(

        test,
        y_test,
        best["prediction"],
        future

    )
    print()
    print("=" * 80)
    print("BUILDING HDGEN DASHBOARD")
    print("=" * 80)

    build_dashboard()

    print()

    print("="*80)

    print("HDGEN COMPLETED")

    print("="*80)

    return {

        "selected_genes": selected_genes,

        "best_model": best["model_name"],

        "forecast": future,

        "history": optimization["history"]

    }
    
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_genome_decoder()