import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ==========================================================
# METRICS
# ==========================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# MACHINE LEARNING MODELS
# ==========================================================

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor
)

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

# ==========================================================
# STATISTICAL MODELS
# ==========================================================

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt,
    ExponentialSmoothing
)

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ==========================================================
# SUPABASE
# ==========================================================

from load_supabase import supabase

# ==========================================================
# CONFIGURATION
# ==========================================================

GENE_SELECTION_METHOD = "ALL"

# OPTIONS
# ALL
# GQI
# PARETO
# CORRELATION
# SHAP

TRAIN_RATIO = 0.80

FORECAST_HORIZON = 30

RANDOM_STATE = 42
# ==========================================================
# CHROMOSOMES
# ==========================================================

CHROMOSOMES = {

    "Memory": [

        "memory_short",
        "memory_weekly",
        "memory_biweekly",
        "memory_monthly",
        "memory_state",
        "rolling_memory_strength",
        "memory_half_life",
        "memory_ratio",
        "memory_drift",
        "memory_stability",
        "memory_persistence_rate"

    ],

    "Trend": [

        "local_mean_30",
        "local_std_30",
        "local_cv_30",
        "trend_30",
        "trend_90",
        "trend_gap",
        "momentum_30",
        "acceleration_30",
        "trend_stability",
        "trend_direction_persistence"

    ],

    "Complexity": [

        "demand_entropy_30",
        "demand_entropy_90",
        "skewness_30",
        "kurtosis_30",
        "transition_entropy_30",
        "complexity_drift",
        "complexity_stability"

    ],

    "Shock": [

        "absolute_shock",
        "relative_shock",
        "shock_score",
        "shock_rate_30",
        "transition_direction",
        "transition_volatility",
        "shock_persistence",
        "shock_recovery",
        "shock_cluster_density",
        "shock_energy",
        "shock_decay"

    ],

    "Seasonality": [

        "weekly_similarity",
        "seasonal_strength",
        "seasonal_consistency",
        "seasonal_phase",
        "seasonal_drift"

    ]

}


def get_chromosome(gene):

    for chromosome, genes in CHROMOSOMES.items():

        if gene in genes:

            return chromosome

    return "Unknown"
# ==========================================================
# LOAD HDGEN BEHAVIOUR GENOME
# ==========================================================

def load_behaviour_genome():

    print("\n" + "=" * 70)
    print("LOADING HDGEN BEHAVIOUR GENOME")
    print("=" * 70)

    response = (
        supabase
        .table("hdgen_behaviour_genome")
        .select("*")
        .order("sales_date")
        .execute()
    )

    genome = pd.DataFrame(response.data)

    genome["sales_date"] = pd.to_datetime(
        genome["sales_date"]
    )

    print("Rows :", len(genome))
    print("Columns :", len(genome.columns))

    return genome


# ==========================================================
# LOAD GENE QUALITY
# ==========================================================

def load_gene_quality():

    print("\n" + "=" * 70)
    print("LOADING HDGEN GENE QUALITY")
    print("=" * 70)

    response = (
        supabase
        .table("hdgen_gene_quality")
        .select("*")
        .execute()
    )

    quality = pd.DataFrame(response.data)

    print("Genes :", len(quality))

    return quality
# ==========================================================
# SELECT GENES
# ==========================================================

def select_genes(quality):

    if GENE_SELECTION_METHOD == "ALL":

        genes = quality["gene_name"].tolist()

    elif GENE_SELECTION_METHOD == "GQI":

        genes = quality.loc[
            quality["selected"] == True,
            "gene_name"
        ].tolist()

    elif GENE_SELECTION_METHOD == "PARETO":

        genes = quality.loc[
            quality["selected_pareto"] == True,
            "gene_name"
        ].tolist()

    elif GENE_SELECTION_METHOD == "CORRELATION":

        genes = quality.loc[
            quality["selected_correlation"] == True,
            "gene_name"
        ].tolist()

    elif GENE_SELECTION_METHOD == "SHAP":

        genes = quality.loc[
            quality["selected_shap"] == True,
            "gene_name"
        ].tolist()

    else:

        genes = quality["gene_name"].tolist()

    print()

    print("Genes Selected :", len(genes))

    return genes
# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

def split_series(series):

    lag_df = pd.DataFrame()

    lag_df["y"] = series

    for i in range(1, 8):

        lag_df[f"lag_{i}"] = series.shift(i)

    lag_df = lag_df.dropna()

    split = int(len(lag_df) * TRAIN_RATIO)

    train = lag_df["y"].iloc[:split]

    test = lag_df["y"].iloc[split:]

    return train, test


# ==========================================================
# EVALUATE FORECAST
# ==========================================================

def evaluate_forecast(
        actual,
        prediction
):

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

    mape = np.mean(

        np.abs(

            (actual - prediction)

            /

            (actual + 1e-9)

        )

    ) * 100

    r2 = r2_score(
        actual,
        prediction
    )

    return {

        "mae": mae,

        "rmse": rmse,

        "mape": mape,

        "r2": r2

    }
# ==========================================================
# FUTURE FORECAST HELPERS
# ==========================================================

def future_naive(series):

    return np.repeat(
        series.iloc[-1],
        FORECAST_HORIZON
    )


def future_moving_average(series, window=7):

    history = series.tolist()

    future = []

    for _ in range(FORECAST_HORIZON):

        pred = np.mean(history[-window:])

        future.append(pred)

        history.append(pred)

    return np.array(future)
def create_lag_dataset(series, lags=7, full=False):

    df = pd.DataFrame()

    df["y"] = series

    for i in range(1, lags + 1):

        df[f"lag_{i}"] = series.shift(i)

    df = df.dropna()

    X = df.drop(columns=["y"])

    y = df["y"]

    if full:

        return X, y

    split = int(len(df) * TRAIN_RATIO)

    return (

        X.iloc[:split],
        X.iloc[split:],
        y.iloc[:split],
        y.iloc[split:]

    )
def get_statistical_train_test(series):

    lag_df = pd.DataFrame()

    lag_df["y"] = series

    for i in range(1, 8):
        lag_df[f"lag_{i}"] = series.shift(i)

    lag_df = lag_df.dropna()

    split = int(len(lag_df) * TRAIN_RATIO)

    train = lag_df["y"].iloc[:split]

    test = lag_df["y"].iloc[split:]

    return train, test
# ==========================================================
# NAIVE FORECAST
# ==========================================================

def run_naive(series):

    train, test = get_statistical_train_test(series)

    prediction = np.repeat(
        train.iloc[-1],
        len(test)
    )

    metrics = evaluate_forecast(
        test.values,
        prediction
    )

    future = future_naive(series)

    return {

        "model_name":"Naive",

        **metrics,

        "test_prediction":prediction.tolist(),

        "future_forecast":future.tolist()

    }


# ==========================================================
# MOVING AVERAGE
# ==========================================================

def run_moving_average(
        series,
        window=7
    ):


    train, test = get_statistical_train_test(series)

    history = train.tolist()

    prediction = []

    for actual in test:

        pred = np.mean(
            history[-window:]
        )

        prediction.append(pred)

        history.append(actual)

    prediction = np.array(prediction)
    future = future_moving_average(series, window)

    metrics = evaluate_forecast(

        test.values,

        prediction

    )

    return {

        "model_name":"Moving Average",

        **metrics,

        "test_prediction":prediction.tolist(),

        "future_forecast":future.tolist()

    }


# ==========================================================
# SIMPLE EXPONENTIAL SMOOTHING
# ==========================================================

def run_ses(series):

    train, test = get_statistical_train_test(series)
    try:
        model = SimpleExpSmoothing(

            train

        ).fit()

        prediction = model.forecast(

            len(test)

        )

        metrics = evaluate_forecast(

            test.values,

            prediction.values

        )
        final_model = SimpleExpSmoothing(
            series
        ).fit()

        future = final_model.forecast(
            FORECAST_HORIZON
        )

        return {

            "model_name":"SES",

            **metrics,

            "test_prediction":prediction.values.tolist(),

            "future_forecast":future.tolist()

        }
    except Exception:

        return {

            "model_name": "SES",

            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "r2": np.nan,

            "test_prediction": [],

            "future_forecast": []
        }

# ==========================================================
# HOLT LINEAR TREND
# ==========================================================

def run_holt(series):

    train, test = get_statistical_train_test(series)

    try:

        model = Holt(
            train
        ).fit(
            optimized=True
        )

        prediction = model.forecast(
            len(test)
        )

        metrics = evaluate_forecast(
            test.values,
            prediction.values
        )
        # -------------------------------------
        # Train on FULL history
        # -------------------------------------

        final_model = Holt(
            series
        ).fit(
            optimized=True
        )

        future = final_model.forecast(
            FORECAST_HORIZON
        )

        return {

            "model_name":"Holt",

            **metrics,

            "future_forecast":future.tolist(),
            "test_prediction":prediction.values.tolist()

        }

    except Exception:

        return {

            "model_name": "Holt",

            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "r2": np.nan,
            "test_prediction": [],
            "future_forecast":[]

        }
# ==========================================================
# HOLT WINTERS
# ==========================================================

def run_holt_winters(series):

    train, test = get_statistical_train_test(series)

    try:

        model = ExponentialSmoothing(

            train,

            trend="add",

            seasonal="add",

            seasonal_periods=7

        ).fit(
            optimized=True
        )

        prediction = model.forecast(
            len(test)
        )

        metrics = evaluate_forecast(
            test.values,
            prediction.values
        )
        # -------------------------------------
        # Train on FULL history
        # -------------------------------------

        final_model = ExponentialSmoothing(

            series,

            trend="add",

            seasonal="add",

            seasonal_periods=7

        ).fit(
            optimized=True
        )

        future = final_model.forecast(
            FORECAST_HORIZON
        )
        print(
            f"HW -> test={len(test)}, "
            f"prediction={len(prediction)}, "
            f"future={len(future)}"
        )

        return {

            "model_name":"Holt-Winters",

            **metrics,

            "future_forecast":future.tolist(),
            "test_prediction":prediction.values.tolist()

        }
        

    except Exception:
       

        return {

            "model_name": "Holt-Winters",

            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "r2": np.nan,
            "test_prediction": [],
            "future_forecast":[]

        }
def run_arima(series):

    train, test = get_statistical_train_test(series)

    try:

        model = ARIMA(
            train,
            order=(1,1,1)
        )

        fitted = model.fit()

        prediction = fitted.forecast(
            steps=len(test)
        )

        prediction = np.asarray(prediction)

        metrics = evaluate_forecast(
            test.values,
            prediction
        )
        # -------------------------------------
        # Train on FULL history
        # -------------------------------------

        final_model = ARIMA(

            series,

            order=(1,1,1)

        ).fit()

        future = final_model.forecast(
            steps=FORECAST_HORIZON
        )

        future = np.asarray(future)

        return {

            "model_name":"ARIMA",

            **metrics,

            "future_forecast":future.tolist(),
            "test_prediction":prediction.tolist()

        }

    except Exception as e:

        print("ARIMA Error :", e)

        return {

            "model_name":"ARIMA",

            "mae":np.nan,

            "rmse":np.nan,

            "mape":np.nan,

            "r2":np.nan,
            "test_prediction": [],

            "future_forecast":[]

        }
# ==========================================================
# SARIMA
# ==========================================================

def run_sarima(series):

    train, test = get_statistical_train_test(series)

    try:

        model = SARIMAX(
            train,
            order=(1,1,1),
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit(

            disp=False

        )

        prediction = fitted.forecast(

            steps=len(test)

        )

        prediction = np.asarray(prediction)

        metrics = evaluate_forecast(

            test.values,

            prediction

        )
        # -------------------------------------
        # Train on FULL history
        # -------------------------------------

        final_model = SARIMAX(

            series,

            order=(1,1,1),

            seasonal_order=(1,1,1,7),

            enforce_stationarity=False,

            enforce_invertibility=False

        ).fit(
            disp=False
        )

        future = final_model.forecast(
            steps=FORECAST_HORIZON
        )

        future = np.asarray(future)
        print(
            f"SARIMA -> test={len(test)}, "
            f"prediction={len(prediction)}, "
            f"future={len(future)}"
        )

        return {

            "model_name":"SARIMA",

            **metrics,

            "future_forecast":future.tolist(),
            "test_prediction":prediction.tolist()

        }

    except Exception as e:

        print("SARIMA Error :", e)

        return {

            "model_name":"SARIMA",

            "mae":np.nan,

            "rmse":np.nan,

            "mape":np.nan,

            "r2":np.nan,
            "test_prediction": [],
            "future_forecast":[]

        }

# ==========================================================
# RECURSIVE ML FORECAST
# ==========================================================

def recursive_ml_forecast(
        model,
        series,
        lags=7,
        horizon=FORECAST_HORIZON
):

    history = series.tolist()

    future = []

    for _ in range(horizon):

        columns = [f"lag_{i}" for i in range(1, lags + 1)]

        x = pd.DataFrame(
            [history[-lags:]],
            columns=columns
        )

        pred = model.predict(x)[0]

        future.append(float(pred))

        history.append(pred)

    return future
# ==========================================================
# LINEAR REGRESSION
# ==========================================================

def run_linear(series):

    X_train, X_test, y_train, y_test = create_lag_dataset(series)

    model = LinearRegression()

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    metrics = evaluate_forecast(
        y_test.values,
        prediction
    )

    # Retrain using all available lagged observations
    X_full, y_full = create_lag_dataset(
        series,
        full=True
    )

    model.fit(
        X_full,
        y_full
    )

    future = recursive_ml_forecast(
        model,
        series
    )

    return {

        "model_name":"Linear Regression",

        **metrics,

        "test_prediction":prediction.tolist(),

        "future_forecast":future

    }
# ==========================================================
# RANDOM FOREST
# ==========================================================

def run_random_forest(series):

    X_train, X_test, y_train, y_test = create_lag_dataset(

        series

    )

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

    metrics = evaluate_forecast(

        y_test.values,

        prediction

    )
    X_full, y_full = create_lag_dataset(
        series,
        full=True
    )

    model.fit(
        X_full,
        y_full
    )

    future = recursive_ml_forecast(
        model,
        series
    )

    return {

        "model_name":"Random Forest",

        **metrics,

        "future_forecast":future,
        "test_prediction":prediction.tolist()

    }

# ==========================================================
# XGBOOST
# ==========================================================

def run_xgboost(series):

    X_train, X_test, y_train, y_test = create_lag_dataset(
        series
    )

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

    metrics = evaluate_forecast(

        y_test.values,

        prediction

    )

    # ----------------------------------
    # Retrain on full history
    # ----------------------------------

    X_full, y_full = create_lag_dataset(
        series,
        full=True
    )

    model.fit(

        X_full,

        y_full

    )

    future = recursive_ml_forecast(

        model,

        series

    )

    return {

        "model_name":"XGBoost",

        **metrics,

        "future_forecast":future,
        "test_prediction":prediction.tolist()

    }
# ==========================================================
# LIGHTGBM
# ==========================================================

def run_lightgbm(series):

    X_train, X_test, y_train, y_test = create_lag_dataset(
        series
    )

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

    metrics = evaluate_forecast(

        y_test.values,

        prediction

    )

    # ----------------------------------
    # Retrain on full history
    # ----------------------------------

    X_full, y_full = create_lag_dataset(
        series,
        full=True
    )

    model.fit(

        X_full,

        y_full

    )

    future = recursive_ml_forecast(

        model,

        series

    )

    return {

        "model_name":"LightGBM",

        **metrics,

        "future_forecast":future,
        "test_prediction":prediction.tolist()

    }
# ==========================================================
# CATBOOST
# ==========================================================

def run_catboost(series):

    try:

        X_train, X_test, y_train, y_test = create_lag_dataset(series)

        model = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=42,
            verbose=False
        )

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        metrics = evaluate_forecast(
            y_test.values,
            prediction
        )

        X_full, y_full = create_lag_dataset(
            series,
            full=True
        )

        model.fit(X_full, y_full)

        future = recursive_ml_forecast(
            model,
            series
        )

        return {
            "model_name": "CatBoost",
            **metrics,
            "future_forecast": future,
            "test_prediction": prediction.tolist()
        }

    except Exception as e:

        print("CatBoost Error:", e)

        return {
            "model_name": "CatBoost",
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "r2": np.nan,
            "future_forecast": [],
            "test_prediction": []
        }

# ==========================================================
# RANK MODELS
# ==========================================================

def rank_models(results):

    results = results.copy()

    # Lower is better
    results["rmse_rank"] = results["rmse"].rank(method="min")
    results["mae_rank"] = results["mae"].rank(method="min")
    results["mape_rank"] = results["mape"].rank(method="min")

    # Higher is better
    results["r2_rank"] = results["r2"].rank(
        ascending=False,
        method="min"
    )

    results["score"] = (
        results["rmse_rank"] +
        results["mae_rank"] +
        results["mape_rank"] +
        results["r2_rank"]
    )

    results = results.sort_values("score")

    results["rank"] = range(
        1,
        len(results) + 1
    )

    return results
# ==========================================================
# BEST MODEL
# ==========================================================

def select_best_model(results):

    ranked = rank_models(results)

    return ranked.iloc[0], ranked
# ==========================================================
# RUN ALL MODELS FOR ONE GENE
# ==========================================================

def run_gene_forecasting(

        genome,

        gene_name

):

    print()

    print("=" * 70)

    print("GENE :", gene_name)

    print("=" * 70)

    series = (

        genome

        .sort_values(

            "sales_date"

        )[gene_name]

        .astype(float)

        .reset_index(drop=True)

    )

    results = []

    results.append(run_naive(series))

    results.append(run_moving_average(series))

    results.append(run_ses(series))

    results.append(run_holt(series))

    results.append(run_holt_winters(series))

    results.append(run_arima(series))

    results.append(run_sarima(series))

    results.append(run_linear(series))

    results.append(run_random_forest(series))

    results.append(run_xgboost(series))

    results.append(run_lightgbm(series))

    results.append(run_catboost(series))

    results = pd.DataFrame(results)
    print("\nReturned prediction lengths:")

    for _, row in results.iterrows():

        print(
            row["model_name"],
            len(row["test_prediction"]),
            len(row["future_forecast"])
        )

    best_model, ranked_models = select_best_model(results)

    return best_model, ranked_models
# ==========================================================
# FORECAST WITH BEST MODEL
# ==========================================================

def forecast_best_model(
        model_name,
        series
):

    if model_name == "Naive":

        return future_naive(series)

    elif model_name == "Moving Average":

        return future_moving_average(series)

    elif model_name == "SES":

        model = SimpleExpSmoothing(
            series
        ).fit()

        return model.forecast(
            FORECAST_HORIZON
        ).tolist()

    elif model_name == "Holt":

        model = Holt(
            series
        ).fit(
            optimized=True
        )

        return model.forecast(
            FORECAST_HORIZON
        ).tolist()

    elif model_name == "Holt-Winters":

        model = ExponentialSmoothing(

            series,

            trend="add",

            seasonal="add",

            seasonal_periods=7

        ).fit(
            optimized=True
        )

        return model.forecast(
            FORECAST_HORIZON
        ).tolist()

    elif model_name == "ARIMA":

        model = ARIMA(

            series,

            order=(1,1,1)

        ).fit()

        return model.forecast(
            steps=FORECAST_HORIZON
        ).tolist()

    elif model_name == "SARIMA":

        model = SARIMAX(

            series,

            order=(1,1,1),

            seasonal_order=(1,1,1,7),

            enforce_stationarity=False,

            enforce_invertibility=False

        ).fit(
            disp=False
        )

        return model.forecast(
            steps=FORECAST_HORIZON
        ).tolist()

    elif model_name == "Linear Regression":

        model = LinearRegression()

    elif model_name == "Random Forest":

        model = RandomForestRegressor(
            random_state=RANDOM_STATE
        )

    elif model_name == "XGBoost":

        model = XGBRegressor(
            random_state=RANDOM_STATE
        )

    elif model_name == "LightGBM":

        model = LGBMRegressor(
            random_state=RANDOM_STATE
        )

    elif model_name == "CatBoost":

        model = CatBoostRegressor(
            verbose=False,
            random_state=RANDOM_STATE
        )

    else:

        return []

    if model_name in [

        "Linear Regression",

        "Random Forest",

        "XGBoost",

        "LightGBM",

        "CatBoost"

    ]:

        X_full, y_full = create_lag_dataset(

            series,

            full=True

        )

        model.fit(

            X_full,

            y_full

        )

        return recursive_ml_forecast(

            model,

            series

        )
# ==========================================================
# SAVE MODEL RESULTS
# ==========================================================

def save_hdgen_gene_model_results(results):

    print("\n" + "=" * 70)
    print("SAVING MODEL RESULTS")
    print("=" * 70)

    records = []

    for _, row in results.iterrows():

        records.append({

            "gene_name": row["gene_name"],

            "chromosome": row["chromosome"],

            "model_name": row["model_name"],

            "mae": None if pd.isna(row["mae"]) else float(row["mae"]),

            "rmse": None if pd.isna(row["rmse"]) else float(row["rmse"]),

            "mape": None if pd.isna(row["mape"]) else float(row["mape"]),

            "r2": None if pd.isna(row["r2"]) else float(row["r2"]),

            "training_rows": int(row.get("training_rows", 0)),

            "testing_rows": int(row.get("testing_rows", 0)),

            "feature_count": int(row.get("feature_count", 0)),

            "model_rank": int(row["rank"]),

            "selected": bool(row["rank"] == 1),

            "train_start": None if pd.isna(row.get("train_start")) else str(row["train_start"]),

            "train_end": None if pd.isna(row.get("train_end")) else str(row["train_end"]),

            "test_start": None if pd.isna(row.get("test_start")) else str(row["test_start"]),

            "test_end": None if pd.isna(row.get("test_end")) else str(row["test_end"]),

            "forecast_horizon": FORECAST_HORIZON,
             "experiment_id": "AGFE_V1"

        })

    supabase.table(
        "hdgen_gene_model_results"
    ).upsert(
        records,
        on_conflict="gene_name,model_name,experiment_id"
    ).execute()

    print("Saved :", len(records))
# ==========================================================
# SAVE FINAL GENE FORECASTS
# ==========================================================

def save_hdgen_gene_forecasts(records):

    print("\n" + "=" * 70)
    print("SAVING FINAL GENE FORECASTS")
    print("=" * 70)

    if len(records) == 0:

        print("No records.")

        return

    supabase.table(

        "hdgen_gene_forecasts"

    ).upsert(

        records,

        on_conflict="gene_name,forecast_date,experiment_id"

    ).execute()

    print("Saved :", len(records))
# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("ADAPTIVE GENE FORECASTING ENGINE (AGFE)")
    print("=" * 70)

    # ------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------

    genome = load_behaviour_genome()

    quality = load_gene_quality()

    genes = select_genes(quality)
    print("\n" + "=" * 70)
    print("GENES TO FORECAST")
    print("=" * 70)

    print("Count:", len(genes))

    for g in genes:
        print(g)

    # ------------------------------------------------------
    # FORECAST GENES
    # ------------------------------------------------------

    all_results = []
    forecast_records = []

    for i, gene in enumerate(genes, start=1):

        print(f"\n[{i}/{len(genes)}] Processing : {gene}")

        try:

            best_model, model_results = run_gene_forecasting(
                genome,
                gene
            )
            best_prediction = model_results.loc[
                model_results["rank"] == 1,
                "test_prediction"
            ].iloc[0]
            series = (

                genome

                .sort_values("sales_date")[gene]

                .astype(float)

                .reset_index(drop=True)

            )

            future = forecast_best_model(

                best_model["model_name"],

                series

            )
            # -----------------------------------------
            # Build the same lagged dataset used by ML
            # -----------------------------------------

            lag_df = pd.DataFrame()

            lag_df["sales_date"] = genome["sales_date"]

            lag_df["y"] = series

            for i in range(1, 8):

                lag_df[f"lag_{i}"] = series.shift(i)

            lag_df = lag_df.dropna()

            split = int(len(lag_df) * TRAIN_RATIO)

            test_dates = lag_df["sales_date"].iloc[split:]

            test_values = lag_df["y"].iloc[split:].values

            print(
                f"{gene:30s}",
                f"test_dates={len(test_dates)}",
                f"test_values={len(test_values)}",
                f"best_prediction={len(best_prediction)}"
            )

            n = min(
                len(test_dates),
                len(test_values),
                len(best_prediction)
            )

            for d, actual, pred in zip(
                test_dates[:n],
                test_values[:n],
                best_prediction[:n]
            ):

                forecast_records.append({

                    "forecast_date": str(d.date()),

                    "forecast_day": 0,

                    "data_type": "TEST",

                    "gene_name": gene,

                    "chromosome": get_chromosome(gene),

                    "actual_value": float(actual),

                    "forecast_value": float(pred),

                    "model_name": best_model["model_name"],

                    "forecast_horizon": FORECAST_HORIZON,

                    "mae": float(best_model["mae"]),

                    "rmse": float(best_model["rmse"]),

                    "mape": float(best_model["mape"]),

                    "r2": float(best_model["r2"]),

                    "training_rows": split,
                    "testing_rows": len(lag_df) - split,

                    "feature_count": len(
                        create_lag_dataset(
                            series,
                            full=True
                        )[0].columns
                    ),

                    "selected_method": "Adaptive Ranking",

                    "model_reason": "Lowest Overall Score",

                    "model_rank": int(best_model["rank"]),

                    "selected": True,

                    "experiment_id": "AGFE_V1"

                })
            last_date = genome["sales_date"].max()

            for day, value in enumerate(future, start=1):

                forecast_records.append({

                    "forecast_date": str(

                        last_date + pd.Timedelta(days=day)

                    )[:10],

                    "forecast_day": day,

                    "data_type":"FUTURE",

                    "gene_name":gene,

                    "chromosome":get_chromosome(gene),

                    "actual_value":None,

                    "forecast_value":float(value),

                    "model_name":best_model["model_name"],

                    "forecast_horizon":FORECAST_HORIZON,

                    "mae":float(best_model["mae"]),

                    "rmse":float(best_model["rmse"]),

                    "mape":float(best_model["mape"]),

                    "r2":float(best_model["r2"]),

                    "training_rows": split,
                    "testing_rows": len(lag_df) - split,

                    "feature_count": len(
                        create_lag_dataset(
                            series,
                            full=True
                        )[0].columns
                    ),

                    "selected_method":"Adaptive Ranking",

                    "model_reason":"Lowest Overall Score",

                    "model_rank": int(best_model["rank"]),

                    "selected":True,

                    "experiment_id":"AGFE_V1"

                })

            train_rows = split

            test_rows = len(lag_df) - split
            model_results["train_end"] = lag_df["sales_date"].iloc[split - 1]

            model_results["test_start"] = lag_df["sales_date"].iloc[split]

            model_results["test_end"] = lag_df["sales_date"].iloc[-1]

            model_results["gene_name"] = gene

            model_results["chromosome"] = get_chromosome(gene)

            model_results["training_rows"] = train_rows

            model_results["testing_rows"] = test_rows

            model_results["feature_count"] = len(
                create_lag_dataset(
                    series,
                    full=True
                )[0].columns
            )

            model_results["train_start"] = lag_df["sales_date"].iloc[0]

            model_results["train_end"] = lag_df["sales_date"].iloc[split - 1]

            model_results["test_start"] = lag_df["sales_date"].iloc[split]

            model_results["test_end"] = lag_df["sales_date"].iloc[-1]

            all_results.append(model_results)

            print(
                f"Best Model : {best_model['model_name']} "
                f"(RMSE={best_model['rmse']:.3f})"
            )

       

        except Exception as e:

            print(f"Error in {gene} : {e}")

    # ------------------------------------------------------
    # COMBINE RESULTS
    # ------------------------------------------------------

    if len(all_results) > 0:

        all_results = pd.concat(

            all_results,

            ignore_index=True

        )

        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        print(all_results.head())

        print(all_results.columns.tolist())
        print("\n" + "=" * 70)
        print("GENES IN MODEL RESULTS")
        print("=" * 70)

        print("Unique:", all_results["gene_name"].nunique())

        print(sorted(all_results["gene_name"].unique()))
        save_hdgen_gene_model_results(
           all_results
        )
        save_hdgen_gene_forecasts(
            forecast_records
        )

    else:

        print("\nNo forecasting results generated.")
    