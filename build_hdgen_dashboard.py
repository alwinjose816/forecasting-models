# ============================================================
# BUILD HDGEN DASHBOARD
# ============================================================

from load_supabase import supabase
from load_data import load_dealer_orders

import pandas as pd
import numpy as np

EXPERIMENT_TABLE = "hdgen_genome_experiments"
FORECAST_TABLE = "hdgen_optimizer_forecasts"

BEST_MODEL_TABLE = "hdgen_best_model"
BEST_FORECAST_TABLE = "hdgen_best_forecast"

# ============================================================
# LOAD BEST EXPERIMENT
# ============================================================

def load_best_experiment():

    response = (
        supabase
        .table(EXPERIMENT_TABLE)
        .select("*")
        .eq("is_best", True)
        .limit(1)
        .execute()
    )

    if len(response.data) == 0:

        response = (
            supabase
            .table(EXPERIMENT_TABLE)
            .select("*")
            .order("fitness_score")
            .limit(1)
            .execute()
        )

    return response.data[0]


# ============================================================
# LOAD FORECAST
# ============================================================

def load_forecast(experiment_id):

    response = (
        supabase
        .table(FORECAST_TABLE)
        .select("*")
        .eq("experiment_id", experiment_id)
        .order("forecast_date")
        .execute()
    )

    return pd.DataFrame(response.data)


# ============================================================
# LOAD ACTUAL DEMAND
# ============================================================

def load_actual():

    df = load_dealer_orders()

    demand = (
        df
        .groupby("order_date")["total_weight_mt"]
        .sum()
        .reset_index()
    )

    demand.rename(
        columns={
            "order_date": "forecast_date",
            "total_weight_mt": "actual"
        },
        inplace=True
    )

    demand["forecast_date"] = pd.to_datetime(
        demand["forecast_date"]
    )

    return demand
# ============================================================
# MERGE ACTUAL + FORECAST
# ============================================================

def prepare_forecast(best_exp):

    experiment_id = best_exp["experiment_id"]

    forecast = load_forecast(experiment_id)
    if forecast.empty:

        raise Exception(
            f"No forecast found for experiment {experiment_id}"
        )

    actual = load_actual()

    forecast["forecast_date"] = pd.to_datetime(
        forecast["forecast_date"]
    )

    # rename forecast column if required
    if "forecast" in forecast.columns:

        forecast.rename(
            columns={
                "forecast": "predicted"
            },
            inplace=True
        )

    elif "predicted_demand" in forecast.columns:

        forecast.rename(
            columns={
                "predicted_demand": "predicted"
            },
            inplace=True
        )

    merged = forecast.merge(

        actual,

        on="forecast_date",

        how="left"

    )
    print("\nMerged Forecast Preview")
    print(merged.head())

    print("\nMissing Actual Values:", merged["actual"].isna().sum())
    print("Total Rows:", len(merged))

    merged["error"] = (
        merged["actual"] -
        merged["predicted"]
    )

    merged["absolute_error"] = (
        merged["error"].abs()
    )

    merged["percentage_error"] = np.where(

        merged["actual"] == 0,

        np.nan,

        merged["absolute_error"] /
        merged["actual"] * 100

    )

    return merged
def calculate_metrics(df):

    valid = df.dropna(subset=["actual", "predicted"])

    if len(valid) == 0:

        return {
            "mae": None,
            "rmse": None,
            "mape": None,
            "r2": None
        }

    mae = np.mean(np.abs(valid["actual"] - valid["predicted"]))

    rmse = np.sqrt(
        np.mean((valid["actual"] - valid["predicted"]) ** 2)
    )

    mape = np.mean(
        np.abs(
            (valid["actual"] - valid["predicted"])
            / valid["actual"]
        )
    ) * 100

    ss_res = np.sum(
        (valid["actual"] - valid["predicted"]) ** 2
    )

    ss_tot = np.sum(
        (valid["actual"] - valid["actual"].mean()) ** 2
    )

    if ss_tot == 0:
        r2 = None
    else:
        r2 = 1 - ss_res / ss_tot

    return {
        "mae": None if pd.isna(mae) else float(round(mae, 3)),
        "rmse": None if pd.isna(rmse) else float(round(rmse, 3)),
        "mape": None if pd.isna(mape) else float(round(mape, 3)),
        "r2": None if r2 is None or pd.isna(r2) else float(round(r2, 4))
    }
# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(best_exp, metrics):

    row = {

        "experiment_id":
            best_exp["experiment_id"],

        "generation":
            best_exp["generation"],

        "population": best_exp.get("population", 200),

        "chromosome":
            best_exp["chromosome"],

        "selected_genes":
            best_exp["genes_selected"],

        "selected_gene_names":
            best_exp["selected_gene_names"],

        "best_decoder":
             best_exp["best_model"],

        "fitness":
            best_exp["fitness_score"],

        "mae":
            metrics["mae"],

        "rmse":
            metrics["rmse"],

        "mape":
            metrics["mape"],

        "r2":
            metrics["r2"],

        "forecast_horizon":
            30,

        "training_samples":
            None,

        "testing_samples":
            None

    }

    supabase.table(
        BEST_MODEL_TABLE
    ).upsert(
        row,
        on_conflict="experiment_id"
    ).execute()
    # ============================================================
# SAVE BEST FORECAST
# ============================================================

def save_best_forecast(best_exp, forecast_df):

    experiment_id = best_exp["experiment_id"]

    # Delete old forecast for this experiment
    (
        supabase
        .table(BEST_FORECAST_TABLE)
        .delete()
        .eq("experiment_id", experiment_id)
        .execute()
    )

    rows = []

    for _, row in forecast_df.iterrows():

        rows.append({

            "experiment_id": experiment_id,

            "forecast_date": row["forecast_date"].strftime("%Y-%m-%d"),

            "actual_demand":
                None if pd.isna(row["actual"])
                else float(row["actual"]),

            "predicted_demand":
                float(row["predicted"]),

            "forecast_error":
                None if pd.isna(row["error"])
                else float(row["error"]),

            "absolute_error":
                None if pd.isna(row["absolute_error"])
                else float(row["absolute_error"]),

            "percentage_error":
                None if pd.isna(row["percentage_error"])
                else float(row["percentage_error"])

        })

    if len(rows) > 0:

        (
            supabase
            .table(BEST_FORECAST_TABLE)
            .insert(rows)
            .execute()
        )

    print("Saved", len(rows), "forecast rows.")
# ============================================================
# BUILD DASHBOARD
# ============================================================

def build_dashboard():

    print("=" * 60)
    print("HDGEN DASHBOARD BUILDER")
    print("=" * 60)

    best = load_best_experiment()

    print("Best Experiment :", best["experiment_id"])
    print("Best Decoder    :", best["best_model"])
    print("Generation      :", best["generation"])
    print("Selected Genes  :", best["genes_selected"])

    forecast = prepare_forecast(best)

    if forecast["actual"].notna().sum() > 0:

        metrics = calculate_metrics(forecast)

    else:

        metrics = {

            "mae": best.get("mae"),

            "rmse": best.get("rmse"),

            "mape": best.get("mape"),

            "r2": best.get("r2")

        }

    print("\nMetrics")
    print(metrics)

    save_best_model(best, metrics)

    save_best_forecast(best, forecast)

    print("\nDashboard tables updated successfully.")
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_dashboard()

