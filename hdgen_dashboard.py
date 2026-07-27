from load_supabase import supabase
import pandas as pd

EXPERIMENT_ID = "HDGEN_V1"

def run_hdgen_dashboard(product=None):

    best = (
        supabase
        .table("hdgen_decoder_results")
        .select("*")
        .eq("selected", True)
        .eq("experiment_id", EXPERIMENT_ID)
        .execute()
    ).data[0]

    forecast = (
        supabase
        .table("hdgen_final_demand_forecasts")
        .select("*")
        .eq("experiment_id", EXPERIMENT_ID)
        .order("forecast_date")
        .execute()
    ).data

    df = pd.DataFrame(forecast)

    test = df[df["forecast_type"] == "TEST"]
    future = df[df["forecast_type"] == "FUTURE"]

    return {

        "mae": best["mae"],
        "rmse": best["rmse"],
        "mape": best["mape"],
        "r2": best["r2"],

        "dates": test["forecast_date"].tolist(),
        "actual": test["actual_demand"].tolist(),
        "forecast": test["predicted_demand"].tolist(),

        "future_dates": future["forecast_date"].tolist(),
        "future_forecast": future["predicted_demand"].tolist()

    }