from flask import Flask, jsonify, request
from flask_cors import CORS
from load_supabase import supabase
from collections import Counter
import pandas as pd

from naive import run_naive_forecast
from holt_winters import (
    run_holt_winters_forecast
)
from sarima import run_sarima_forecast

from random_forest import (
    run_random_forest_forecast
)
from xgboost_forecast import (
    run_xgboost_forecast
)
from lightgbm_forecast import (
    run_lightgbm_forecast
)
from lstm_forecast import (
    run_lstm_forecast
)
from sarima import (
    run_sarima_forecast
)
from demand_memory import (
    run_demand_memory
)
from all_product_demand_memory import (
    analyze_all_products
)
from maff_xgboost_model import run_maff_xgboost
from dmsm_forecast import (
    run_dmsm_forecast
)
from dmsm_xgboost import (
    run_dmsm_xgboost
)

from dmsm_xgboost_memorystate import (
    run_dmsm_xgboost as run_dmsm_xgboost_memorystate
)
from dmsm_lightgbm_memorystate import (
    run_dmsm_lightgbm as run_dmsm_lightgbm_memorystate
)

from dmsm_lightgbm import (
    run_dmsm_lightgbm
)
from deep_state_xgboost import (
    run_deep_state_xgboost
)
from deep_state_xgboost_research import (
    run_deep_state_xgboost_research
)
from run_state_aware_xgboost import (
    run_state_aware_xgboost
)
from run_regime_aware_xgboost import (
    run_regime_aware_xgboost
)
from demand_states import (
    run_demand_states

)
from demand_regimes import (
    run_demand_regimes
)
from demand_transitions import (
    run_demand_transitions
)
from deep_demand_states import (
    run_deep_demand_states,   
)
from product_state_embeddings import (
    run_product_embeddings
)
from run_transition_aware_xgboost import (
    run_transition_aware_xgboost
)
from drff_forecast import (
    run_drff_forecast
)
from drff_xgboostv5 import (
    run_drff_xgboostv5
)
from drff_xgboostv1 import (
    run_drff_xgboostv1
)
from xgboost_forecast_DRFIv2 import (
    run_xgboost_forecast_DRFIv2
)

from load_data import load_dealer_orders
from hdgen_dashboard import run_hdgen_dashboard
from prophet_forecast import (
    run_prophet_forecast
)
from catboost_forecast import run_catboost_forecast
app = Flask(__name__)
CORS(app)
@app.route("/test")
def test():
    return "Test route works"
@app.route("/naive")
def naive():

    product = request.args.get("product")

    result = run_naive_forecast(product)

    return jsonify(result)
@app.route("/holt-winters")
def holt_winters():

    product = request.args.get("product")

    result = run_holt_winters_forecast(
        product
    )

    return jsonify(result)
@app.route("/sarima")
def sarima():

    product = request.args.get("product")

    result = run_sarima_forecast(
        product
    )

    return jsonify(result)
@app.route("/prophet")
def prophet():

    product = request.args.get("product")

    result = run_prophet_forecast(
        product
    )

    return jsonify(result)
@app.route("/random-forest")
def random_forest():

    product = request.args.get(
        "product"
    )

    result = (
        run_random_forest_forecast(
            product
        )
    )

    return jsonify(result)
@app.route("/xgboost")
def xgboost():

    product = request.args.get(
        "product"
    )

    result = run_xgboost_forecast(
        product
    )

    return jsonify(result)
@app.route("/lightgbm")
def lightgbm():

    product = request.args.get(
        "product"
    )

    result = run_lightgbm_forecast(
        product
    )

    return jsonify(result)
@app.route("/catboost")
def catboost():

    product = request.args.get(
        "product",
        "Overall"
    )

    result = run_catboost_forecast(product)

    return jsonify(result)
@app.route("/lstm")
def lstm():

    product = request.args.get(
        "product"
    )

    result = run_lstm_forecast(
        product
    )

    return jsonify(result)
@app.route("/compare")
def compare():

    product = request.args.get("product", "Overall")

    result = {}

    try:
        result["Naive"] = run_naive_forecast(product)
        print("Naive OK")
    except Exception as e:
        print("Naive:", e)

    try:
        result["HoltWinters"] = run_holt_winters_forecast(product)
        print("HW OK")
    except Exception as e:
        print("HW:", e)

    try:
        result["SARIMA"] = run_sarima_forecast(product)
        print("SARIMA OK")
    except Exception as e:
        print("SARIMA:", e)

    try:
        result["RandomForest"] = run_random_forest_forecast(product)
        print("RF OK")
    except Exception as e:
        print("RF:", e)

    try:
        result["XGBoost"] = run_xgboost_forecast(product)
        print("XGB OK")
    except Exception as e:
        print("XGB:", e)

    try:
        result["CatBoost"] = run_catboost_forecast(product)
        print("CatBoost OK")
    except Exception as e:
        print("CatBoost:", e)

    try:
        result["LightGBM"] = run_lightgbm_forecast(product)
        print("LGBM OK")
    except Exception as e:
        print("LGBM:", e)
    try:
        result["HDGEN"] = run_hdgen_dashboard(product)
        print("HDGEN OK")
    except Exception as e:
        print("HDGEN:", e)
    try:
        result["Prophet"] = run_prophet_forecast(product)
        print("Prophet OK")
    except Exception as e:
        print("Prophet:", e)

    return jsonify(result)
@app.route("/maff-xgboost")
def maff_xgboost_model():

    product = request.args.get(
        "product",
        "Overall"
    )

    result = run_maff_xgboost(
        product
    )

    return jsonify(result)
@app.route(
    "/all-memory-profiles"
)
def all_memory_profiles():

    table = analyze_all_products()

    return table.to_dict(
        orient="records"
    )
@app.route("/dmsm-forecast")
def dmsm_forecast():

    product = request.args.get(
        "product",
        "Overall"
    )

    result = run_dmsm_forecast(
        product
    )

    return jsonify(result)
@app.route("/dmsm-xgboost")
def dmsm_xgboost():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_dmsm_xgboost(product)
    )
@app.route(
    "/dmsm-xgboost-memorystate"
)
def dmsm_xgboost_memorystate():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_dmsm_xgboost_memorystate(product)
    )
@app.route("/xgboost-comparison")
def xgboost_comparison():

    product = request.args.get(
        "product",
        "Overall"
    )

    baseline = run_xgboost_forecast(
        product
    )

    dmsm = run_dmsm_xgboost_memorystate(
        product
    )

    dmsm_residual = (
        run_dmsm_xgboost(
            product
        )
    )

    return jsonify({

        "baseline": {
            "mae": baseline["mae"],
            "rmse": baseline["rmse"],
            "mape": baseline["mape"]
        },

        "dmsm": {
            "mae": dmsm["mae"],
            "rmse": dmsm["rmse"],
            "mape": dmsm["mape"]
        },

        "dmsm_residual": {
            "mae": dmsm_residual["mae"],
            "rmse": dmsm_residual["rmse"],
            "mape": dmsm_residual["mape"]
        }

    })
@app.route("/dmsm-lightgbm")
def dmsm_lightgbm():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_dmsm_lightgbm(product)
    )


@app.route("/dmsm-lightgbm-memorystate")
def dmsm_lightgbm_memorystate():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_dmsm_lightgbm_memorystate(product)
    )
@app.route("/lightgbm-comparison")
def lightgbm_comparison():

    product = request.args.get(
        "product",
        "Overall"
    )

    baseline = run_lightgbm_forecast(
        product
    )

    dmsm = run_dmsm_lightgbm_memorystate(
        product
    )

    dmsm_residual = (
        run_dmsm_lightgbm(
            product
        )
    )

    return jsonify({

        "baseline": {
            "mae": baseline["mae"],
            "rmse": baseline["rmse"],
            "mape": baseline["mape"]
        },

        "dmsm": {
            "mae": dmsm["mae"],
            "rmse": dmsm["rmse"],
            "mape": dmsm["mape"]
        },

        "dmsm_residual": {
            "mae": dmsm_residual["mae"],
            "rmse": dmsm_residual["rmse"],
            "mape": dmsm_residual["mape"]
        }

    })
@app.route("/deep-state-xgboost")
def deep_state_xgboost():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_deep_state_xgboost(product)
    )
@app.route("/demand-regimes")
def demand_regimes():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_demand_regimes(product)
    )
@app.route("/deep-state-xgboost-research")
def deep_state_xgboost_research():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_deep_state_xgboost_research(
            product
        )
    )
@app.route("/state-aware-xgboost")
def state_aware_xgboost():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_state_aware_xgboost(
            product
        )
    )
@app.route("/regime-aware-xgboost")
def regime_aware_xgboost():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_regime_aware_xgboost(
            product
        )
    )
@app.route("/demand-states")
def demand_states():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_demand_states(product)
    )
@app.route("/products")
def products():

    df = load_dealer_orders()

    products = sorted(
        df["product_code"]
        .dropna()
        .unique()
        .tolist()
    )
    return jsonify(products)
@app.route("/demand-transitions")
def demand_transitions():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_demand_transitions(product)
    )
@app.route("/deep-demand-states")
def deep_demand_states():

    product = request.args.get(
        "product",
        "Overall"
    )

    latent_dim = int(
        request.args.get(
            "latent_dim",
            8
        )
    )

    result = run_deep_demand_states(
        product=product,
        latent_dim=latent_dim
    )

    return jsonify(result)
@app.route("/product-embeddings")
def product_embeddings():

    return jsonify(
        run_product_embeddings()
    )
@app.route(
    "/transition-aware-xgboost"
)
def transition_aware_xgboost():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_transition_aware_xgboost(
            product
        )
    )
@app.route("/drff")
def drff():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_drff_forecast(product)
    )
@app.route("/drff-xgboost-v5")
def drff_xgboost_v5():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_drff_xgboostv5(
            product
        )
    )
@app.route(
    "/drff-xgboost-v1"
)
def drff_xgboost_v1():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_drff_xgboostv1(
            product
        )
    )

@app.route("/drfi-validation")
def drfi_validation():

    product = request.args.get(
        "product",
        "Overall"
    )

    return jsonify(
        run_xgboost_forecast_DRFIv2(
            product
        )
    )
@app.route("/api/hdgen")
def get_hdgen():

    data = (
        supabase
        .table("hdgen_behaviour_genome")
        .select("*")
        .order("sales_date")
        .execute()
    )

    return jsonify(data.data)
print(app.url_map)
@app.route("/api/agfe/summary")
def agfe_summary():

    return jsonify({

        "genes": 44,

        "models": 12,

        "forecast_horizon": 30,

        "selection": "ALL"

    })
from collections import Counter

@app.route("/api/agfe/model_distribution")
def agfe_model_distribution():

    response = (
        supabase
        .table("hdgen_gene_model_results")
        .select("model_name")
        .eq("selected", True)
        .execute()
    )

    counts = Counter(
        row["model_name"]
        for row in response.data
    )

    result = []

    for model, count in counts.items():

        result.append({

            "model": model,

            "count": count

        })

    return jsonify(result)
@app.route("/api/agfe/model_results")
def agfe_model_results():

    response = (
        supabase
        .table("hdgen_gene_model_results")
        .select("*")
        .order("gene_name")
        .execute()
    )

    return jsonify(response.data)
@app.route("/api/agfe/best_models")
def agfe_best_models():

    response = (
        supabase
        .table("hdgen_gene_model_results")
        .select("*")
        .eq("selected", True)
        .order("gene_name")
        .execute()
    )

    return jsonify(response.data)
@app.route("/api/agfe/gene_forecast/<gene>")
def agfe_gene_forecast(gene):

    response = (
        supabase
        .table("hdgen_gene_forecasts")
        .select("*")
        .eq("gene_name", gene)
        .order("forecast_date")
        .execute()
    )

    return jsonify(response.data)
@app.route("/api/agfe/performance")
def agfe_performance():

    response = (
        supabase
        .table("hdgen_gene_model_results")
        .select("model_name,mae,rmse,mape,r2")
        .execute()
    )

    import pandas as pd

    df = pd.DataFrame(response.data)

    summary = (
        df.groupby("model_name")
        .agg({
            "mae":"mean",
            "rmse":"mean",
            "mape":"mean",
            "r2":"mean"
        })
        .reset_index()
    )

    return jsonify(summary.to_dict("records"))

@app.route("/api/agfe/genes")
def agfe_genes():

    response = (
        supabase
        .table("hdgen_gene_forecasts")
        .select("gene_name")
        .execute()
    )

    genes = sorted(
        list(
            set(
                row["gene_name"]
                for row in response.data
            )
        )
    )

    return jsonify(genes)

@app.route("/api/decoder/comparison")
def decoder_comparison():

    response = (
        supabase
        .table("hdgen_decoder_results")
        .select("model_name,rmse,mae,mape,r2")
        .order("model_rank")
        .execute()
    )

    return jsonify(response.data)
@app.route("/api/decoder/forecast")
def decoder_forecast():

    response = (
        supabase
        .table("hdgen_final_demand_forecasts")
        .select("*")
        .order("forecast_date")
        .execute()
    )

    return jsonify(response.data)

 
@app.route("/api/decoder/chromosomes")
def decoder_chromosomes():

    response = (
        supabase
        .table("hdgen_gene_model_results")
        .select("chromosome")
        .eq("selected", True)
        .execute()
    )

    from collections import Counter

    counts = Counter(
        row["chromosome"]
        for row in response.data
    )

    result = [

        {
            "chromosome":k,
            "count":v
        }

        for k,v in counts.items()

    ]

    return jsonify(result) 
@app.route("/api/decoder/summary")
def decoder_summary():

    response = (
        supabase
        .table("hdgen_decoder_results")
        .select("*")
        .order("model_rank")
        .limit(1)
        .execute()
    )

    if len(response.data) == 0:
        return jsonify({})

    best = response.data[0]

    return jsonify({

        "best_model": best["model_name"],
        "rmse": round(best["rmse"],3),
        "mae": round(best["mae"],3),
        "mape": round(best["mape"],3),
        "r2": round(best["r2"],4),
        "forecast_horizon": 30

    })
@app.route("/api/decoder/ranking")
def decoder_ranking():

    response = (
        supabase
        .table("hdgen_decoder_results")
        .select(
            "model_rank,model_name,rmse,mae,mape,r2"
        )
        .order("model_rank")
        .execute()
    )

    return jsonify(response.data)
@app.route("/api/decoder/errors")
def decoder_errors():

    response = (
        supabase
        .table("hdgen_final_demand_forecasts")
        .select(
            "forecast_date,actual_demand,predicted_demand"
        )
        .eq("forecast_type", "TEST")
        .order("forecast_date")
        .execute()
    )

    data = response.data

    for row in data:

        row["error"] = (
            row["actual_demand"]
            - row["predicted_demand"]
        )

    return jsonify(data)
@app.route("/api/decoder/monthly")
def decoder_monthly():

    response = (
        supabase
        .table("hdgen_final_demand_forecasts")
        .select(
            "forecast_date,actual_demand,predicted_demand"
        )
        .eq("forecast_type", "TEST")
        .order("forecast_date")
        .execute()
    )

    df = pd.DataFrame(response.data)

    if len(df) == 0:
        return jsonify([])

    df["forecast_date"] = pd.to_datetime(df["forecast_date"])

    df["Month"] = df["forecast_date"].dt.strftime("%Y-%m")

    monthly = (

        df.groupby("Month")[

            ["actual_demand","predicted_demand"]

        ]

        .sum()

        .reset_index()

    )

    return jsonify(

        monthly.to_dict(

            orient="records"

        )

    )
@app.route("/api/decoder/statistics")
def decoder_statistics():

    result = (
        supabase
        .table("hdgen_decoder_results")
        .select("*")
        .order("model_rank")
        .limit(1)
        .execute()
    )

    if len(result.data) == 0:
        return jsonify({})

    best = result.data[0]

    forecast = (
        supabase
        .table("hdgen_final_demand_forecasts")
        .select("*")
        .execute()
    )

    df = pd.DataFrame(forecast.data)

    train_rows = int(best.get("training_rows", 0))
    test_rows = int(best.get("testing_rows", 0))

    return jsonify({

        "best_model": best["model_name"],

        "rmse": round(best["rmse"],3),

        "mae": round(best["mae"],3),

        "mape": round(best["mape"],2),

        "r2": round(best["r2"],4),

        "forecast_horizon": len(
            df[df.forecast_type=="FUTURE"]
        ),

        "training_rows": train_rows,

        "testing_rows": test_rows

    })
@app.route("/api/hdgen/dashboard")
def hdgen_dashboard():

    result = (
        supabase
        .table("hdgen_genome_experiments")
        .select("*")
        .eq("is_best", True)
        .limit(1)
        .execute()
    )

    if len(result.data) == 0:
        return jsonify({})

    row = result.data[0]

    return jsonify({

        "best_fitness": row["fitness_score"],
        "best_decoder": row["best_model"],
        "selected_genes": row["genes_selected"],
        "generation": row["generation"],
        "genome_id": row["genome_id"],   # <-- Add this line
        "population": 200,
        "experiment_id": row["experiment_id"]

    })
@app.route("/api/hdgen/fitness")
def hdgen_fitness():

    result = (
        supabase
        .table("hdgen_genome_experiments")
        .select(
            "generation,genome_id,best_model,genes_selected,fitness_score"
        )
        .order("generation")
        .order("genome_id")
        .execute()
    )

    data = []

    for row in result.data:

        data.append({

            "generation": row["generation"],
            "genome_id": row["genome_id"],
            "decoder": row["best_model"],
            "selected_genes": row["genes_selected"],
            "fitness": row["fitness_score"]

        })

    return jsonify(data)
@app.route("/api/hdgen/decoder")
def hdgen_decoder():

    result = (
        supabase
        .table("hdgen_genome_experiments")
        .select("best_model,fitness_score")
        .eq("is_best", True)
        .limit(1)
        .execute()
    )

    if len(result.data) == 0:
        return jsonify([])

    row = result.data[0]

    return jsonify([{

        "model": row["best_model"],
        "fitness": row["fitness_score"],
        "best": True

    }])
@app.route("/api/hdgen/forecast")
def hdgen_forecast():

    result = (
        supabase
        .table("hdgen_best_forecast")
        .select("*")
        .order("forecast_date")
        .execute()
    )

    data = []

    for row in result.data:

        data.append({

            "forecast_date": row["forecast_date"],
            "actual": row["actual_demand"],
            "forecast": row["predicted_demand"]

        })

    return jsonify(data)
@app.route("/api/hdgen/gene-importance")
def hdgen_gene_importance():

    result = (
        supabase
        .table("hdgen_genome_genes")
        .select("*")
        .eq("selected", True)
        .execute()
    )

    counts = Counter(
        row["gene_name"]
        for row in result.data
    )

    output = []

    for gene, count in counts.items():

        output.append({

            "gene": gene,
            "importance": count

        })

    output.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return jsonify(output)
@app.route("/api/hdgen/best-genome")
def hdgen_best_genome():

    result = (
        supabase
        .table("hdgen_genome_experiments")
        .select("*")
        .eq("is_best", True)
        .limit(1)
        .execute()
    )

    if len(result.data) == 0:
        return jsonify({})

    row = result.data[0]

    return jsonify({

        "experiment_id": row["experiment_id"],
        "generation": row["generation"],
        "decoder": row["best_model"],
        "fitness": row["fitness_score"],
        "selected_genes": row["genes_selected"],
        "chromosome": row["chromosome"],
        "mae": row["mae"],
        "rmse": row["rmse"],
        "mape": row["mape"],
        "r2": row["r2"]

    })
@app.route("/api/hdgen/selected-genes")
def hdgen_selected_genes():

    # Get best genome
    best = (
        supabase
        .table("hdgen_genome_experiments")
        .select("genome_id")
        .eq("is_best", True)
        .limit(1)
        .execute()
    )

    if len(best.data) == 0:
        return jsonify([])

    genome_id = best.data[0]["genome_id"]

    # Get genes for that genome
    result = (
        supabase
        .table("hdgen_genome_genes")
        .select("gene_name,selected")
        .eq("genome_id", genome_id)
        .order("id")
        .execute()
    )

    return jsonify(result.data)
@app.route("/api/hdgen/genome-history")
def hdgen_genome_history():

    result = (
        supabase
        .table("hdgen_genome_experiments")
        .select("*")
        .order("generation")
        .order("genome_id")
        .execute()
    )

    history = []

    for row in result.data:

        history.append({

            "generation": row["generation"],
            "genome_id": row["genome_id"],
            "fitness": row["fitness_score"],
            "decoder": row["best_model"],
            "chromosome": row["chromosome"]

        })

    return jsonify(history)
@app.route("/api/hdgen/experiment-history")
def hdgen_experiment_history():

    result = (
        supabase
        .table("hdgen_genome_experiments")
        .select(
            "experiment_id,generation,genome_id,best_model,genes_selected,fitness_score,mae,rmse,mape,r2"
        )
        .order("generation")
        .order("genome_id")
        .limit(5800)      # or 10000
        .execute()
    )

    return jsonify(result.data)
@app.route("/api/hdgen/metrics")
def hdgen_metrics():

    result = (
        supabase
        .table("hdgen_genome_experiments")
        .select("*")
        .eq("is_best", True)
        .limit(1)
        .execute()
    )

    if len(result.data) == 0:
        return jsonify({})

    row = result.data[0]

    return jsonify({

        "decoder": row["best_model"],
        "mae": row["mae"],
        "rmse": row["rmse"],
        "mape": row["mape"],
        "r2": row["r2"]

    })
@app.route("/api/testforecast")
def testforecast():

    result = (
        supabase
        .table("hdgen_optimizer_forecasts")
        .select("*")
        .limit(1)
        .execute()
    )

    return jsonify(result.data)
if __name__ == "__main__":
    app.run(debug=True)