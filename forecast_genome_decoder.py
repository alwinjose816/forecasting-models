from load_data import load_dealer_orders

from load_supabase import supabase

from supabase import create_client

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")
EXPERIMENT_ID = "HDGEN_V1"

TRAIN_RATIO = 0.80

FUTURE_DAYS = 30
def calculate_metrics(actual, prediction):

    mae = mean_absolute_error(actual, prediction)

    rmse = np.sqrt(
        mean_squared_error(actual, prediction)
    )

    mask = actual != 0

    if mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (actual[mask] - prediction[mask])
                / actual[mask]
            )
        ) * 100

    else:

        mape = np.nan

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
def load_gene_forecasts():

    response = (
        supabase
        .table("hdgen_gene_forecasts")
        .select("*")
        .eq("selected", True)
        .execute()
    )

    df = pd.DataFrame(response.data)

    return df
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

    )

    demand = (

        demand

        .set_index("order_date")

        .asfreq("D", fill_value=0)

        .reset_index()

    )

    demand.rename(

        columns={

            "order_date": "forecast_date",

            "total_weight_mt": "demand"

        },

        inplace=True

    )

    return demand
def load_training_genome_dataset():

    response = (

        supabase

        .table("hdgen_behaviour_genome")

        .select("*")

        .order("sales_date")

        .execute()

    )

    df = pd.DataFrame(response.data)

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    df = df.rename(

        columns={

            "sales_date":"forecast_date"

        }

    )

    forecast = load_gene_forecasts()

    forecast["forecast_date"] = pd.to_datetime(
        forecast["forecast_date"]
    )

    test_start = forecast[
        forecast["data_type"] == "TEST"
    ]["forecast_date"].min()

    train = df[
        df["forecast_date"] < test_start
    ].copy()

    return train
def build_decoder_dataset():

    forecasts = load_gene_forecasts()
    forecasts["forecast_date"] = pd.to_datetime(
        forecasts["forecast_date"]
    )
  

  

    test = forecasts[
        forecasts["data_type"] == "TEST"
    ].copy()

    future = forecasts[
        forecasts["data_type"] == "FUTURE"
    ].copy()

    test_matrix = test.pivot(

        index="forecast_date",

        columns="gene_name",

        values="forecast_value"

    )
    test_matrix = test_matrix.sort_index()

    future_matrix = future.pivot(

        index="forecast_date",

        columns="gene_name",

        values="forecast_value"

    )
    future_matrix = future_matrix.sort_index()

    demand = load_actual_demand()
    demand["forecast_date"] = pd.to_datetime(
        demand["forecast_date"]
    )

    dataset = (
        test_matrix
        .reset_index()
        .merge(
            demand,
            on="forecast_date",
            how="left"
        )
    )

  

    # Remove incomplete rows
    dataset = dataset.dropna().reset_index(drop=True)

    # Validate genome completeness
    print("\nDecoder Dataset")
    print("TEST Matrix   :", test_matrix.shape)
    print("FUTURE Matrix :", future_matrix.shape)

    expected_genes = len(test["gene_name"].unique())

    assert test_matrix.shape[1] == expected_genes, \
        "TEST matrix has missing genes."

    assert future_matrix.shape[1] == expected_genes, \
        "FUTURE matrix has missing genes."
    print(dataset.head())

    print(dataset.shape)

    print(dataset.isna().sum())
  

    return dataset, future_matrix.reset_index()

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
    # Keep only common genes
    common_features = sorted(
        list(set(X_train.columns) & set(X_test.columns))
    )

    X_train = X_train[common_features]
    X_test = X_test[common_features]

    assert list(X_train.columns) == list(X_test.columns)

    y_test = test["demand"]

    scaler = StandardScaler()

    X_train = pd.DataFrame(

        scaler.fit_transform(X_train),

        columns=X_train.columns

    )

    X_test = pd.DataFrame(

        scaler.transform(X_test),

        columns=X_train.columns

    )

    return (

        X_train,

        X_test,

        y_train,

        y_test,

        train,

        test,

        future,

        scaler

    )
def retrain_best_decoder(best_name):

  

    # Load entire behaviour genome
    response = (
        supabase
        .table("hdgen_behaviour_genome")
        .select("*")
        .order("sales_date")
        .execute()
    )

    genome = pd.DataFrame(response.data)

    genome = genome.rename(
        columns={
            "sales_date": "forecast_date"
        }
    )

    X = genome.drop(
        columns=[
            "forecast_date",
            "product",
            "demand",
            "id",
            "created_at"
        ],
        errors="ignore"
    )

    y = genome["demand"]

    # Keep same genes used by decoder
    feature_order = sorted(list(X.columns))
    X = X[feature_order]

    scaler = StandardScaler()

    X = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_order
    )

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

    model.fit(X, y)

    return model, scaler, feature_order
def save_decoder_results(

    ranked_models,

    train,

    test,

    feature_count

):

    records = []

    for _, row in ranked_models.iterrows():

        records.append({

            "model_name":row["model_name"],

            "mae":float(row["mae"]),

            "rmse":float(row["rmse"]),

            "mape":float(row["mape"]),

            "r2":float(row["r2"]),

            "training_rows":len(train),

            "testing_rows":len(test),

            "feature_count":feature_count,

            "model_rank":int(row["rank"]),

            "selected":bool(row["rank"]==1),

            "experiment_id":EXPERIMENT_ID,

            "train_start": str(train["forecast_date"].min().date()),

            "train_end": str(train["forecast_date"].max().date()),

            "test_start": str(test["forecast_date"].min().date()),

            "test_end": str(test["forecast_date"].max().date()),

        })

    (
        supabase
        .table("hdgen_decoder_results")
        .upsert(

            records,

            on_conflict="model_name,experiment_id"

        )
        .execute()
    )

    print()

    print("Decoder Results Saved :",len(records))

def forecast_future_demand(
    model,
    scaler,
    future_matrix,
    feature_order
):

    future_dates = future_matrix["forecast_date"]

    X_future = future_matrix.drop(
        columns=["forecast_date"]
    )

    X_future = X_future.reindex(
        columns=feature_order
    )
    missing = set(feature_order) - set(X_future.columns)

    if missing:
        raise ValueError(f"Missing future genes: {missing}")

    X_future = X_future.reindex(columns=feature_order)

    X_future = pd.DataFrame(

        scaler.transform(X_future),

        columns=X_future.columns

    )

    prediction = model.predict(
        X_future
    )

    return pd.DataFrame({

        "forecast_date":future_dates,

        "predicted_demand":prediction

    })
def save_final_forecast(

    test,

    y_test,

    test_prediction,

    future,

    model_name,

    metrics

):

    records = []

    # ---------------- TEST ----------------

    for d,a,p in zip(

        test["forecast_date"],

        y_test,

        test_prediction

    ):

        records.append({

            "forecast_date": d.date().isoformat(),

            "forecast_type":"TEST",

            "actual_demand": float(a),

            "predicted_demand": float(p),

            "model_name": model_name,

            "mae": float(metrics["mae"]),

            "rmse": float(metrics["rmse"]),

            "mape": float(metrics["mape"]),

            "r2": float(metrics["r2"]),

            "experiment_id": EXPERIMENT_ID

        })

    # ---------------- FUTURE ----------------

    for _,row in future.iterrows():

        records.append({

            "forecast_date": row["forecast_date"].date().isoformat(),

            "forecast_type":"FUTURE",

            "actual_demand":None,

            "predicted_demand":float(

                row["predicted_demand"]

            ),

            "model_name":model_name,

            "mae":None,

            "rmse":None,

            "mape":None,

            "r2":None,

            "experiment_id":EXPERIMENT_ID

        })

    (
        supabase
        .table("hdgen_final_demand_forecasts")
        .upsert(

            records,

            on_conflict="forecast_date,forecast_type,experiment_id"

        )
        .execute()
    )

    print()

    print("Final Forecast Saved :",len(records))
def run_genome_decoder():

    print()

    print("="*80)

    print("HDGEN GENOME DECODER")

    print("="*80)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train,
        test,
        future_matrix,
        scaler
    ) = prepare_decoder_data()
    print("\nDecoder Training")

    print("Training Rows :", len(train))

    print("Testing Rows  :", len(test))

    print("Genes         :", X_train.shape[1])

    results = run_decoder_models(

        X_train,
        X_test,
        y_train,
        y_test

    )

    best, ranked = select_best_decoder(results)
    joblib.dump(
        best["model"],
        "hdgen_best_decoder.pkl"
    )
    print("\nBest Decoder")
    print("-" * 40)

    print("Model :", best["model_name"])

    print(f"MAE  : {best['mae']:.3f}")
    print(f"RMSE : {best['rmse']:.3f}")
    print(f"MAPE : {best['mape']:.2f}")
    print(f"R²   : {best['r2']:.4f}")

    print()

    print(ranked[
        [
            "rank",
            "model_name",
            "mae",
            "rmse",
            "mape",
            "r2"
        ]
    ])

    save_decoder_results(

        ranked,

        train,

        test,

        X_train.shape[1]

    )

    best_model, scaler, feature_order = retrain_best_decoder(
        best["model_name"]
    )

    future_prediction = forecast_future_demand(
        best_model,
        scaler,
        future_matrix,
        feature_order
    )

    save_final_forecast(

        test,

        y_test,

        best["prediction"],

        future_prediction,

        best["model_name"],

        {

            "mae":best["mae"],

            "rmse":best["rmse"],

            "mape":best["mape"],

            "r2":best["r2"]

        }

    )

    print()

    print("="*80)

    print("HDGEN COMPLETE")

    print("="*80)
def run_hdgen_forecast(product="Overall"):

    # Run decoder if you want
    run_genome_decoder()

    response = (
        supabase
        .table("hdgen_decoder_results")
        .select("*")
        .eq("selected", True)
        .eq("experiment_id", EXPERIMENT_ID)
        .execute()
    )

    best = response.data[0]

    return {
        "mae": best["mae"],
        "rmse": best["rmse"],
        "mape": best["mape"],
        "r2": best["r2"]
    }
if __name__ == "__main__":

    run_genome_decoder()