from forecast_demand_memory import (
    run_forecast_demand_memory
)

from xgboost_forecast_validation import (
    run_xgboost_forecast_validation
)

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


def run_forecastability_error_model(
    product="Overall"
):

    # =====================================
    # LOAD FORECASTABILITY FEATURES
    # =====================================

    fi_result = run_forecast_demand_memory(
        product
    )

    fi_df = pd.DataFrame(
        fi_result["features"]
    )

    fi_df["date"] = pd.to_datetime(
        fi_df["date"]
    )

    # =====================================
    # LOAD XGBOOST FORECAST RESULTS
    # =====================================

    xgb_result = run_xgboost_forecast_validation(
        product
    )

    forecast_df = pd.DataFrame({

        "date":
        pd.to_datetime(
            xgb_result["dates"]
        ),

        "actual":
        xgb_result["actual"],

        "forecast":
        xgb_result["forecast"]

    })

    # =====================================
    # FORECAST ERROR
    # =====================================

    forecast_df["forecast_error"] = np.abs(

        forecast_df["actual"]

        -

        forecast_df["forecast"]

    )

    # =====================================
    # MERGE
    # =====================================

    df = forecast_df.merge(

        fi_df[

            [
                "date",
                "forecastability_index",
                "forecastability_state",
                "cv_30",
                "entropy_30",
                "shock_rate_30",
                "rolling_memory_strength",
                "seasonality_strength"
            ]

        ],

        on="date",

        how="left"
    )

    # =====================================
    # ENCODE STATE
    # =====================================

    encoder = LabelEncoder()

    df["state_encoded"] = (
        encoder.fit_transform(
            df["forecastability_state"]
        )
    )

    # =====================================
    # FEATURES
    # =====================================

    feature_cols = [

        "forecastability_index",

        "state_encoded",

        "cv_30",

        "entropy_30",

        "shock_rate_30",

        "rolling_memory_strength",

        "seasonality_strength"

    ]

    X = df[
        feature_cols
    ]

    y = df[
        "forecast_error"
    ]

    # =====================================
    # TRAIN TEST SPLIT
    # =====================================

    split_index = int(
        len(df) * 0.8
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # =====================================
    # ERROR PREDICTOR
    # =====================================

    model = XGBRegressor(

        n_estimators=300,

        max_depth=3,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    pred_error = model.predict(
        X_test
    )

    # =====================================
    # METRICS
    # =====================================

    mae = mean_absolute_error(
        y_test,
        pred_error
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred_error
        )
    )

    r2 = r2_score(
        y_test,
        pred_error
    )

    # =====================================
    # CONFIDENCE SCORE
    # =====================================

    max_error = y.max()

    confidence = (

        1

        -

        pred_error

        /

        (
            max_error + 1e-6
        )

    )

    confidence = np.clip(
        confidence,
        0,
        1
    )

    # =====================================
    # FEATURE IMPORTANCE
    # =====================================

    importance = pd.DataFrame({

        "feature":
        feature_cols,

        "importance":
        model.feature_importances_

    })

    importance = (

        importance

        .sort_values(
            "importance",
            ascending=False
        )

    )

    print("\n")
    print("=" * 60)
    print("FORECASTABILITY ERROR MODEL")
    print("=" * 60)

    print(
        f"MAE : {mae:.2f}"
    )

    print(
        f"RMSE: {rmse:.2f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )

    print("\nFeature Importance")

    print(
        importance
    )

    print("\nSample Predictions")

    sample = pd.DataFrame({

        "actual_error":
        y_test.values[:10],

        "predicted_error":
        pred_error[:10],

        "confidence":
        confidence[:10]

    })

    print(
        sample.round(2)
    )

    return {

        "mae":
        float(mae),

        "rmse":
        float(rmse),

        "r2":
        float(r2),

        "feature_importance":
        importance.to_dict(
            "records"
        )
    }


if __name__ == "__main__":

    run_forecastability_error_model(
        "Overall"
    )