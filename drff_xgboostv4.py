from forecast_deep_demand_transitions import (
    run_forecast_deep_demand_transitions
)

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from xgboost import XGBRegressor



def run_drff_xgboostv4(
    product="Overall"
):

    # ==========================================
    # LOAD TRANSITION DATA
    # ==========================================

    result = run_forecast_deep_demand_transitions(
        product
    )

    if "error" in result:
        return result

    df = pd.DataFrame(
        result["regimes"]
    )
    print(df.columns.tolist())
    required_cols = [

        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",

        "rolling_mean_7",
        "rolling_mean_30",

        "rolling_std_7",
        "rolling_std_30",

        "memory_state",
        "rolling_memory_strength",
        "memory_half_life",

        "state",
        "state_stability",
        "expected_duration",
        "trend_30",
        "trend_90",

        "shock_score",
        "z_score",

        "weekly_autocorrelation",

        "entropy_30",
        "entropy_90",

        "product_cv",
        "product_entropy",

        "transition_entropy",
        "transition_strength",

        "z1","z2","z3","z4",
        "z5","z6","z7","z8"
    ]

    missing = [
        c for c in required_cols
        if c not in df.columns
    ]

    if missing:
        return {
            "error":
            f"Missing columns: {missing}"
        }
    #print(df.columns.tolist())
    

    split_index = result[
        "split_index"
    ]

    # ==========================================
    # CLEAN
    # ==========================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    # ==========================================
    # TRAIN STATE PROFILES
    # ==========================================

    train_df = df.iloc[
        :split_index
    ].copy()



   
    test_df = df.iloc[
        split_index:
    ].copy()

   

   

    # ==========================================
    # DRFF FORECAST
    # ==========================================

    feature_cols = [

        "lag_7",
        "lag_14",
        "lag_30",

        "rolling_mean_7",
        "rolling_mean_30",

        "rolling_std_30",

        "trend_90",

        "shock_score",
        "z_score",

        "entropy_30",

        "transition_entropy",

        "memory_state",
        "rolling_memory_strength",
        "memory_half_life",

        "expected_duration",

        "z1",
        "z2",
        "z3",
        "z4",
        "z5"
    ]
    X_train = train_df[
        feature_cols
    ]

    X_test = test_df[
        feature_cols
    ]
   
    y_train = train_df[
        "demand"
    ]

    model = XGBRegressor(

        n_estimators=1000,

        max_depth=6,

        learning_rate=0.02,

        subsample=0.8,

        colsample_bytree=0.8,

        min_child_weight=2,

        gamma=0.05,

        reg_alpha=0.1,

        reg_lambda=1.0,

        objective="reg:squarederror",

        random_state=42
    )
    model.fit(
        X_train,
        y_train
    )

    forecasts = model.predict(
        X_test
    )
    feature_importance = dict(

        zip(
            feature_cols,
            model.feature_importances_
        )

    )

    y_test = test_df[
        "demand"
    ].values

    forecasts = np.array(
        forecasts
    )
 
    # ==========================================
    # METRICS
    # ==========================================

    mae = mean_absolute_error(
        y_test,
        forecasts
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            forecasts
        )
    )

    nonzero_mask = (
        y_test != 0
    )

    mape = (

        np.abs(

            (
                y_test[
                    nonzero_mask
                ]
                -
                forecasts[
                    nonzero_mask
                ]
            )

            /

            y_test[
                nonzero_mask
            ]

        )

    ).mean() * 100
    

    # ==========================================
    # FUTURE 30 DAYS
    # ==========================================

    history = (
        df["demand"]
        .tolist()
    )

    future_forecast = []

    
    state_sequence = (
        df["state"]
        .tail(30)
        .tolist()
    )
    current_state = (
        state_sequence[-1]
    )


    stability = result[
        "stability_map"
    ].get(
        str(current_state),
        0.5
    )

    duration = result[
        "expected_duration_map"
    ].get(
        str(current_state),
        1
    )
    last_row = df.iloc[-1]
    memory_state = last_row["memory_state"]
    shock_score = last_row["shock_score"]

    z_score = last_row["z_score"]
    history = df["demand"].tolist()
   

   
    alpha = min(
            1.0,
            1.0 /
            max(
                last_row[
                    "memory_half_life"
                ],
                1
            )
        )

    for i in range(30):
        lag_7 = history[-7]
        lag_14 = history[-14]
        lag_30 = history[-30]

        rolling_mean_7 = np.mean(
            history[-7:]
        )

        rolling_mean_30 = np.mean(
            history[-30:]
        )
       

        trend_90 = (
            rolling_mean_30
            -
            np.mean(history[-60:-30])
        )

        rolling_std_30 = np.std(
            history[-30:]
        )
        entropy_30 = pd.Series(
            history[-30:]
        ).rank(
            pct=True
        )

        entropy_30 = -np.sum(
            entropy_30 *
            np.log(
                entropy_30 + 1e-10
            )
        ) / len(entropy_30)

       

        

        future_X = pd.DataFrame([{
           

            "lag_7": lag_7,

            "lag_14": lag_14,

            "lag_30": lag_30,

            "rolling_mean_7":
            rolling_mean_7,

            "rolling_mean_30":
            rolling_mean_30,

            "rolling_std_30":
            rolling_std_30,
           

            "trend_90":
            trend_90,
            "shock_score":
            shock_score,

            "z_score":
            z_score,

           

            "entropy_30":
            entropy_30,

          

         


            "transition_entropy":
            last_row["transition_entropy"],

           

            "memory_state": memory_state,

         


  


            "expected_duration": duration,

            "rolling_memory_strength":
            last_row["rolling_memory_strength"],

            "memory_half_life":
            last_row["memory_half_life"],

            "z1": last_row["z1"],
            "z2": last_row["z2"],
            "z3": last_row["z3"],
            "z4": last_row["z4"],
            "z5": last_row["z5"],
           
        }])
        future_X = future_X[
            feature_cols
        ]
        pred = model.predict(
            future_X
        )[0]  
        shock_score = (
            pred /
            max(
                rolling_mean_30,
                1
            )
        )

        z_score = (
            pred -
            rolling_mean_30
        ) / max(
            rolling_std_30,
            1
        )   

       

       

        future_forecast.append(
            float(pred)
        )

        history.append(pred)

        duration = max(
            duration - 1,
            1
        )
        
        memory_state = (
            (1 - alpha)
            * memory_state
            +
            alpha * pred
        )
       

       

    future_dates = pd.date_range(

        start=(
            pd.to_datetime(
                df["date"].max()
            )
            +
            pd.Timedelta(days=1)
        ),

        periods=30,

        freq="D"
    )

    return {

        "model": "DRFF_XGBoost",
        "feature_importance":
        feature_importance,

        "mae":
        round(mae, 2),

        "rmse":
        round(rmse, 2),

        "mape":
        round(mape, 2),

        "dates":
        test_df["date"]
        .astype(str)
        .tolist(),

        "actual":
        y_test.tolist(),

        "forecast":
        forecasts.tolist(),

        "future_dates":
        future_dates
        .astype(str)
        .tolist(),

        "future_forecast":
        future_forecast
    }


if __name__ == "__main__":

    result = run_drff_xgboostv4(
        "Overall"
    )

    print(
        "\nMAE:",
        result["mae"]
    )

    print(
        "\nRMSE:",
        result["rmse"]
    )

    print(
        "\nMAPE:",
        result["mape"]
    )
  
    print("\nFeature Importance")

    
    for k, v in result["feature_importance"].items():
  

        print(
            f"{k}: {v:.4f}"
        )