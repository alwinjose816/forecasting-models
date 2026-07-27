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



def run_drff_xgboostv5(
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


        "trend_90",

    

        "transition_entropy",

        "memory_state",
        "rolling_memory_strength",
        "memory_half_life",

        
     
        "weekly_autocorrelation",

        "z1",
        "z2",
        "z3"
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
    feature_importance = {

        feature: float(importance)

        for feature, importance in zip(
            feature_cols,
            model.feature_importances_
        )

    }

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
    states = df["state"].values

    unique_states = sorted(
        np.unique(states)
    )

    n_states = len(unique_states)

    transition_matrix = np.zeros(
        (n_states, n_states)
    )

    for i in range(len(states) - 1):

        s1 = int(states[i])
        s2 = int(states[i + 1])

        transition_matrix[s1, s2] += 1

    for i in range(n_states):

        row_sum = transition_matrix[i].sum()

        if row_sum > 0:

            transition_matrix[i] /= row_sum

        else:

            transition_matrix[i] = (
                np.ones(n_states)
                / n_states
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
    # ====================================
    # STATE LATENT CENTROIDS
    # ====================================

    state_embeddings = (
        df.groupby("state")[
            ["z1","z2","z3"]
        ]
        .mean()
    )
    memory_state = last_row["memory_state"]
    

   
    alpha = min(
        0.3,
        1.0 /
        max(
            last_row["memory_half_life"],
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
        probs = transition_matrix[
            int(current_state)
        ]

        transition_entropy = -np.sum(
            probs * np.log(
                probs + 1e-10
            )
        )
    
        # ====================================
        # DYNAMIC LATENT EMBEDDINGS
        # ====================================

        z_values = state_embeddings.loc[
            current_state
        ]

        z1 = z_values["z1"]
        z2 = z_values["z2"]
        z3 = z_values["z3"]
       

       

        weekly_autocorrelation = pd.Series(
            history[-30:]
        ).autocorr(lag=7)

        if np.isnan(
            weekly_autocorrelation
        ):
            weekly_autocorrelation = 0
        rolling_memory_strength = pd.Series(
            history[-90:]
        ).autocorr(lag=1)

        if np.isnan(
            rolling_memory_strength
        ):
            rolling_memory_strength = 0

        rolling_memory_strength = np.clip(
            abs(rolling_memory_strength),
            0.01,
            0.99
        )

        memory_half_life = (
            -np.log(2)
            /
            np.log(
                rolling_memory_strength
            )
        )

        future_X = pd.DataFrame([{
           

            "lag_7": lag_7,

            "lag_14": lag_14,

            "lag_30": lag_30,

            "rolling_mean_7":
            rolling_mean_7,

            "rolling_mean_30":
            rolling_mean_30,

      
           

            "trend_90":
            trend_90,
           "weekly_autocorrelation":
            weekly_autocorrelation,
          
            

           
   

          

         


            "transition_entropy":
            transition_entropy,

           

            "memory_state": memory_state,

         


  
            "state":
            current_state,

        
          

            "rolling_memory_strength":
            rolling_memory_strength,

            "memory_half_life":
            memory_half_life,
            "z1": z1,
            "z2": z2,
            "z3": z3,

          
           
        }])
        future_X = future_X[
            feature_cols
        ]
        pred = float(
            model.predict(
                future_X
            )[0]
        ) 
        pred = min(
            pred,
            rolling_mean_30 * 1.5
        )
        recent_min = np.min(
            history[-90:]
        )

        recent_max = np.max(
            history[-90:]
        )

        pred = np.clip(
            pred,
            recent_min * 0.5,
            recent_max * 1.5
        )
                

       

       

       

        future_forecast.append(
            float(pred)
        )
        print(
            f"Day {i+1}",
            f"Pred={pred:.2f}",
            f"State={current_state}",
            f"Duration={duration}",
            f"Memory={memory_state:.2f}"
        )

        history.append(pred)
        duration -= 1
        if duration <= 0:

            probs = transition_matrix[
                int(current_state)
            ]

            current_state = np.random.choice(
                np.arange(len(probs)),
                p=probs
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

       
   

        # ====================================
        # MEMORY UPDATE
        # ====================================

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
    future_forecast = [
        float(x)
        for x in future_forecast
    ]
    

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
        [float(x) for x in y_test],

        "forecast":
        [float(x) for x in forecasts],

        "future_dates":
        future_dates
        .astype(str)
        .tolist(),

        "future_forecast":
        future_forecast
    }


if __name__ == "__main__":

    result = run_drff_xgboostv5(
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