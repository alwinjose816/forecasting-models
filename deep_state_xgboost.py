from deep_demand_states import run_deep_demand_states

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def run_deep_state_xgboost(
    product="Overall",
    latent_dim=8
):

    # ==========================================
    # LOAD DEEP STATES
    # ==========================================

    result = run_deep_demand_states(
        product=product,
        latent_dim=latent_dim
    )

    if "error" in result:
        return result

    df = pd.DataFrame(
        result["states"]
    )

    # ==========================================
    # FEATURES
    # ==========================================

    baseline_features = [

        "lag1",
        "lag7",
        "lag14",
        "lag30",

        "rolling_mean_7",
        "rolling_mean_30",

        "day_of_week",
        "month"
    ]

    latent_features = [
        f"z{i}"
        for i in range(
            1,
            latent_dim + 1
        )
    ]

    state_dummies = pd.get_dummies(
        df["state"],
        prefix="state"
    )

    df = pd.concat(
        [df, state_dummies],
        axis=1
    )

    state_features = list(
        state_dummies.columns
    )

    deep_features = (
        baseline_features
        + latent_features
        + state_features
    )

    # ==========================================
    # CLEAN
    # ==========================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    # ==========================================
    # TARGET
    # ==========================================

    y = df["demand"]

    # ==========================================
    # BASELINE DATA
    # ==========================================

    X_baseline = df[
        baseline_features
    ]

    # ==========================================
    # DEEP DATA
    # ==========================================

    X_deep = df[
        deep_features
    ]

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    split_index = int(
        len(df) * 0.8
    )

    Xb_train = X_baseline.iloc[
        :split_index
    ]

    Xb_test = X_baseline.iloc[
        split_index:
    ]

    Xd_train = X_deep.iloc[
        :split_index
    ]

    Xd_test = X_deep.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    # ==========================================
    # BASELINE MODEL
    # ==========================================

    baseline_model = XGBRegressor(

        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42
    )

    baseline_model.fit(
        Xb_train,
        y_train
    )

    baseline_pred = baseline_model.predict(
        Xb_test
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_pred
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_pred
        )
    )

    nonzero_mask = (
        y_test != 0
    )

    baseline_mape = (

        np.abs(
            (
                y_test[
                    nonzero_mask
                ]
                -
                baseline_pred[
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
    # DEEP MODEL
    # ==========================================

    deep_model = XGBRegressor(

        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42
    )

    deep_model.fit(
        Xd_train,
        y_train
    )
    feature_importance = dict(
        zip(
            Xd_train.columns,
            deep_model.feature_importances_
        )
    )

    deep_pred = deep_model.predict(
        Xd_test
    )

    deep_mae = mean_absolute_error(
        y_test,
        deep_pred
    )

    deep_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            deep_pred
        )
    )

    deep_mape = (

        np.abs(
            (
                y_test[
                    nonzero_mask
                ]
                -
                deep_pred[
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
    # IMPROVEMENT
    # ==========================================

    mae_improvement = (

        (baseline_mae - deep_mae)
        /
        baseline_mae

    ) * 100

    rmse_improvement = (

        (baseline_rmse - deep_rmse)
        /
        baseline_rmse

    ) * 100

    mape_improvement = (

        (baseline_mape - deep_mape)
        /
        baseline_mape

    ) * 100

    # ==========================================
    # RETURN
    # ==========================================

    return {

        "baseline": {

            "mae":
            round(
                baseline_mae,
                2
            ),

            "rmse":
            round(
                baseline_rmse,
                2
            ),

            "mape":
            round(
                baseline_mape,
                2
            )
        },

        "deep_state": {

            "mae":
            round(
                deep_mae,
                2
            ),

            "rmse":
            round(
                deep_rmse,
                2
            ),

            "mape":
            round(
                deep_mape,
                2
            )
        },

        "improvement": {

            "mae_percent":
            round(
                mae_improvement,
                2
            ),

            "rmse_percent":
            round(
                rmse_improvement,
                2
            ),

            "mape_percent":
            round(
                mape_improvement,
                2
            )
        },

        "dates":
        df.iloc[
            split_index:
        ]["date"].astype(str).tolist(),

        "actual":
        y_test.tolist(),

        "baseline_forecast":
        baseline_pred.tolist(),

        "deep_forecast":
        deep_pred.tolist(),
        "feature_importance":
        feature_importance
    }


if __name__ == "__main__":

    result = run_deep_state_xgboost(
        product="Overall",
        latent_dim=8
    )

    print("\nBASELINE")
    print(result["baseline"])

    print("\nDEEP STATE")
    print(result["deep_state"])

    print("\nIMPROVEMENT (%)")
    print(result["improvement"])