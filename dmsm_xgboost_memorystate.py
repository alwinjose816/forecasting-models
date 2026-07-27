from load_data import load_dealer_orders

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def run_dmsm_xgboost(product):

    # ==========================
    # Load Data
    # ==========================

    df = load_dealer_orders()

    if product == "Overall":
        product_df = df.copy()
    else:
        product_df = df[
            df["product_code"] == product
        ].copy()

    daily = (
        product_df
        .groupby(
            product_df["order_date"].dt.date
        )["total_weight_mt"]
        .sum()
        .reset_index()
    )

    daily.columns = [
        "date",
        "demand"
    ]

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily = daily.sort_values(
        "date"
    )

    # ==========================
    # DMSM MEMORY STATE
    # ==========================

    y = daily["demand"].values

    from demand_memory import run_demand_memory

    memory_info = run_demand_memory(product)

    ML50 = memory_info["ml50"]

    alpha = 1 - np.exp(
        -np.log(2) / ML50
    )

    memory_state = [float(y[0])]

    for i in range(1, len(y)):

        s = (
            alpha * y[i - 1]
            +
            (1 - alpha)
            * memory_state[-1]
        )

        memory_state.append(
            float(s)
        )

    daily["memory_state"] = (
        memory_state
    )

   

    # ==========================
    # Lag Features
    # ==========================

    daily["lag1"] = (
        daily["demand"]
        .shift(1)
    )

    daily["lag7"] = (
        daily["demand"]
        .shift(7)
    )

    daily["lag14"] = (
        daily["demand"]
        .shift(14)
    )

    daily["lag30"] = (
        daily["demand"]
        .shift(30)
    )

    # ==========================
    # Rolling Features
    # ==========================

    daily["rolling_mean_7"] = (
        daily["demand"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    daily["rolling_std_7"] = (
        daily["demand"]
        .shift(1)
        .rolling(7)
        .std()
    )

    daily["rolling_mean_30"] = (
        daily["demand"]
        .shift(1)
        .rolling(30)
        .mean()
    )

    # ==========================
    # Calendar Features
    # ==========================

    daily["day_of_week"] = (
        daily["date"]
        .dt.dayofweek
    )

    daily["month"] = (
        daily["date"]
        .dt.month
    )

    daily["quarter"] = (
        daily["date"]
        .dt.quarter
    )

    daily = daily.dropna()

    # ==========================
    # Feature Set
    # ==========================

    FEATURES = [

        "lag1",
        "lag7",
        "lag14",
        "lag30",

        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_30",

        "memory_state",

        "day_of_week",
        "month",
        "quarter"
    ]

    X = daily[FEATURES]

    y = daily["demand"]

    # ==========================
    # Train Test Split
    # ==========================

    split_index = int(
        len(daily) * 0.8
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    # ==========================
    # XGBoost
    # ==========================

    model = XGBRegressor(

        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,

        subsample=0.8,

        colsample_bytree=0.8,

        objective=
        "reg:squarederror",

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    test_forecast = (
        model.predict(X_test)
    )

    # ==========================
    # Feature Importance
    # ==========================

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance\n")
    importance_data = (
        importance.to_dict(
            orient="records"
        )
    )

    # ==========================
    # Metrics
    # ==========================

    mae = mean_absolute_error(
        y_test,
        test_forecast
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_forecast
        )
    )

    smape = np.mean(

        2 *
        np.abs(
            y_test.values
            -
            test_forecast
        )

        /

        (
            np.abs(
                y_test.values
            )
            +
            np.abs(
                test_forecast
            )
            +
            1e-8
        )

    ) * 100

    # ==========================
    # Future Forecast
    # ==========================

    history = (
        daily["demand"]
        .tolist()
    )

    memory_history = (
        daily["memory_state"]
        .tolist()
    )

    future_forecast = []

    future_memory_state = []

    future_memory_residual = []

    for i in range(30):

        future_date = (

            daily["date"].max()

            +

            pd.Timedelta(
                days=i + 1
            )
        )

        lag1 = history[-1]
        lag7 = history[-7]
        lag14 = history[-14]
        lag30 = history[-30]

        memory_state = (
            alpha * history[-1]
            +
            (1 - alpha)
            * memory_history[-1]
        )


        future_memory_state.append(
            float(memory_state)
        )

     

        row = [[

            lag1,
            lag7,
            lag14,
            lag30,

            np.mean(
                history[-7:]
            ),

            np.std(
                history[-7:]
            ),

            np.mean(
                history[-30:]
            ),

            memory_state,
          

            future_date.dayofweek,
            future_date.month,
            future_date.quarter

        ]]

        pred = model.predict(
            row
        )[0]

        future_forecast.append(
            float(pred)
        )

        history.append(
            pred
        )

        memory_history.append(
            memory_state
        )

    future_dates = pd.date_range(

        start=
        daily["date"].max()

        +

        pd.Timedelta(days=1),

        periods=30,

        freq="D"
    )
    nonzero_mask = y_test != 0

    mape = (
        np.abs(
            (
                y_test[nonzero_mask]
                - test_forecast[nonzero_mask]
            )
            /
            y_test[nonzero_mask]
        )
    ).mean() * 100

    return {

        "product":
        product,

        "mae":
        round(mae, 2),

        "rmse":
        round(rmse, 2),
        "feature_importance":
        importance_data,
        

    

        "ml50":
        ML50,

        "dates":
        daily.iloc[
            split_index:
        ]["date"]
        .astype(str)
        .tolist(),

        "actual":
        y_test.tolist(),

        "forecast":
        test_forecast.tolist(),

        "future_dates":
        future_dates
        .astype(str)
        .tolist(),
        "mape": round(mape, 2),
        "smape": round(smape, 2),

        "future_forecast":
        future_forecast,

        "future_memory_state":
        future_memory_state

    }