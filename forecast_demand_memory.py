from load_data import load_dealer_orders

import pandas as pd
import numpy as np
from scipy.stats import linregress

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import STL
from scipy.stats import entropy





def calc_entropy(window):

    try:

        hist, _ = np.histogram(
            window,
            bins=10
        )

        hist = hist + 1e-10

        return float(
            entropy(hist)
        )

    except Exception:
        return np.nan
def calc_trend(window):

    try:

        x = np.arange(len(window))

        slope, _, _, _, _ = linregress(
            x,
            window
        )

        return slope

    except:
        return np.nan


def run_forecast_demand_memory(product):

    # ==================================================
    # LOAD DATA
    # ==================================================

    df = load_dealer_orders()

    if product == "Overall":
        product_df = df.copy()
    else:
        product_df = df[
            df["product_code"] == product
        ].copy()

    demand = (
        product_df
        .groupby("order_date")["total_weight_mt"]
        .sum()
        .reset_index()
    )

    demand["order_date"] = pd.to_datetime(
        demand["order_date"]
    )

    demand = demand.sort_values(
        "order_date"
    )

    # ==================================================
    # DAILY CALENDAR
    # ==================================================

    daily = (
        demand
        .set_index("order_date")
        .asfreq("D", fill_value=0)
        .reset_index()
    )

    y = daily["total_weight_mt"]
    split_index = int(
        len(y) * 0.8
    )

    train_y = y.iloc[:split_index]
    test_y = y.iloc[split_index:]
    

    if len(y) < 90:
        return {
            "error": "Need at least 90 observations"
        }
    product_mean = float(
        train_y.mean()
    )

    product_std = float(
        train_y.std()
    )

    product_cv = (
        product_std
        /
        (
            product_mean
            + 1e-6
        )
    )

    hist, _ = np.histogram(
        train_y,
        bins=10
    )

    product_entropy = float(
        entropy(hist + 1e-10)
    )

    product_memory_strength = float(
        abs(
            acf(
                train_y,
                nlags=1
            )[1]
        )
    )
        

    # ==================================================
    # MEMORY METRICS
    # ==================================================

    MAX_LAG = min(
        365,
        len(train_y) // 2
    )

    acf_values = acf(
        train_y,
        nlags=MAX_LAG
    )

    pacf_values = pacf(
        train_y,
        nlags=MAX_LAG,
        method="ywm"
    )

    # --------------------------------------------------
    # Memory Strength
    # --------------------------------------------------

    MS = float(
        abs(acf_values[1])
    )

    # --------------------------------------------------
    # Memory Length
    # --------------------------------------------------

    limit = 1.96 / np.sqrt(
        len(train_y)
    )

    ML = MAX_LAG

    for lag in range(
        1,
        len(acf_values) - 10
    ):

        window = np.abs(
            acf_values[
                lag:lag + 10
            ]
        )

        if np.all(
            window < limit
        ):
            ML = lag
            break

    # --------------------------------------------------
    # Half Life
    # --------------------------------------------------

    first_pacf = abs(
        pacf_values[1]
    )

    half_level = (
        first_pacf * 0.5
    )

    ML50 = ML

    for lag in range(
        2,
        len(pacf_values) - 4
    ):

        window = np.abs(
            pacf_values[
                lag:lag + 5
            ]
        )

        if np.all(
            window < half_level
        ):
            ML50 = lag
            break

    # ==================================================
    # MEMORY STATE
    # ==================================================

    alpha = 1 - np.exp(
        -np.log(2) / max(ML50, 1)
    )

    memory_state = [float(y.iloc[0])]

    for i in range(
        1,
        len(y)
    ):

        state = (
            alpha * y.iloc[i - 1]
            +
            (1 - alpha)
            * memory_state[-1]
        )

        memory_state.append(
            float(state)
        )
        memory_residual = (
            y
            -
            pd.Series(memory_state)
        )
    # ==================================================
    # MEMORY SUMMARY
    # ==================================================

    weekly_lags = [
        lag
        for lag in range(
            7,
            len(acf_values),
            7
        )
    ]

    weekly_memory = (
        float(
            np.mean(
                np.abs(
                    acf_values[
                        weekly_lags
                    ]
                )
            )
        )
        if len(weekly_lags) > 0
        else 0
    )

    try:

        stl = STL(
            train_y,
            period=7,
            robust=True
        ).fit()

        seasonal = stl.seasonal
        residual = stl.resid

        SS = float(
            1
            -
            (
                np.var(residual)
                /
                (
                    np.var(seasonal)
                    +
                    np.var(residual)
                    +
                    1e-8
                )
            )
        )

    except Exception:

        seasonal = np.zeros(
            len(y)
        )

        residual = np.zeros(
            len(y)
        )

        SS = 0

    hist, _ = np.histogram(
        train_y,
        bins=10,
        density=True
    )

    hist = hist + 1e-10

    E = entropy(hist)

    E_norm = float(
        E / np.log(len(hist))
    )
    features_df = pd.DataFrame({

        "date": daily["order_date"],

        "demand": y,

        "memory_state": memory_state,

        "memory_residual": memory_residual
    })
   
    features_df["log_demand"] = (
        np.log1p(
            y.shift(1)
        )
    )

    features_df["log_memory_state"] = (
        np.log1p(memory_state)
    )
    product_cv = (
        product_std
        /
        (
            product_mean
            + 1e-6
        )
    )
    features_df["product_mean"] = product_mean

    features_df["product_cv"] = product_cv

    features_df["product_entropy"] = (
        product_entropy
    )

    features_df["product_memory_strength"] = (
        product_memory_strength
    )

  

    # ==================================================
    # DEMAND FEATURES
    # ==================================================

    for lag in [1, 7, 14, 30]:

        features_df[
            f"lag_{lag}"
        ] = y.shift(lag)

    for w in [7, 30, 90]:

        features_df[
            f"rolling_mean_{w}"
        ] = (
            y.shift(1)
            .rolling(
                w,
                min_periods=1
            )
            .mean()
        )

        features_df[
            f"rolling_std_{w}"
        ] = (
            y.shift(1)
            .rolling(
                w,
                min_periods=2
            )
            .std()
        )

    for w in [30, 90]:

        features_df[f"cv_{w}"] = (
            features_df[f"rolling_std_{w}"]
            /
            (
                features_df[f"rolling_mean_{w}"]
                + 1e-6
            )
        )
    

    
    features_df["trend_30"] = (
        y.shift(1)
        .rolling(30)
        .apply(calc_trend, raw=False)
    )

    features_df["trend_90"] = (
        y.shift(1)
        .rolling(90)
        .apply(calc_trend, raw=False)
    )
    features_df["trend_gap"] = (
        features_df["rolling_mean_7"]
        -
        features_df["rolling_mean_30"]
    )
    features_df["shock_score"] = (
        y.shift(1)
        /
        (
            features_df["rolling_mean_30"]
            + 1e-6
        )
    )

    features_df["shock_score"] = (
        features_df["shock_score"]
        .clip(
            lower=0,
            upper=10
        )
    )

    features_df["z_score"] = (
        y.shift(1)
        -
        features_df["rolling_mean_30"]
    ) / (
        features_df["rolling_std_30"]
        + 1e-6
    )
    # ==================================================
    # SEASONALITY FEATURES
    # ==================================================

    features_df[
        "day_of_week"
    ] = (
        features_df["date"]
        .dt.dayofweek
    )

    features_df[
        "week_of_year"
    ] = (
        features_df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )
    features_df["month"] = (
        features_df["date"]
        .dt.month
    )

    features_df["quarter"] = (
        features_df["date"]
        .dt.quarter
    )

    features_df["weekly_autocorrelation"] = (
        y.shift(1)
        .rolling(90)
        .corr(y.shift(7))
    )
    features_df["seasonality_strength"] = (
        y.shift(1)
        .rolling(90)
        .corr(y.shift(7))
        .abs()
    )
    # ==================================================
    # SHOCK FREQUENCY
    # ==================================================

    daily_shock = (
        np.abs(
            y.shift(1).pct_change()
        ) > 0.5
    ).astype(int)

    features_df["shock_rate_30"] = (
        daily_shock
        .rolling(30)
        .mean()
    )

    # ==================================================
    # COMPLEXITY FEATURES
    # ==================================================

    features_df[
        "entropy_30"
    ] = (
        y.shift(1)
        .rolling(30)
        .apply(
            calc_entropy
        )
    )

    features_df[
        "entropy_90"
    ] = (
        y.shift(1)
        .rolling(90)
        .apply(
            calc_entropy
        )
    )

    features_df[
        "skewness_30"
    ] = (
        y.shift(1)
        .rolling(30)
        .skew()
    )

    features_df[
        "skewness_90"
    ] = (
        y.shift(1)
        .rolling(90)
        .skew()
    )

    features_df[
        "kurtosis_30"
    ] = (
        y.shift(1)
        .rolling(30)
        .kurt()
    )

    features_df[
        "kurtosis_90"
    ] = (
        y.shift(1)
        .rolling(90)
        .kurt()
    )

    # ==================================================
    # INTERMITTENCY FEATURES
    # ==================================================

    days_since_last = []

    last_order = None

    days_since_last = []

    last_order = None

    for i, value in enumerate(
        y.shift(1).fillna(0)
    ):

        if value > 0:

            days_since_last.append(0)

            last_order = i

        else:

            if last_order is None:

                days_since_last.append(np.nan)

            else:

                days_since_last.append(
                    i - last_order
                )

    features_df[
        "days_since_last_order"
    ] = days_since_last

    for w in [7, 30, 90]:

        features_df[
            f"rolling_order_frequency_{w}"
        ] = (
            (y > 0)
            .shift(1)
            .rolling(w)
            .mean()
        )

    for w in [30, 90]:

        features_df[
            f"zero_demand_ratio_{w}"
        ] = (
            (y == 0)
            .shift(1)
            .rolling(w)
            .mean()
        )
    features_df["memory_state_ratio"] = (
        features_df["memory_state"]
        /
        (
            features_df["rolling_mean_30"]
            + 1e-6
        )
    )

    # ==================================================
    # CLEANUP
    # ==================================================

   
    # ==========================================
    # DYNAMIC MEMORY FEATURES
    # ==========================================

    def rolling_ms(x):

        return abs(
            acf(
                x,
                nlags=1,
                fft=False
            )[1]
        )

    features_df["rolling_memory_strength"] = (
        y.shift(1)
        .rolling(90)
        .apply(rolling_ms)
    )

    ms = (
        features_df["rolling_memory_strength"]
        .abs()
        .clip(
            lower=0.01,
            upper=0.99
        )
    )

    features_df["memory_half_life"] = (
        -np.log(2)
        / np.log(ms)
    )

  
    features_df = features_df.replace(
        [np.inf, -np.inf],
        np.nan
    )
    
   

    numeric_cols = (
        features_df
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    train_medians = (
        features_df
        .iloc[:split_index]
        [numeric_cols]
        .median()
    )

    features_df[numeric_cols] = (
        features_df[numeric_cols]
        .fillna(train_medians)
    )
    # ==========================================
    # FORECASTABILITY INDEX
    # ==========================================

    fi_features = [
        "cv_30",
        "entropy_30",
        "shock_rate_30",
        "rolling_memory_strength",
        "seasonality_strength"
    ]

    train_part = features_df.iloc[:split_index]

    for col in fi_features:

        min_v = train_part[col].min()
        max_v = train_part[col].max()

        features_df[col + "_norm"] = (
            features_df[col] - min_v
        ) / (
            max_v - min_v + 1e-6
        )
    features_df["forecastability_index"] = (

        0.25 * (
            1 - features_df["cv_30_norm"]
        )

        + 0.20 * (
            1 - features_df["entropy_30_norm"]
        )

        + 0.20 * (
            1 - features_df["shock_rate_30_norm"]
        )

        + 0.20 * (
            features_df[
                "rolling_memory_strength_norm"
            ]
        )

        + 0.15 * (
            features_df[
                "seasonality_strength_norm"
            ]
        )
    )
    features_df["forecastability_index"] = (
        features_df["forecastability_index"]
        .clip(0, 1)
    )
    features_df["forecast_zone"] = pd.cut(

        features_df["forecastability_index"],

        bins=[
            0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0
        ],

        labels=[
            "Z1_VeryLow",
            "Z2_Low",
            "Z3_Medium",
            "Z4_High",
            "Z5_VeryHigh"
        ]
    )
    print(
        features_df["forecastability_index"]
        .describe()
    )
    # ==========================================
    # FORECASTABILITY STATES
    # ==========================================

    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    features_df["fi_cluster"] = (
        kmeans.fit_predict(
            features_df[
                ["forecastability_index"]
            ]
        )
    )

    cluster_means = (
        features_df
        .groupby("fi_cluster")
        ["forecastability_index"]
        .mean()
        .sort_values()
    )

    cluster_order = (
        cluster_means.index.tolist()
    )

    state_map = {

        cluster_order[0]: "L",

        cluster_order[1]: "M",

        cluster_order[2]: "H"
    }

    features_df["forecastability_state"] = (
        features_df["fi_cluster"]
        .map(state_map)
    )

 

    # ==================================================
    # RETURN
    # ==================================================

    return {

         "split_index":
            int(split_index),
        "features":
        features_df.to_dict("records"),

        # ==================================
        # GLOBAL MEMORY METRICS
        # ==================================

        "ms": float(MS),

        "ml": float(ML),

        "ml50": float(ML50),

        "ss": float(SS),

        "entropy": float(E_norm),

        # ==================================
        # PRODUCT-LEVEL INTERMITTENCY
        # ==================================

        "trend_90": float(
            features_df["trend_90"]
            .iloc[-1]
        ),

        "zero_demand_ratio_90": float(
            features_df[
                "zero_demand_ratio_90"
            ].iloc[-1]
        ),

        # ==================================
        # SUMMARY
        # ==================================

        "memory_summary": {

            "weekly_memory":
            weekly_memory,

            "seasonal_strength":
            SS,

            "entropy":
            E_norm,
           
        }
    }
if __name__ == "__main__":

    result = run_forecast_demand_memory(
        "Overall"
    )

    df = pd.DataFrame(
        result["features"]
    )

    print(
        df["forecastability_index"]
        .describe()
    )

    print("Done")

    print(
        df[
            [
                "forecastability_index",
                "forecast_zone"
            ]
        ].tail()
    )

    # ==================================
    # VALIDATION
    # ==================================

    df["naive_forecast"] = (
        df["demand"].shift(1)
    )

    df["naive_error"] = (
        df["demand"]
        - df["naive_forecast"]
    ).abs()

    print(
        df.groupby("forecast_zone")[
            "naive_error"
        ].mean()
    )
    print(
        df["forecast_zone"]
        .value_counts()
    )
    print(
        df.groupby("forecast_zone")[
            "forecastability_index"
        ].agg(
            ["min", "mean", "max"]
        )
    )
    print(
        df["forecastability_state"]
        .value_counts()
    )

    print(
        df.groupby(
            "forecastability_state"
        )[
            "forecastability_index"
        ].mean()
    )