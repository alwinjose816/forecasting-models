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


def run_demand_memory(product):

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
    

    if len(y) < 90:
        return {
            "error": "Need at least 90 observations"
        }
    product_mean = float(
        y.mean()
    )

    product_std = float(
        y.std()
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
        y,
        bins=10
    )

    product_entropy = float(
        entropy(hist + 1e-10)
    )

    product_memory_strength = float(
        abs(acf(y, nlags=1)[1])
    )
    

    # ==================================================
    # MEMORY METRICS
    # ==================================================

    MAX_LAG = min(
        365,
        len(y) // 2
    )

    acf_values = acf(
        y,
        nlags=MAX_LAG
    )

    pacf_values = pacf(
        y,
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

    limit = 1.96 / np.sqrt(len(y))

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

    ML50 = 30

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
        y.values
        -
        np.array(memory_state)
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
            y,
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
        y,
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
        np.log1p(y)
    )

    features_df["log_memory_state"] = (
        np.log1p(memory_state)
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
        y
        /
        (
            features_df["rolling_mean_30"]
            + 1e-6
        )
    )

    # Prevent extreme spikes from dominating clustering
    features_df["shock_score"] = (
        features_df["shock_score"]
        .clip(
            lower=0,
            upper=10
        )
    )

    features_df["z_score"] = (
        y
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
        .corr(y.shift(8))
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

    for i, value in enumerate(y):

        if value > 0:

            days_since_last.append(0)

            last_order = i

        else:

            if last_order is None:
                days_since_last.append(
                    np.nan
                )
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

    features_df["rolling_memory_strength"] = (
        y.shift(1)
        .rolling(90)
        .corr(y.shift(2))
    )

    ms = features_df["rolling_memory_strength"].clip(
        lower=0.01,
        upper=0.99
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

    features_df[numeric_cols] = (
        features_df[numeric_cols]
        .fillna(
            features_df[numeric_cols]
            .median()
        )
    )

 

    # ==================================================
    # RETURN
    # ==================================================

    return {

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
            E_norm
        }
    }