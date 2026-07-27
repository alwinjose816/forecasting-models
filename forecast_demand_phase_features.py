from load_data import load_dealer_orders

import numpy as np
import pandas as pd

from scipy.stats import entropy
from scipy.stats import linregress

from statsmodels.tsa.stattools import acf


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calc_entropy(window):
    """
    Shannon entropy of a rolling window.
    """

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
    """
    Linear trend slope.
    """

    try:

        x = np.arange(len(window))

        slope, _, _, _, _ = linregress(
            x,
            window
        )

        return float(slope)

    except Exception:

        return np.nan


def rolling_memory_strength(window):

    """
    Rolling lag-1 autocorrelation.
    """

    try:

        return abs(
            acf(
                window,
                nlags=1,
                fft=False
            )[1]
        )

    except Exception:

        return np.nan
# ========================================================
    # TRANSITION ENTROPY
    # ========================================================

def calc_transition_entropy(window):

    try:

        signs = np.sign(window)

        values, counts = np.unique(
            signs,
            return_counts=True
        )

        p = counts / counts.sum()

        return float(
            entropy(
                p + 1e-10
            )
        )

    except Exception:

        return np.nan


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_forecast_demand_phase_features(product="Overall"):

    # ========================================================
    # LOAD DATA
    # ========================================================

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

    demand = demand.sort_values(
        "order_date"
    )

    # ========================================================
    # DAILY CALENDAR
    # ========================================================

    daily = (

        demand

        .set_index("order_date")

        .asfreq("D", fill_value=0)

        .reset_index()

    )

    y = daily["total_weight_mt"]

    split_index = int(
        len(y) * 0.80
    )

    features = pd.DataFrame()

    features["sales_date"] = daily["order_date"]

    features["demand"] = y
    features["product"] = product

    # ========================================================
    # DEMAND MEMORY
    # ========================================================

    features["memory_short"] = y.shift(1)

    features["memory_weekly"] = y.shift(7)

    features["memory_biweekly"] = y.shift(14)

    features["memory_monthly"] = y.shift(30)

    # ========================================================
    # ROLLING STATISTICS
    # ========================================================

    windows = [7, 14, 30, 90]

    for w in windows:

        features[f"local_mean_{w}"] = (

            y.shift(1)

            .rolling(
                w,
                min_periods=1
            )

            .mean()

        )

        features[f"local_std_{w}"] = (

            y.shift(1)

            .rolling(
                w,
                min_periods=2
            )

            .std()

        )

        features[f"local_cv_{w}"] = (

            features[f"local_std_{w}"]

            /

            (

                features[f"local_mean_{w}"]

                + 1e-6

            )

        )

    # ========================================================
    # TREND
    # ========================================================

    features["trend_7"] = (

        y.shift(1)

        .rolling(7)

        .apply(
            calc_trend,
            raw=False
        )

    )

    features["trend_14"] = (

        y.shift(1)

        .rolling(14)

        .apply(
            calc_trend,
            raw=False
        )

    )

    features["trend_30"] = (

        y.shift(1)

        .rolling(30)

        .apply(
            calc_trend,
            raw=False
        )

    )

    features["trend_90"] = (

        y.shift(1)

        .rolling(90)

        .apply(
            calc_trend,
            raw=False
        )

    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    features["momentum_7"] = (

        y.shift(1)

        -

        y.shift(8)

    )

    features["momentum_14"] = (

        y.shift(1)

        -

        y.shift(15)

    )

    features["momentum_30"] = (

        y.shift(1)

        -

        y.shift(31)

    )

    # ========================================================
    # ACCELERATION
    # ========================================================

    features["acceleration_7"] = (

        features["momentum_7"]

        -

        features["momentum_7"].shift(1)

    )

    features["acceleration_14"] = (

        features["momentum_14"]

        -

        features["momentum_14"].shift(1)

    )

    features["acceleration_30"] = (

        features["momentum_30"]

        -

        features["momentum_30"].shift(1)

    )

    # ========================================================
    # PART 2 CONTINUES...
    # ========================================================
        # ========================================================
    # ENTROPY FEATURES
    # ========================================================

    for w in [7, 14, 30, 90]:

        features[f"demand_entropy_{w}"] = (

            y.shift(1)

            .rolling(
                w,
                min_periods=max(5, w // 2)
            )

            .apply(
                calc_entropy,
                raw=False
            )

        )

    # ========================================================
    # SHOCK SCORE
    # ========================================================

    features["shock_score"] = (

        (

            y.shift(1)

            -

            features["local_mean_30"]

        )

        /

        (

            features["local_std_30"]

            +

            1e-6

        )

    )

    features["shock_score"] = (

        features["shock_score"]

        .clip(-10, 10)

    )

    features["absolute_shock"] = (

        features["shock_score"]

        .abs()

    )

    # ========================================================
    # TRANSITION STRENGTH
    # ========================================================

    transition = (

        y.shift(1)

        .diff()

    )

    features["transition_strength"] = (

        transition.abs()

        /

        (

            features["local_std_30"]

            +

            1e-6

        )

    )

    features["transition_direction"] = np.sign(
        transition
    )

    

    features["transition_entropy_30"] = (

        transition

        .rolling(
            30,
            min_periods=15
        )

        .apply(
            calc_transition_entropy,
            raw=False
        )

    )

    features["transition_entropy_90"] = (

        transition

        .rolling(
            90,
            min_periods=45
        )

        .apply(
            calc_transition_entropy,
            raw=False
        )

    )

    # ========================================================
    # TRANSITION VOLATILITY
    # ========================================================

    features["transition_volatility"] = (

        transition

        .rolling(
            30,
            min_periods=10
        )

        .std()

    )

    train_part = features.iloc[:split_index]

    cv_max = train_part["local_cv_30"].dropna().max()

    entropy_max = train_part["demand_entropy_30"].dropna().max()

    shock_max = train_part["absolute_shock"].dropna().max()

    transition_max = train_part["transition_strength"].dropna().max()

    cv_norm = (
        features["local_cv_30"]
        /
        (cv_max + 1e-6)
    )

    entropy_norm = (
        features["demand_entropy_30"]
        /
        (entropy_max + 1e-6)
    )

    shock_norm = (
        features["absolute_shock"]
        /
        (shock_max + 1e-6)
    )

    transition_norm = (
        features["transition_strength"]
        /
        (transition_max + 1e-6)
    )

    features["state_stability"] = (

        1

        -

        (

            0.30 * cv_norm

            +

            0.25 * entropy_norm

            +

            0.25 * shock_norm

            +

            0.20 * transition_norm

        )

    )
   

    

    features["state_stability"] = (

        features["state_stability"]

        .clip(0, 1)

    )

    # ========================================================
    # DEMAND COMPLEXITY
    # ========================================================

    features["skewness_30"] = (

        y.shift(1)

        .rolling(30)

        .skew()

    )

    features["kurtosis_30"] = (

        y.shift(1)

        .rolling(30)

        .kurt()

    )

    # ========================================================
    # VOLATILITY CHANGE
    # ========================================================

    features["volatility_change"] = (

        features["local_std_7"]

        -

        features["local_std_30"]

    )

    # ========================================================
    # DEMAND DIRECTION
    # ========================================================

    features["demand_direction"] = np.sign(

        y.shift(1)

        -

        y.shift(2)

    )

    # ========================================================
    # SHOCK RATE
    # ========================================================

    shock_event = (

        features["absolute_shock"]

        > 2

    ).astype(int)

    features["shock_rate_30"] = (

        shock_event

        .rolling(
            30,
            min_periods=5
        )

        .mean()

    )

    features["shock_rate_90"] = (

        shock_event

        .rolling(
            90,
            min_periods=20
        )

        .mean()

    )

    # ========================================================
    # PART 3 CONTINUES...
    # ========================================================
        # ========================================================
    # DYNAMIC MEMORY FEATURES
    # ========================================================

    features["rolling_memory_strength"] = (

        y.shift(1)

        .rolling(
            90,
            min_periods=30
        )

        .apply(
            rolling_memory_strength,
            raw=False
        )

    )

    features["rolling_memory_strength"] = (

        features["rolling_memory_strength"]

        .clip(
            lower=0.001,
            upper=0.999
        )

    )

    # ========================================================
    # MEMORY HALF LIFE
    # ========================================================

    memory_strength = (
        features["rolling_memory_strength"]
        .clip(0.001, 0.999)
    )

    features["memory_half_life"] = (
        -np.log(2)
        /
        np.log(memory_strength)
    )

    features["memory_half_life"] = (
        features["memory_half_life"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # ========================================================
    # EXPONENTIAL MEMORY STATE
    # ========================================================

    memory_state = []

    state = y.iloc[0]

    alpha_series = (
        1 / features["memory_half_life"]
        .fillna(7)
    ).clip(0.01, 0.99)

    for i, value in enumerate(y.shift(1).fillna(y.iloc[0])):

        alpha = alpha_series.iloc[i]

        state = (
            alpha * value
            +
            (1 - alpha) * state
        )

        memory_state.append(state)

    features["memory_state"] = memory_state

    features["memory_residual"] = (

        y

        -

        features["memory_state"]

    )

    features["memory_ratio"] = (

        features["memory_state"]

        /

        (

            features["local_mean_30"]

            +

            1e-6

        )

    )

    # ========================================================
    # MEMORY DRIFT
    # ========================================================

    features["memory_drift"] = (

        features["memory_state"]

        -

        features["memory_state"].shift(7)

    )

    # ========================================================
    # SEASONALITY FEATURES
    # ========================================================

    features["day_of_week"] = (

        features["sales_date"]

        .dt.dayofweek

    )

    features["week_of_year"] = (

        features["sales_date"]

        .dt.isocalendar()

        .week

        .astype(int)

    )

    features["month"] = (

        features["sales_date"]

        .dt.month

    )

    features["quarter"] = (

        features["sales_date"]

        .dt.quarter

    )

    # ========================================================
    # WEEKLY CORRELATION
    # ========================================================

    features["weekly_similarity"] = (

        y.shift(1)

        .rolling(
            90,
            min_periods=30
        )

        .corr(

            y.shift(7)

        )

    )

    features["weekly_similarity"] = (

        features["weekly_similarity"]

        .fillna(0)

    )

    

    # ========================================================
    # PERSISTENCE SCORE
    # ========================================================

    features["behaviour_persistence"] = (
        (
            features["rolling_memory_strength"] *
            features["state_stability"]
        ) ** 0.5
    )

    features["behaviour_persistence"] = (

        features["behaviour_persistence"]

        .clip(0, 1)

    )

    # ========================================================
    # LOCAL DEMAND DENSITY
    # ========================================================

    features["demand_density"] = (

        (

            y.shift(1)

            >

            0

        )

        .rolling(
            30,
            min_periods=5
        )

        .mean()

    )

    # ========================================================
    # INTERMITTENCY
    # ========================================================

    last_seen = None

    gaps = []

    for i, value in enumerate(

        y.shift(1).fillna(0)

    ):

        if value > 0:

            gaps.append(0)

            last_seen = i

        else:

            if last_seen is None:

                gaps.append(np.nan)

            else:

                gaps.append(

                    i - last_seen

                )

    features["days_since_last_order"] = gaps

    # ========================================================
    # PART 4 CONTINUES...
    # ========================================================
        # ========================================================
    # DYNAMIC REGIME FORECASTABILITY INDEX (DRFI)
    # ========================================================

    drfi_features = [

        "local_cv_30",

        "demand_entropy_30",

        "absolute_shock",

        "transition_strength",

        "rolling_memory_strength",

        "weekly_similarity"

    ]

    train_features = features.iloc[:split_index]

    # --------------------------------------------
    # Min-Max normalization using TRAIN ONLY
    # --------------------------------------------

    for col in drfi_features:

        train_min = train_features[col].dropna().min()

        train_max = train_features[col].dropna().max()

        features[col + "_norm"] = (

            features[col] - train_min

        ) / (

            train_max - train_min + 1e-6

        )

        features[col + "_norm"] = (

            features[col + "_norm"]

            .clip(0, 1)

        )
    # ========================================================
    # PHASE POTENTIAL
    # ========================================================

    features["phase_potential"] = (

        0.30 * features["absolute_shock_norm"]

        +

        0.30 * features["transition_strength_norm"]

        +

        0.20 * features["demand_entropy_30_norm"]

        +

        0.20 * (1 - features["state_stability"])

    )
    features["phase_potential"] = (
        features["phase_potential"]
        .clip(0, 1)
    )

    # ========================================================
    # MEMORY COMPONENT
    # ========================================================

    features["drfi_memory"] = (

        features["rolling_memory_strength_norm"]

    )

    # ========================================================
    # STABILITY COMPONENT
    # ========================================================

    features["drfi_stability"] = (

        1

        -

        (

            0.50

            *

            features["local_cv_30_norm"]

            +

            0.50

            *

            features["demand_entropy_30_norm"]

        )

    )

    # ========================================================
    # SHOCK COMPONENT
    # ========================================================

    features["drfi_shock"] = (

        1

        -

        (

            0.60

            *

            features["absolute_shock_norm"]

            +

            0.40

            *

            features["transition_strength_norm"]

        )

    )

    # ========================================================
    # SEASONAL COMPONENT
    # ========================================================

    features["drfi_seasonality"] = (

        features["weekly_similarity_norm"]

    )

  

    # ========================================================
    # DRFI COMPONENTS ONLY
    # ========================================================

    # The final DRFI will be learned in Phase 3
    # after the forecasting model is trained.
    # Here we keep only the individual components.

   
    # ========================================================
    # BEHAVIOUR VECTOR
    # ========================================================

    phase_features = [

        "memory_short",

        "memory_weekly",

        "memory_biweekly",

        "memory_monthly",

        "local_mean_30",

        "local_std_30",

        "local_cv_30",

        "trend_30",

        "momentum_30",

        "acceleration_30",

        "demand_entropy_30",

        "absolute_shock",

        "transition_strength",

        "transition_entropy_30",
        "drfi_memory",
        "drfi_stability",
        "drfi_shock",
        "drfi_seasonality",
        

   

        "rolling_memory_strength",

        "memory_half_life",

        "memory_ratio",

        "memory_drift",

        "weekly_similarity",


        "behaviour_persistence",
        "phase_potential"


    ]

    
    # ========================================================
    # PHASE VELOCITY
    # ========================================================

    behaviour_matrix = features[
        phase_features
    ].fillna(0)

    features["phase_velocity"] = (

        np.sqrt(

            (
                behaviour_matrix
                .diff()
                ** 2
            ).sum(axis=1)

        )

    )
    # ========================================================
    # PHASE CURVATURE
    # ========================================================

    features["phase_curvature"] = (
        features["phase_velocity"]
        .diff()
    )
    phase_features.extend([
        "phase_velocity",
        "phase_curvature"
    ])
    features["behaviour_dimension"] = len(phase_features)
   

    # ========================================================
    # PHASE QUALITY METRICS
    # ========================================================

    features["complexity_score"] = (

        0.50

        *

        features["demand_entropy_30"]

        +

        0.50

        *

        features["local_cv_30"]

    )

    

    

    # ========================================================
    # PART 5 CONTINUES...
    # ========================================================
        # ========================================================
    # CLEAN NUMERIC FEATURES
    # ========================================================

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric_cols = (
        features
        .select_dtypes(include=np.number)
        .columns
    )

    train_numeric = (
        features
        .iloc[:split_index]
        [numeric_cols]
    )

    train_medians = train_numeric.median()

    features[numeric_cols] = (
        features[numeric_cols]
        .fillna(train_medians)
    )

    # ========================================================
    # CLIP EXTREME OUTLIERS
    # ========================================================

    for col in numeric_cols:

        q1 = train_numeric[col].quantile(0.01)

        q99 = train_numeric[col].quantile(0.99)

        features[col] = (
            features[col]
            .clip(q1, q99)
        )

    # ========================================================
    # STANDARDIZE USING TRAIN ONLY
    # ========================================================

    normalized_features = [

        "memory_short",
        "memory_weekly",
        "memory_biweekly",
        "memory_monthly",

        "local_mean_30",
        "local_std_30",
        "local_cv_30",

        "trend_30",

        "momentum_30",
        "acceleration_30",

        "demand_entropy_30",

        "absolute_shock",

        "transition_strength",
        "transition_entropy_30",

       

        "rolling_memory_strength",

        "memory_half_life",

        "memory_ratio",

        "memory_drift",

        "weekly_similarity",

       

        "behaviour_persistence",
        "phase_potential",
        "phase_curvature",
        "phase_velocity"


   
    ]

    feature_statistics = {}

    for col in normalized_features:

        mean = (
            train_numeric[col]
            .mean()
        )

        std = (
            train_numeric[col]
            .std()
            + 1e-6
        )

        feature_statistics[col] = {

            "mean": float(mean),

            "std": float(std)

        }

        features[col] = (

            features[col]

            - mean

        ) / std

    # ========================================================
    # SPLIT
    # ========================================================

    train_features = (
        features
        .iloc[:split_index]
        .copy()
    )

    test_features = (
        features
        .iloc[split_index:]
        .copy()
    )

    # ========================================================
    # QUALITY METRICS
    # ========================================================

    quality = {

        "rows": len(features),

        "train_rows": len(train_features),

        "test_rows": len(test_features),

        "behaviour_dimensions": len(phase_features),

        "missing_values":

            int(
                features
                .isna()
                .sum()
                .sum()
            ),

        "split_index":

            split_index

    }

    # ========================================================
    # EXPORT READY DATASET
    # ========================================================

    phase_dataset = features.copy()
    phase_dataset["created_at"] = pd.Timestamp.now()
    phase_dataset["behaviour_dimension"] = len(phase_features)
    phase_dataset = phase_dataset.replace(
        [np.inf, -np.inf],
        np.nan
    )

    phase_dataset = phase_dataset.where(
        pd.notnull(phase_dataset),
        None
    )

    # ========================================================
    # PART 6 CONTINUES...
    # ========================================================
        # ========================================================
    # RETURN
    # ========================================================

    return {

        # ==========================================
        # FULL FEATURE DATASET
        # ==========================================

        "features":
            features.to_dict("records"),

        # ==========================================
        # PHASE DATASET
        # ==========================================

        "phase_dataset":
            phase_dataset,

        # ==========================================
        # TRAIN / TEST
        # ==========================================

        "train":
            train_features,

        "test":
            test_features,

        # ==========================================
        # FEATURE LIST
        # ==========================================

        "behaviour_columns":
            phase_features,

        # ==========================================
        # NORMALIZATION PARAMETERS
        # ==========================================

        "feature_statistics":
            feature_statistics,

        # ==========================================
        # QUALITY
        # ==========================================

        "quality":
            quality
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = run_forecast_demand_phase_features(
        "Overall"
    )

    features = pd.DataFrame(
        result["features"]
    )

    print("\n")
    print("=" * 70)
    print("DEMAND PHASE FEATURE EXTRACTION")
    print("=" * 70)

    print("\nQuality")

    print(
        result["quality"]
    )

    print("\n")

    print("Behaviour Dimensions")

    print(
        len(
            result["behaviour_columns"]
        )
    )

    print("\n")

    print("Behaviour Features")

    for col in result["behaviour_columns"]:

        print(col)

    print("\n")

    print("Preview")

    print(
        features[
        [
            "sales_date",
            "demand",
            "drfi_memory",
            "drfi_stability",
            "drfi_shock",
            "state_stability"
        ]
        ].tail()
    )

    print("\n")

    print("Summary")

    print(

        features[
        [
            "drfi_memory",
            "drfi_stability",
            "drfi_shock",
            "drfi_seasonality",
          
            "phase_potential",
            "phase_velocity"
        ]
        ].describe()

    )

    print("\n")

    print("Saving dataset...")

    phase_dataset = result["phase_dataset"]

    from load_supabase import save_phase_features

    try:
        save_phase_features(phase_dataset)
        print("Saved to Supabase")
    except Exception as e:
        print(f"Supabase save failed: {e}")

    print("\nCompleted Successfully.")
 