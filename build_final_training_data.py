# ==========================================================
# BUILD FINAL TRAINING DATA
# HDGEN Feature Generation Engine
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import entropy
from scipy.stats import linregress

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime, date

from load_data import load_dealer_orders
from load_supabase import supabase
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.signal import hilbert



EPS = 1e-6
# ==========================================================
# ENTROPY
# ==========================================================

def calc_entropy(window):

    try:

        hist, _ = np.histogram(
            window,
            bins=10
        )

        hist = hist + EPS

        return float(
            entropy(hist)
        )

    except Exception:

        return np.nan


# ==========================================================
# TREND
# ==========================================================

def calc_trend(window):

    try:

        x = np.arange(len(window))

        slope, _, _, _, _ = linregress(
            x,
            window
        )

        return slope

    except Exception:

        return np.nan
# ==========================================================
# LOAD DEMAND
# ==========================================================

def load_dataset(product="Overall"):

    print("\n" + "="*70)
    print("LOADING DEMAND")
    print("="*70)

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

    daily = (

        demand

        .set_index("order_date")

        .asfreq("D", fill_value=0)

        .reset_index()

    )

    daily.rename(

        columns={

            "order_date": "sales_date",

            "total_weight_mt": "demand"

        },

        inplace=True

    )

    daily["product"] = product

    print()

    print("Rows :", len(daily))

    print()

    return daily
# ==========================================================
# GENERATE MEMORY FEATURES
# ==========================================================

def generate_memory_features(df):

    print("\n" + "=" * 70)
    print("GENERATING MEMORY FEATURES")
    print("=" * 70)

    df = df.copy()

    y = df["demand"]

    df["memory_short"] = y.rolling(3, min_periods=1).mean()

    df["memory_weekly"] = y.rolling(7, min_periods=1).mean()

    df["memory_biweekly"] = y.rolling(14, min_periods=1).mean()

    df["memory_monthly"] = y.rolling(30, min_periods=1).mean()

    # ======================================================
    # GLOBAL MEMORY ANALYSIS
    # ======================================================

    MAX_LAG = min(
        365,
        len(y) // 2
    )

    acf_values = acf(
        y,
        nlags=MAX_LAG,
        fft=False
    )

    pacf_values = pacf(
        y,
        nlags=MAX_LAG,
        method="ywm"
    )

    # ======================================================
    # MEMORY LENGTH
    # ======================================================

    limit = 1.96 / np.sqrt(len(y))

    memory_length = MAX_LAG

    for lag in range(1, len(acf_values) - 10):

        if np.all(
            np.abs(
                acf_values[lag:lag + 10]
            ) < limit
        ):

            memory_length = lag

            break

    # ======================================================
    # MEMORY HALF LIFE
    # ======================================================

    first_pacf = abs(
        pacf_values[1]
    )

    half_level = first_pacf * 0.50

    half_life = memory_length

    for lag in range(2, len(pacf_values) - 5):

        if np.all(
            np.abs(
                pacf_values[lag:lag + 5]
            ) < half_level
        ):

            half_life = lag

            break

    # ======================================================
    # ROLLING MEMORY STRENGTH
    # ======================================================

    def rolling_memory(window):

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


    df["rolling_memory_strength"] = (

        y

        .rolling(
            window=90,
            min_periods=30
        )

        .apply(
            rolling_memory,
            raw=False
        )

    )

    # ======================================================
    # MEMORY HALF LIFE
    # ======================================================

    ms = (

        df["rolling_memory_strength"]

        .clip(
            lower=0.01,
            upper=0.99
        )

    )

    df["memory_half_life"] = (

        -np.log(2)

        /

        np.log(ms)

    )

    # ======================================================
    # MEMORY STATE
    # ======================================================

    alpha = (

        1

        -

        np.exp(

            -np.log(2)

            /

            max(
                half_life,
                1
            )

        )

    )

    memory_state = [

        float(
            y.iloc[0]
        )

    ]

    for i in range(1, len(df)):

        state = (

            alpha * y.iloc[i]

            +

            (1-alpha)

            * memory_state[-1]

        )

        memory_state.append(
            float(state)
        )

    df["memory_state"] = memory_state

    # ======================================================
    # MEMORY RATIO
    # ======================================================

    rolling_mean = (

        y

        .rolling(
            30,
            min_periods=1
        )

        .mean()

    )

    df["memory_ratio"] = (

        df["memory_state"]

        /

        (

            rolling_mean

            + EPS

        )

    )

    # ======================================================
    # MEMORY DRIFT
    # ======================================================

    df["memory_drift"] = (

        df["memory_state"]

        .diff()

    )
    # ======================================================
    # MEMORY STABILITY
    # ======================================================

    df["memory_stability"] = (

        1

        /

        (

            df["memory_state"]

            .rolling(
                30,
                min_periods=10
            )

            .std()

            +

            EPS

        )

    )

    # ======================================================
    # MEMORY PERSISTENCE RATE
    # ======================================================

    df["memory_persistence_rate"] = (

        df["rolling_memory_strength"]

        .diff()

    )

    # ======================================================
    # CLEAN MISSING VALUES
    # ======================================================

    memory_columns = [

        "memory_short",
        "memory_weekly",
        "memory_biweekly",
        "memory_monthly",

        "memory_state",

        "rolling_memory_strength",
        "memory_half_life",
        "memory_ratio",
        "memory_stability",
        "memory_persistence_rate",
        "memory_drift"

    ]

    # ======================================================
    # INITIALIZE MEMORY FEATURES
    # ======================================================

    

    df["rolling_memory_strength"] = (
        df["rolling_memory_strength"]
        .fillna(0)
    )

    df["memory_half_life"] = (
        df["memory_half_life"]
        .fillna(1)
    )

    df["memory_ratio"] = (
        df["memory_ratio"]
        .fillna(1)
    )

    df["memory_drift"] = (
        df["memory_drift"]
        .fillna(0)
    )
    df["memory_stability"] = (
        df["memory_stability"]
        .fillna(0)
    )

    df["memory_persistence_rate"] = (
        df["memory_persistence_rate"]
        .fillna(0)
    )
    

    print()

    print(
        df[
            memory_columns
        ].head()
    )

    print()

    print("Memory Features Generated.")
    print("\nMemory Summary")

    print(df[memory_columns].describe().round(3))

    return df
# ==========================================================
# GENERATE TREND CHROMOSOME
# ==========================================================

def generate_trend_features(df):

    print("\n" + "=" * 70)
    print("GENERATING TREND CHROMOSOME")
    print("=" * 70)

    df = df.copy()

    y = df["demand"]

    # ======================================================
    # LOCAL STATISTICS
    # ======================================================

    history = y

    df["local_mean_30"] = (

        history

        .rolling(
            30,
            min_periods=1
        )

        .mean()

    )

    df["local_std_30"] = (

        history

        .rolling(
            30,
            min_periods=1
        )

        .std()

    )

    df["local_cv_30"] = (

        df["local_std_30"]

        /

        (

            df["local_mean_30"]

            + EPS

        )

    )

    # ======================================================
    # TREND
    # ======================================================

    df["trend_30"] = (

        history

        .rolling(
            30,
            min_periods=10
        )

        .apply(
            calc_trend,
            raw=False
        )

    )

    df["trend_90"] = (

        history

        .rolling(
            90,
            min_periods=30
        )

        .apply(
            calc_trend,
            raw=False
        )

    )

    df["trend_gap"] = (

        df["trend_30"]

        -

        df["trend_90"]

    )

    # ======================================================
    # MOMENTUM
    # ======================================================

    df["momentum_30"] = (

        history

        -

        history.shift(30)

    )

    # ======================================================
    # ACCELERATION
    # ======================================================

    df["acceleration_30"] = (

        df["momentum_30"]

        -

        df["momentum_30"].shift(1)

    )

    # ======================================================
    # INITIALIZATION
    # ======================================================

    trend_columns = [

        "local_mean_30",
        "local_std_30",
        "local_cv_30",

        "trend_30",
        "trend_90",
        "trend_gap",
        "trend_stability",
        "trend_direction_persistence",

        "momentum_30",
        "acceleration_30"

    ]


    # ======================================================
    # INITIALIZE TREND CHROMOSOME
    # ======================================================

    # Mean should start from the first observed demand
    df["local_mean_30"] = (
        df["local_mean_30"]
        .fillna(df["demand"].iloc[0])
    )

    # No variation at the beginning
    df["local_std_30"] = (
        df["local_std_30"]
        .fillna(0)
    )

    # CV is zero because std is zero
    df["local_cv_30"] = (
        df["local_cv_30"]
        .fillna(0)
    )
    
    # No trend can be estimated initially
    df["trend_30"] = (
        df["trend_30"]
        .fillna(0)
    )

    df["trend_90"] = (
        df["trend_90"]
        .fillna(0)
    )

    df["trend_gap"] = (
        df["trend_gap"]
        .fillna(0)
    )

    # No momentum at the beginning
    df["momentum_30"] = (
        df["momentum_30"]
        .fillna(0)
    )

    # No acceleration at the beginning
    df["acceleration_30"] = (
        df["acceleration_30"]
        .fillna(0)
    )
    # ======================================================
    # TREND STABILITY
    # ======================================================

    df["trend_stability"] = (

        1

        /

        (

            df["trend_30"]

            .rolling(
                30,
                min_periods=10
            )

            .std()

            +

            EPS

        )

    )

    # ======================================================
    # TREND DIRECTION PERSISTENCE
    # ======================================================

    direction = np.sign(

        df["trend_30"]

    )

    df["trend_direction_persistence"] = (

        (

            direction

            ==

            direction.shift(1)

        )

        .astype(int)

        .rolling(
            30,
            min_periods=10
        )

        .mean()

    )
    df["trend_stability"] = (
        df["trend_stability"]
        .fillna(0)
    )

    df["trend_direction_persistence"] = (
        df["trend_direction_persistence"]
        .fillna(0)
    )

    print()

    print(df[
        trend_columns
    ].head())

    print()

    print(df[
        trend_columns
    ].describe().round(3))

    print()

    print("Trend Chromosome Generated.")

    return df
# ==========================================================
# GENERATE COMPLEXITY CHROMOSOME
# ==========================================================

def generate_complexity_features(df):

    print("\n" + "=" * 70)
    print("GENERATING COMPLEXITY CHROMOSOME")
    print("=" * 70)

    df = df.copy()

    history = df["demand"]

    # ======================================================
    # ENTROPY
    # ======================================================

    df["demand_entropy_30"] = (

        history

        .rolling(
            30,
            min_periods=10
        )

        .apply(
            calc_entropy,
            raw=False
        )

    )

    df["demand_entropy_90"] = (

        history

        .rolling(
            90,
            min_periods=30
        )

        .apply(
            calc_entropy,
            raw=False
        )

    )

    # ======================================================
    # SKEWNESS
    # ======================================================

    df["skewness_30"] = (

        history

        .rolling(
            30,
            min_periods=10
        )

        .skew()

    )

    # ======================================================
    # KURTOSIS
    # ======================================================

    df["kurtosis_30"] = (

        history

        .rolling(
            30,
            min_periods=10
        )

        .kurt()

    )

    # ======================================================
    # TRANSITION ENTROPY
    # ======================================================

    demand_change = history.diff()

    df["transition_entropy_30"] = (

        demand_change

        .rolling(
            30,
            min_periods=10
        )

        .apply(
            calc_entropy,
            raw=False
        )

    )
    # ======================================================
    # COMPLEXITY DRIFT
    # ======================================================

    df["complexity_drift"] = (

        df["demand_entropy_30"]

        .diff()

    )

    # ======================================================
    # COMPLEXITY STABILITY
    # ======================================================

    df["complexity_stability"] = (

        1

        /

        (

            df["demand_entropy_30"]

            .rolling(
                30,
                min_periods=10
            )

            .std()

            +

            EPS

        )

    )

    # ======================================================
    # INITIALIZATION
    # ======================================================

    complexity_columns = [

        "demand_entropy_30",
        "demand_entropy_90",

        "skewness_30",
        "kurtosis_30",
        "complexity_drift",
        "complexity_stability",

        "transition_entropy_30"

    ]
    df["complexity_drift"] = (
        df["complexity_drift"]
        .fillna(0)
    )

    df["complexity_stability"] = (
        df["complexity_stability"]
        .fillna(0)
    )

    for col in complexity_columns:

        df[col] = (

            df[col]

            .fillna(0)

        )

    print()

    print(
        df[
            complexity_columns
        ].head()
    )

    print()

    print(
        df[
            complexity_columns
        ].describe().round(3)
    )

    print()

    print("Complexity Chromosome Generated.")

    return df
# ==========================================================
# GENERATE SHOCK CHROMOSOME
# ==========================================================

def generate_shock_features(df):

    print("\n" + "=" * 70)
    print("GENERATING SHOCK CHROMOSOME")
    print("=" * 70)

    df = df.copy()

    history = df["demand"]

    # ======================================================
    # DEMAND CHANGE
    # ======================================================

    demand_change = history.diff()

    # ======================================================
    # ABSOLUTE SHOCK
    # ======================================================

    df["absolute_shock"] = demand_change.abs()

    # ======================================================
    # RELATIVE SHOCK
    # ======================================================

    rolling_mean = (

        history

        .rolling(
            30,
            min_periods=10
        )

        .mean()

    )

    df["relative_shock"] = (

        df["absolute_shock"]

        /

        (

            rolling_mean

            + EPS

        )

    )

    # ======================================================
    # SHOCK SCORE (Historical Z-score)
    # ======================================================

    shock_mean = (

        df["absolute_shock"]

        .rolling(
            30,
            min_periods=10
        )

        .mean()

    )

    shock_std = (

        df["absolute_shock"]

        .rolling(
            30,
            min_periods=10
        )

        .std()

    )

    df["shock_score"] = (

        df["absolute_shock"]

        -

        shock_mean

    ) / (

        shock_std

        + EPS

    )

    # ======================================================
    # SHOCK EVENT
    # ======================================================

    rolling_mean = (

        df["absolute_shock"]

        .rolling(
            90,
            min_periods=30
        )

        .mean()

    )

    rolling_std = (

        df["absolute_shock"]

        .rolling(
            90,
            min_periods=30
        )

        .std()

    )

    threshold = (

        rolling_mean

        +

        rolling_std

    )

    shock_event = (

        df["absolute_shock"]

        >

        threshold

    ).astype(int)

    # ======================================================
    # SHOCK RATE
    # ======================================================

    df["shock_rate_30"] = (

        shock_event

        .rolling(
            30,
            min_periods=10
        )

        .mean()

    )
    # ======================================================
    # SHOCK PERSISTENCE
    # ======================================================

    df["shock_persistence"] = (

        df["shock_score"]

        .rolling(
            7,
            min_periods=1
        )

        .mean()

    )
    # ======================================================
    # SHOCK RECOVERY
    # ======================================================

    rolling_mean = (

        history

        .rolling(
            30,
            min_periods=1
        )

        .mean()

    )

    df["shock_recovery"] = (

        1

        -

        (

            (

                history

                -

                rolling_mean

            ).abs()

            /

            (

                rolling_mean.abs()

                +

                EPS

            )

        )

    )

    df["shock_recovery"] = (

        df["shock_recovery"]

        .clip(0,1)

    )

    # ======================================================
    # SHOCK CLUSTER DENSITY
    # ======================================================

    shock_event_binary = (

        df["shock_score"]

        >

        1

    ).astype(int)

    df["shock_cluster_density"] = (

        shock_event_binary

        .rolling(
            14,
            min_periods=1
        )

        .mean()

    )
    # ======================================================
    # SHOCK ENERGY
    # ======================================================

    df["shock_energy"] = (

        df["absolute_shock"]

        **2

    ).rolling(

        30,

        min_periods=10

    ).mean()

    # ======================================================
    # SHOCK DECAY
    # ======================================================

    df["shock_decay"] = (

        df["shock_score"]

        -

        df["shock_persistence"]

    )

    # ======================================================
    # TRANSITION DIRECTION
    # ======================================================

    df["transition_direction"] = np.select(

        [

            demand_change > 0,

            demand_change < 0

        ],

        [

            1,

            -1

        ],

        default=0

    )

    # ======================================================
    # TRANSITION VOLATILITY
    # ======================================================

    df["transition_volatility"] = (

        demand_change

        .rolling(
            30,
            min_periods=10
        )

        .std()

    )

    # ======================================================
    # INITIALIZATION
    # ======================================================

    shock_columns = [

        "absolute_shock",

        "relative_shock",

        "shock_score",

        "shock_rate_30",

        "transition_direction",

        "transition_volatility",

        "shock_persistence",

        "shock_recovery",
        "shock_energy",
        "shock_decay",

        "shock_cluster_density"

    ]
    df["shock_persistence"] = (
        df["shock_persistence"].fillna(0)
    )

    df["shock_recovery"] = (
        df["shock_recovery"].fillna(0)
    )
    df["shock_energy"] = (
        df["shock_energy"]
        .fillna(0)
    )

    df["shock_decay"] = (
        df["shock_decay"]
        .fillna(0)
    )

    df["shock_cluster_density"] = (
        df["shock_cluster_density"].fillna(0)
    )

    df["absolute_shock"] = (
        df["absolute_shock"]
        .fillna(0)
    )

    df["relative_shock"] = (
        df["relative_shock"]
        .fillna(0)
    )

    df["shock_score"] = (
        df["shock_score"]
        .fillna(0)
    )

    df["shock_rate_30"] = (
        df["shock_rate_30"]
        .fillna(0)
    )

    df["transition_direction"] = (
        df["transition_direction"]
        .fillna(0)
    )

    df["transition_volatility"] = (
        df["transition_volatility"]
        .fillna(0)
    )

    print()

    print(
        df[
            shock_columns
        ].head()
    )

    print()

    print(
        df[
            shock_columns
        ].describe().round(3)
    )

    print()

    print("Shock Chromosome Generated.")

    return df
# ==========================================================
# HDGEN OBJECTIVE GENE WEIGHTING (CRITIC)
# ==========================================================

def compute_gene_weights(df, genes):

    print("\n" + "=" * 70)
    print("HDGEN OBJECTIVE GENE WEIGHTING")
    print("=" * 70)

    scaler = MinMaxScaler()

    X = pd.DataFrame(

        scaler.fit_transform(df[genes]),

        columns=genes

    )

    # ------------------------------------------------------
    # INFORMATION CONTENT
    # ------------------------------------------------------

    std = X.std()

    # ------------------------------------------------------
    # REDUNDANCY
    # ------------------------------------------------------

    corr = X.corr().abs()

    # Make a writable NumPy copy
    corr_array = corr.to_numpy(copy=True)

    # Zero the diagonal
    np.fill_diagonal(
        corr_array,
        0
    )

    # Convert back to DataFrame
    corr = pd.DataFrame(
        corr_array,
        index=corr.index,
        columns=corr.columns
    )

    redundancy = (

        1

        -

        corr

    ).sum(axis=1)

 

    # ------------------------------------------------------
    # CRITIC SCORE
    # ------------------------------------------------------

    critic = (

        std

        *

        redundancy

    )

    if critic.sum() == 0:

        weights = pd.Series(
            1 / len(genes),
            index=genes
        )

    else:

        weights = critic / critic.sum()

    print()

    print("Learned Gene Weights")

    print(weights.round(4))

    return weights
from scipy.signal import peak_prominences
from scipy.signal import peak_widths

from sklearn.mixture import GaussianMixture
def discover_forecastability_modes(fi):

    print("\n" + "="*70)
    print("HDGEN NATURAL BEHAVIOUR BOUNDARY DETECTION")
    print("="*70)

    fi = pd.Series(fi).dropna()

    values = fi.values
    print(fi.describe())

    print()

    print(
        fi.quantile(
            [0.05,0.25,0.5,0.75,0.95]
        )
    )

    # --------------------------------------------------
    # KDE
    # --------------------------------------------------

    kde = gaussian_kde(
        values,
        bw_method=0.25
    )

    x = np.linspace(

        values.min(),

        values.max(),

        1500

    )

    density = kde(x)

    # --------------------------------------------------
    # Detect ALL Peaks
    # --------------------------------------------------

    peaks, properties = find_peaks(
        density,
        prominence=density.max() * 0.05
    )

    if len(peaks) < 2:

        print("\nOnly one behaviour mode detected.\n")

        return {

            "mode": np.ones(len(fi), dtype=int),

            "boundaries": np.array([]),

            "representative_peaks": np.array([])

        }

    # --------------------------------------------------
    # Peak Features
    # --------------------------------------------------

    heights = density[peaks]

    prominences = peak_prominences(

        density,

        peaks

    )[0]

    widths = peak_widths(

        density,

        peaks

    )[0]

    peak_features = np.column_stack([

        heights,

        prominences,

        widths

    ])
    # --------------------------------------------------
    # NORMALIZED PEAK IMPORTANCE
    # --------------------------------------------------

    height_score = (

        heights

        /

        (

            heights.max()

            + EPS

        )

    )

    prominence_score = (

        prominences

        /

        (

            prominences.max()

            + EPS

        )

    )

    width_score = (

        widths

        /

        (

            widths.max()

            + EPS

        )

    )

    peak_importance = (

        height_score

        +

        prominence_score

        +

        width_score

    ) / 3

    # --------------------------------------------------
    # GMM MODEL SELECTION (BIC)
    # --------------------------------------------------

    best_gmm = None

    best_bic = np.inf

    for k in range(

        1,

        min(5, len(peak_features)) + 1

    ):

        gmm = GaussianMixture(

            n_components=k,

            random_state=42

        )

        gmm.fit(peak_features)

        bic = gmm.bic(peak_features)

        if bic < best_bic:

            best_bic = bic

            best_gmm = gmm

    labels = best_gmm.predict(
        peak_features
    )

    print()

    print("Optimal Peak Groups :", best_gmm.n_components)

    print("Best BIC :", round(best_bic,2))

    print()

    # --------------------------------------------------
    # Major Cluster
    # --------------------------------------------------
    # --------------------------------------------------
    # REPRESENTATIVE PEAK FROM EACH GMM CLUSTER
    # --------------------------------------------------

    major_peaks = []

    print("\nRepresentative Peaks")

    for c in np.unique(labels):

        cluster_index = np.where(labels == c)[0]

        cluster_importance = peak_importance[cluster_index]

        best_peak = cluster_index[
            np.argmax(cluster_importance)
        ]

        representative_peak = peaks[best_peak]

        major_peaks.append(
            representative_peak
        )

        print(

            "Cluster",

            c + 1,

            "Peak =",

            round(x[representative_peak],4)

        )

    major_peaks = np.sort(
        np.array(major_peaks)
    )
    

    # --------------------------------------------------
    # Valley Detection
    # --------------------------------------------------

    thresholds=[]

    for i in range(

        len(major_peaks)-1

    ):

        left=major_peaks[i]

        right=major_peaks[i+1]

        valley=np.argmin(

            density[left:right]

        )

        thresholds.append(

            x[left+valley]

        )

    thresholds=np.array(thresholds)
    behaviour_boundaries = thresholds.copy()
    print()

    print("Behaviour Boundaries")

    print(thresholds)

    print()

    print("Major Behaviour Peaks")

    print(x[major_peaks])

    print()

    # --------------------------------------------------
    # Assign Modes
    # --------------------------------------------------

    mode=np.zeros(len(fi),dtype=int)

    for i,value in enumerate(values):

        mode[i]=np.sum(

            value>thresholds

        )+1

    print()

    print("Detected Modes :",mode.max()+1)

    print()
    print("Behaviour Mode Distribution")

    print(

        pd.Series(mode)

        .value_counts()

        .sort_index()

    )

    print()
    print()

    print("Behaviour Boundaries")

    print(

        np.round(

            behaviour_boundaries,

            4

        )

    )

    print()

    return {

        "mode": mode,

        "boundaries": behaviour_boundaries,

        "representative_peaks": x[major_peaks]

    }
# ==========================================================
# GENERATE SEASONALITY CHROMOSOME
# ==========================================================

def generate_seasonality_features(df):

    print("\n" + "="*70)
    print("GENERATING SEASONALITY CHROMOSOME")
    print("="*70)

    df = df.copy()

    y = df["demand"]

    # ======================================================
    # WEEKLY SIMILARITY
    # ======================================================

    df["weekly_similarity"] = (

        1

        -

        (

            (y - y.shift(7)).abs()

            /

            (

                y.abs()

                + EPS

            )

        )

    ).clip(0,1)

    # ======================================================
    # STL DECOMPOSITION
    # ======================================================

    seasonal_component = np.zeros(len(df))

    WINDOW = 84

    for i in range(len(df)):

        if i < WINDOW:

            continue

        history = y.iloc[i-WINDOW:i]

        stl = STL(
            history,
            period=7,
            robust=True
        ).fit()

        seasonal_component[i] = stl.seasonal.iloc[-1]
    # ======================================================
    # HILBERT SEASONAL PHASE
    # ======================================================

    analytic_signal = hilbert(seasonal_component)

    instantaneous_phase = np.angle(analytic_signal)

    df["seasonal_phase"] = (

        instantaneous_phase + np.pi

    ) / (

        2 * np.pi

    )



    # ======================================================
    # SEASONAL STRENGTH
    # ======================================================

    seasonal_strength = []

    for i in range(len(df)):

        start = max(0, i - 89)

        window = y.iloc[start:i+1]

        if len(window) < 30:

            seasonal_strength.append(np.nan)

            continue

        stl = STL(
            window,
            period=7,
            robust=True
        ).fit()

        window_seasonal = stl.seasonal
        resid = stl.resid
        if i == len(df) - 1:
            print("\n========== STL SEASONAL DEBUG ==========")
            print(window_seasonal.describe())
            print(window_seasonal.head())
            print(window_seasonal.tail())

        strength = 1 - (
            resid.var()
            /
            (
                (window_seasonal + resid).var()
                + EPS
            )
        )

        seasonal_strength.append(strength)

    df["seasonal_strength"] = seasonal_strength

    # ======================================================
    # SEASONAL CONSISTENCY
    # ======================================================

    seasonal_series = pd.Series(
        seasonal_component,
        index=df.index
    )

    df["seasonal_consistency"] = (

        seasonal_series

        .rolling(

            30,

            min_periods=1

        )

        .std()

    )

    df["seasonal_consistency"] = (

        1

        -

        df["seasonal_consistency"]

        /

        (

            df["seasonal_consistency"].max()

            + EPS

        )

    )
    print("\n========== SEASONAL CONSISTENCY ==========")
    print(df["seasonal_consistency"].describe())
    print(df["seasonal_consistency"].head(20))

   
    # ======================================================
    # SEASONAL DRIFT
    # ======================================================

    df["seasonal_drift"] = (

        df["seasonal_strength"]

        .diff()

    )

    # ======================================================
    # CLEAN
    # ======================================================

    cols = [

        "weekly_similarity",

        "seasonal_strength",

        "seasonal_consistency",

        "seasonal_phase",

        "seasonal_drift"

    ]
    df["seasonal_drift"] = (
        df["seasonal_drift"]
        .fillna(0)
    )

    df[cols] = (

        df[cols]

        .fillna(0)

    )

    print()

    print(df[cols].head())

    print()

    print(df[cols].describe().round(3))

    print()

    print("Seasonality Chromosome Generated.")

    return df
# ==========================================================
# GENERATE FORECASTABILITY CHROMOSOME
# ==========================================================

def generate_forecastability_features(df):

    print("\n" + "="*70)
    print("GENERATING FORECASTABILITY CHROMOSOME")
    print("="*70)

    df = df.copy()

   # ======================================================
    # MEMORY COMPONENT
    # ======================================================

    memory = df["rolling_memory_strength"].copy()

    memory = memory.fillna(memory.median())

    memory = (
        memory - memory.min()
    ) / (
        memory.max() - memory.min() + EPS
    )

    df["drfi_memory"] = memory

    # ======================================================
    # STABILITY COMPONENT
    # ======================================================

    df["drfi_stability"] = (

        1

        -

        df["local_cv_30"]

        /

        (

            df["local_cv_30"].max()

            + EPS

        )

    ).clip(0,1)

    # ======================================================
    # COMPLEXITY COMPONENT
    # ======================================================

    df["drfi_complexity"] = (

        1

        -

        df["demand_entropy_30"]

        /

        (

            df["demand_entropy_30"].max()

            + EPS

        )

    ).clip(0,1)

    # ======================================================
    # SHOCK COMPONENT
    # ======================================================

    df["drfi_shock"] = (

        1

        -

        df["relative_shock"]

        /

        (

            df["relative_shock"].max()

            + EPS

        )

    ).clip(0,1)

   
    # ======================================================
    # SEASONALITY COMPONENT (FULL CHROMOSOME)
    # ======================================================

    seasonality_genes = [

        "weekly_similarity",

        "seasonal_strength",

        "seasonal_consistency",

        "seasonal_phase",

        "seasonal_drift"

    ]

    seasonality_weights = compute_gene_weights(

        df,

        seasonality_genes

    )

    df["drfi_seasonality"] = 0

    for gene in seasonality_genes:

        df["drfi_seasonality"] += (

            seasonality_weights[gene]

            *

            df[gene]

        )

    # ======================================================
    # FORECASTABILITY INDEX
    # ======================================================

    genes = [

        "drfi_memory",

        "drfi_stability",

        "drfi_complexity",

        "drfi_shock",

        "drfi_seasonality"

    ]

    weights = compute_gene_weights(

        df,

        genes

    )

    df["forecastability_index"] = 0

    for gene in genes:

        df["forecastability_index"] += (

            weights[gene]

            *

            df[gene]

        )

    # ======================================================
    # NATURAL BEHAVIOUR MODES
    # ======================================================

    result = discover_forecastability_modes(

        df["forecastability_index"]

    )

    df["forecastability_mode"] = result["mode"]

    print()

    print("Representative Peaks")

    print(

        np.round(

            result["representative_peaks"],

            4

        )

    )

    print()

    print("Behaviour Boundaries")

    print(

        np.round(

            result["boundaries"],

            4

        )

    )

    print()

    

    cols = [

        "drfi_memory",

        "drfi_stability",

        "drfi_complexity",

        "drfi_shock",

        "drfi_seasonality",

        "forecastability_index",

        "forecastability_mode"

    ]
    print()

    print(df[cols].head())

    print()

    print(df[
        [

            "forecastability_index"

        ]

    ].describe().round(3))

    print()

    print("Forecastability Chromosome Generated.")

    return df
# ==========================================================
# SAVE HDGEN BEHAVIOUR GENOME
# ==========================================================

def save_hdgen_behaviour_genome(df):

    print("\n" + "=" * 70)
    print("SAVING HDGEN BEHAVIOUR GENOME")
    print("=" * 70)

    records = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)
    # Convert Timestamp to string for JSON serialization
    records["sales_date"] = (
        pd.to_datetime(records["sales_date"])
        .dt.strftime("%Y-%m-%d")
    )

    records = records.to_dict(
        orient="records"
    )

    batch_size = 500

    for i in range(0, len(records), batch_size):

        batch = records[i:i + batch_size]

        (
            supabase
            .table("hdgen_behaviour_genome")
            .upsert(
                batch,
                on_conflict="product,sales_date"
            )
            .execute()
        )

    print()

    print("Rows Saved :", len(records))

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    PRODUCT = "Overall"

    print("\n" + "=" * 70)
    print("HDGEN FINAL TRAINING DATA GENERATION")
    print("=" * 70)

    # ------------------------------------------------------
    # Load Demand
    # ------------------------------------------------------

    df = load_dataset(PRODUCT)

    print("\nDataset Preview")
    print(df.head())

    print("\nRows :", len(df))
    print("Columns :", len(df.columns))

    # ==========================================================
    # MEMORY FEATURES
    # ==========================================================

    df = generate_memory_features(df)

    print()

    print(df.head())

    print()

    print("Columns :", len(df.columns))

    # ==========================================================
    # TREND CHROMOSOME
    # ==========================================================

    df = generate_trend_features(df)

    print()

    print(df.head())

    print()

    print("Columns :", len(df.columns))

    # ==========================================================
    # COMPLEXITY CHROMOSOME
    # ==========================================================

    df = generate_complexity_features(df)

    print()

    print(df.head())

    print()

    print("Columns :", len(df.columns))
    # ==========================================================
    # SHOCK CHROMOSOME
    # ==========================================================

    df = generate_shock_features(df)

    print()

    print(df.head())

    print()

    print("Columns :", len(df.columns))

    

    # ==========================================================
    # SEASONALITY CHROMOSOME
    # ==========================================================

    df = generate_seasonality_features(df)

    print()

    print(df.head())

    print()

    print("Columns :", len(df.columns))
    print()

    

    print()
    


    # ==========================================================
    # VALIDATE HDGEN GENOME
    # ==========================================================

    expected = {

        "sales_date",
        "product",
        "demand",

        # Memory
        "memory_short",
        "memory_weekly",
        "memory_biweekly",
        "memory_monthly",
        "memory_state",
        "rolling_memory_strength",
        "memory_half_life",
        "memory_ratio",
        "memory_drift",
        "memory_stability",
        "memory_persistence_rate",

        # Trend
        "local_mean_30",
        "local_std_30",
        "local_cv_30",
        "trend_30",
        "trend_90",
        "trend_gap",
        "momentum_30",
        "acceleration_30",
        "trend_stability",
        "trend_direction_persistence",

        # Complexity
        "demand_entropy_30",
        "demand_entropy_90",
        "skewness_30",
        "kurtosis_30",
        "transition_entropy_30",
        "complexity_drift",
        "complexity_stability",

        # Shock
        "absolute_shock",
        "relative_shock",
        "shock_score",
        "shock_rate_30",
        "transition_direction",
        "transition_volatility",
        "shock_persistence",
        "shock_recovery",
        "shock_cluster_density",
        "shock_energy",
        "shock_decay",

        # Seasonality
        "weekly_similarity",
        "seasonal_strength",
        "seasonal_consistency",
        "seasonal_phase",
        "seasonal_drift"

    }

    missing = expected - set(df.columns)
    extra = set(df.columns) - expected

    if missing or extra:

        raise ValueError(
            f"""
    HDGEN Schema Validation Failed

    Missing Columns:
    {missing}

    Extra Columns:
    {extra}
    """
        )

    print("✓ HDGEN schema validation passed.")

    save_hdgen_behaviour_genome(df)
    print("\n" + "="*70)
    print("HDGEN Behaviour Genome")
    print("="*70)

    print("Memory Genes      : 11")
    print("Trend Genes       : 10")
    print("Complexity Genes  : 7")
    print("Shock Genes       : 11")
    print("Seasonality Genes : 5")

    print("-"*70)

    print("Total Behaviour Genes : 44")

    print("Rows Saved :", len(df))