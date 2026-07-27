import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.stattools import acf

from load_supabase import supabase
# ==========================================================
# LOAD HDGEN BEHAVIOUR GENOME
# ==========================================================

def load_hdgen_behaviour_genome():

    print("\n" + "=" * 70)
    print("LOADING HDGEN BEHAVIOUR GENOME")
    print("=" * 70)

    response = (
        supabase
        .table("hdgen_behaviour_genome")
        .select("*")
        .order("sales_date")
        .execute()
    )

    df = pd.DataFrame(response.data)
    # ==========================================================
    # CONVERT ALL GENES TO NUMERIC
    # ==========================================================

    for gene in ALL_GENES:

        df[gene] = pd.to_numeric(

            df[gene],

            errors="coerce"

        )

    df["sales_date"] = pd.to_datetime(df["sales_date"])
    print()

    print(df[ALL_GENES].dtypes)

    print()

    print("Rows :", len(df))

    print("Columns :", len(df.columns))

    return df
MEMORY_GENES = [

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
    "memory_persistence_rate"

]

TREND_GENES = [

    "local_mean_30",
    "local_std_30",
    "local_cv_30",

    "trend_30",
    "trend_90",
    "trend_gap",

    "momentum_30",
    "acceleration_30",

    "trend_stability",
    "trend_direction_persistence"

]

COMPLEXITY_GENES = [

    "demand_entropy_30",
    "demand_entropy_90",

    "skewness_30",
    "kurtosis_30",

    "transition_entropy_30",

    "complexity_drift",
    "complexity_stability"

]

SHOCK_GENES = [

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
    "shock_decay"

]

SEASONALITY_GENES = [

    "weekly_similarity",

    "seasonal_strength",

    "seasonal_consistency",

    "seasonal_phase",

    "seasonal_drift"

]

ALL_GENES = (

    MEMORY_GENES
    + TREND_GENES
    + COMPLEXITY_GENES
    + SHOCK_GENES
    + SEASONALITY_GENES

)
# ==========================================================
# VARIANCE SCORE
# ==========================================================

def compute_variance_score(df):

    variance = {}

    for gene in ALL_GENES:

        variance[gene] = df[gene].var()
    print("\nRAW VARIANCE")

    

    variance = pd.Series(variance)

    # Log-transform to reduce the effect of extreme values
    variance = np.log1p(variance)

    # Then normalize
    variance = (
        variance - variance.min()
    ) / (
        variance.max() - variance.min() + 1e-9
    )

    return variance
# ==========================================================
# MISSING SCORE
# ==========================================================

def compute_missing_score(df):

    missing = {}

    for gene in ALL_GENES:

        missing[gene] = (

            1

            -

            df[gene]

            .isna()

            .mean()

        )

    return pd.Series(missing)
# ==========================================================
# STABILITY SCORE
# ==========================================================

def compute_stability_score(df):

    stability = {}

    for gene in ALL_GENES:

        rolling_std = (

            df[gene]

            .rolling(
                30,
                min_periods=10
            )

            .std()

        )

        score = (

            1

            /

            (

                rolling_std.mean()

                + 1e-9

            )

        )

        stability[gene] = score

    stability = pd.Series(stability)

    stability = (

        stability

        - stability.min()

    ) / (

        stability.max()

        - stability.min()

        + 1e-9

    )

    return stability
# ==========================================================
# STABILITY SCORE
# ==========================================================

def compute_stability_score(df):

    stability = {}

    for gene in ALL_GENES:

        rolling_mean = (
            df[gene]
            .rolling(30, min_periods=10)
            .mean()
        )

        rolling_std = (
            df[gene]
            .rolling(30, min_periods=10)
            .std()
        )

        cv = rolling_std / (rolling_mean.abs() + 1e-9)

        score = 1 / (cv.mean() + 1e-9)

        stability[gene] = score

    stability = pd.Series(stability)

    stability = np.log1p(stability)

    stability = (
        stability - stability.min()
    ) / (
        stability.max() - stability.min() + 1e-9
    )

    return stability
# ==========================================================
# PERSISTENCE SCORE
# ==========================================================

def compute_persistence_score(df):

    persistence = {}

    for gene in ALL_GENES:

        x = df[gene].ffill().bfill()

        acf_values = acf(
            x,
            nlags=30,
            fft=True
        )

        persistence[gene] = np.mean(np.abs(acf_values[1:]))

    persistence = pd.Series(persistence)

    persistence = (
        persistence - persistence.min()
    ) / (
        persistence.max() - persistence.min() + 1e-9
    )

    return persistence
# ==========================================================
# MUTUAL INFORMATION
# ==========================================================

def compute_mutual_information(df):

    mi = {}

    for gene in ALL_GENES:

        x = df[gene].shift(1).dropna().values.reshape(-1,1)

        y = df[gene].iloc[1:].values

        score = mutual_info_regression(
            x,
            y,
            random_state=42
        )[0]

        mi[gene] = score

    mi = pd.Series(mi)

    mi = (
        mi - mi.min()
    ) / (
        mi.max() - mi.min() + 1e-9
    )

    return mi
# ==========================================================
# REDUNDANCY SCORE
# ==========================================================

def compute_redundancy_score(df):

    corr = df[ALL_GENES].corr().abs()

    redundancy = {}

    for gene in ALL_GENES:

        others = corr.loc[gene].drop(gene)

        redundancy[gene] = others.mean()

    redundancy = pd.Series(redundancy)

    redundancy = 1 - redundancy

    redundancy = (
        redundancy - redundancy.min()
    ) / (
        redundancy.max() - redundancy.min() + 1e-9
    )

    return redundancy
# ==========================================================
# FORECASTABILITY SCORE
# ==========================================================

def compute_forecastability_score(df):

    score = {}

    for gene in ALL_GENES:

        x = df[gene].shift(1)

        y = df[gene]

        r = x.corr(y)

        score[gene] = abs(r)

    score = pd.Series(score).fillna(0)

    return score
# ==========================================================
# CRITIC OBJECTIVE WEIGHTING
# ==========================================================

def compute_critic_weights(quality):

    metrics = [

        "variance",

        "stability",

        "persistence",

        "mutual_information",

        "redundancy",

        "forecastability"

    ]

    X = quality[metrics].copy()

    X = (
        X - X.min()
    ) / (
        X.max() - X.min() + 1e-9
    )

    std = X.std()

    corr = X.corr()

    information = {}

    for col in metrics:

        conflict = (1 - corr[col]).sum()

        information[col] = std[col] * conflict

    information = pd.Series(information)

    weights = information / information.sum()

    print()

    print("=" * 70)
    print("CRITIC GENE QUALITY WEIGHTS")
    print("=" * 70)

    print(weights.round(4))

    return weights
# ==========================================================
# GENE QUALITY INDEX
# ==========================================================

def compute_gene_quality_index(
    quality,
    weights
):

    quality["GQI"] = 0

    for metric in weights.index:

        quality["GQI"] += (

            quality[metric]

            * weights[metric]

        )

    return quality
# ==========================================================
# ADAPTIVE FUNDAMENTAL GENE DISCOVERY
# ==========================================================

def discover_fundamental_genes(quality):

    # ---------------------------------------------
    # Sort genes by quality
    # ---------------------------------------------

    quality = quality.sort_values(
        "GQI",
        ascending=False
    ).reset_index()

    quality.rename(
        columns={"index": "gene_name"},
        inplace=True
    )

    # ---------------------------------------------
    # Ranking
    # ---------------------------------------------

    quality["rank"] = np.arange(
        1,
        len(quality) + 1
    )

    # ---------------------------------------------
    # Natural gap detection
    # ---------------------------------------------

    quality["gap"] = (

        quality["GQI"]

        -

        quality["GQI"].shift(-1)

    ).fillna(0)

    # ---------------------------------------------
    # Largest gap
    # ---------------------------------------------

    # ---------------------------------------------
    # Adaptive HDGEN Boundary
    # ---------------------------------------------

    candidate = quality[

        quality["GQI"]

        >=

        quality["GQI"].median()

    ]

    boundary = candidate["gap"].idxmax()

    quality["selected"] = False

    quality.loc[
        :boundary,
        "selected"
    ] = True

    print()

    print("=" * 70)
    print("HDGEN FUNDAMENTAL GENE DISCOVERY")
    print("=" * 70)

    print()

    print(
        "Largest Quality Gap :",
        round(
            quality.loc[
                boundary,
                "gap"
            ],
            4
        )
    )

    print()

    print(
        "Selected Genes :",
        quality["selected"].sum()
    )

    print()

    print(

        quality[
            [

                "rank",

                "gene_name",

                "GQI",

                "gap",

                "selected"

            ]

        ]

    )

    return quality
# ==========================================================
# SAVE GENE QUALITY
# ==========================================================

def save_hdgen_gene_quality(quality):

    print()
    print("=" * 70)
    print("SAVING HDGEN GENE QUALITY")
    print("=" * 70)

    chromosome_map = {}

    for g in MEMORY_GENES:
        chromosome_map[g] = "Memory"

    for g in TREND_GENES:
        chromosome_map[g] = "Trend"

    for g in COMPLEXITY_GENES:
        chromosome_map[g] = "Complexity"

    for g in SHOCK_GENES:
        chromosome_map[g] = "Shock"

    for g in SEASONALITY_GENES:
        chromosome_map[g] = "Seasonality"

    quality["chromosome"] = quality["gene_name"].map(chromosome_map)
    quality = quality.replace({
        np.nan: None
    })
    quality = quality.rename(columns={

        "GQI": "gqi"

    })

    records = quality.to_dict(orient="records")
    extra_columns = [

        "pareto_front",
        "crowding_distance",

        "pearson_corr",
        "spearman_corr",

        "shap_importance",

        "selected_gqi",
        "selected_pareto",
        "selected_correlation",
        "selected_shap",

        "selection_score",

        "final_selected"

    ]

    for c in extra_columns:

        if c not in quality.columns:

            quality[c] = None

    (
        supabase
        .table("hdgen_gene_quality")
        .upsert(
            records,
            on_conflict="gene_name"
        )
        .execute()
    )

    print()

    print("Saved :", len(records), "Genes")
if __name__ == "__main__":

    

    df = load_hdgen_behaviour_genome()

    quality = pd.DataFrame({

        "variance": compute_variance_score(df),
        "missing": compute_missing_score(df),
        "stability": compute_stability_score(df),
        "persistence": compute_persistence_score(df),
        "mutual_information": compute_mutual_information(df),
        "redundancy": compute_redundancy_score(df),
        "forecastability": compute_forecastability_score(df)

    })

    weights = compute_critic_weights(quality)

    quality = compute_gene_quality_index(
        quality,
        weights
    )

    quality = discover_fundamental_genes(
        quality
    )
    save_hdgen_gene_quality(
        quality
    )