import pandas as pd
import numpy as np

from scipy.stats import wilcoxon
from scipy import stats

from load_supabase import supabase
def load_forecasts():

    baseline = (
        supabase
        .table("hdgen_baseline_forecasts")
        .select("*")
        .eq("model_name", "CatBoost")
        .execute()
    )

    baseline = pd.DataFrame(
        baseline.data
    )

    hdgen = (
        supabase
        .table("hdgen_final_demand_forecasts")
        .select("*")
        .eq("forecast_type", "TEST")
        .execute()
    )

    hdgen = pd.DataFrame(
        hdgen.data
    )

    baseline["forecast_date"] = pd.to_datetime(
        baseline["forecast_date"]
    )

    hdgen["forecast_date"] = pd.to_datetime(
        hdgen["forecast_date"]
    )

    merged = baseline.merge(

        hdgen[
            [
                "forecast_date",
                "predicted_demand"
            ]
        ],

        on="forecast_date",

        suffixes=(
            "_baseline",
            "_hdgen"
        )

    )

    return merged
def run_wilcoxon(df):

    baseline_error = np.abs(

        df["actual_demand"]

        -

        df["predicted_demand_baseline"]

    )

    hdgen_error = np.abs(

        df["actual_demand"]

        -

        df["predicted_demand_hdgen"]

    )

    stat, p = wilcoxon(

        baseline_error,

        hdgen_error,

       alternative="two-sided"

    )

    return stat, p
def run_diebold_mariano(
    df,
    horizon=1,
    power=2
):

    """
    Diebold-Mariano Test

    H0:
        Both forecasting models have equal predictive accuracy.

    H1:
        HDGEN has significantly better predictive accuracy.
    """

    # Forecast errors

    e1 = (
        df["actual_demand"]
        -
        df["predicted_demand_baseline"]
    ).values

    e2 = (
        df["actual_demand"]
        -
        df["predicted_demand_hdgen"]
    ).values

    # Loss differential

    d = (
        np.abs(e1) ** power
        -
        np.abs(e2) ** power
    )

    T = len(d)

    d_bar = np.mean(d)

    # ---------- Long-run variance (Newey-West) ----------

    gamma0 = np.var(
        d,
        ddof=1
    )

    long_run_var = gamma0

    # For h=1 this loop is skipped automatically.
    for lag in range(1, horizon):

        gamma = np.cov(

            d[:-lag],
            d[lag:],

            ddof=1

        )[0,1]

        weight = 1 - lag / horizon

        long_run_var += 2 * weight * gamma

    # DM statistic

    dm_stat = d_bar / np.sqrt(
        long_run_var / T
    )

    # Harvey-Leybourne-Newbold correction

    correction = np.sqrt(

        (
            T + 1 - 2*horizon
            +
            horizon*(horizon-1)/T
        )
        /
        T

    )

    dm_stat *= correction

    # Two-sided p-value

    p_value = 2 * (
        1 -
        stats.t.cdf(
            abs(dm_stat),
            df=T-1
        )
    )

    return dm_stat, p_value

if __name__ == "__main__":

    df = load_forecasts()

    print("=" * 70)
    print("HDGEN STATISTICAL VALIDATION")
    print("=" * 70)

    print("\nModels Compared")
    print("-" * 40)
    print("Baseline : CatBoost")
    print("Proposed : HDGEN")

    print("\nObservations :", len(df))

    # ==========================================================
    # WILCOXON TEST
    # ==========================================================

    stat, p = run_wilcoxon(df)

    print("\n")
    print("=" * 70)
    print("WILCOXON SIGNED-RANK TEST")
    print("=" * 70)

    print("Statistic :", round(stat, 4))
    print("P-value   :", round(p, 6))

    print()
    print("Null Hypothesis (H0)")
    print("--------------------")
    print("Median forecast errors are equal.")

    print()

    if p < 0.05:

        print("Decision")
        print("--------")
        print("Reject H0")

        print()

        print("Conclusion")
        print("----------")
        print(
            "HDGEN produces significantly "
            "lower forecast errors than "
            "the CatBoost baseline."
        )

    else:

        print("Decision")
        print("--------")
        print("Fail to Reject H0")

        print()

        print("Conclusion")
        print("----------")
        print(
            "No statistically significant "
            "difference was detected."
        )

    # ==========================================================
    # DIEBOLD-MARIANO TEST
    # ==========================================================

    dm_stat, dm_p = run_diebold_mariano(df)

    print("\n")
    print("=" * 70)
    print("DIEBOLD-MARIANO TEST")
    print("=" * 70)

    print("DM Statistic :", round(dm_stat, 4))
    print("P-value      :", round(dm_p, 6))

    print()

    print("Null Hypothesis (H0)")
    print("--------------------")
    print("Both forecasting models have equal predictive accuracy.")

    print()

    if dm_p < 0.05:

        print("Decision")
        print("--------")
        print("Reject H0")

        print()

        print("Conclusion")
        print("----------")
        print(
            "HDGEN provides statistically "
            "significantly better forecasting accuracy."
        )

    else:

        print("Decision")
        print("--------")
        print("Fail to Reject H0")

        print()

        print("Conclusion")
        print("----------")
        print(
            "No statistically significant difference "
            "between HDGEN and the CatBoost baseline."
        )

    # ==========================================================
    # FINAL CONCLUSION
    # ==========================================================

    print("\n")
    print("=" * 70)
    print("FINAL STATISTICAL CONCLUSION")
    print("=" * 70)

    alpha = 0.05

    print(f"Significance Level (α): {alpha}")

    print()

    print(f"Wilcoxon p-value         : {p:.6f}")
    print(f"Diebold-Mariano p-value : {dm_p:.6f}")

    print()

    if (p < alpha) and (dm_p < alpha):

        print("Overall Decision")
        print("----------------")
        print(
            "Both statistical tests reject the null hypothesis."
        )

        print()

        print(
            "The forecasting improvement achieved by HDGEN "
            "over the CatBoost baseline is statistically "
            "significant and is unlikely to have occurred "
            "by random chance."
        )

    elif (p < alpha):

        print("Overall Decision")
        print("----------------")
        print(
            "Wilcoxon is significant, "
            "but Diebold-Mariano is not."
        )

    elif (dm_p < alpha):

        print("Overall Decision")
        print("----------------")
        print(
            "Diebold-Mariano is significant, "
            "but Wilcoxon is not."
        )

    else:

        print("Overall Decision")
        print("----------------")
        print(
            "Neither statistical test found "
            "a statistically significant difference."
        )