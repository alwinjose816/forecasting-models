from demand_states import run_demand_states

import pandas as pd
from umap import UMAP


def run_demand_regimes(product="Overall"):

    # ==========================================
    # LOAD STATES + REGIMES
    # ==========================================

    result = run_demand_states(product)

    if "error" in result:
        return result

    df = pd.DataFrame(
        result["states"]
    )

    latent_cols = [
        c
        for c in df.columns
        if c.startswith("z")
    ]

    if len(latent_cols) == 0:
        return {
            "error":
            "No latent states found"
        }

    if "state" not in df.columns:
        return {
            "error":
            "No regime labels found"
        }

    Z = df[latent_cols]
    profiles = pd.DataFrame(
        result["state_profiles"]
    ).T
    state_characteristics = {}
   

    for state in profiles.index:

        state_id = int(float(state))

        feature_names = (
            profiles.loc[state]
            .abs()
            .sort_values(
                ascending=False
            )
            .head(5)
            .index
        )

        state_characteristics[
            state_id
        ] = {

            "top_features":
            feature_names.tolist(),

            "values": {

                f: float(
                    profiles.loc[state, f]
                )

                for f in feature_names
            }
        }

       
    state_names = {}
    state_name_reason = {}

    for state in profiles.index:
        state_id = int(float(state))

        row = profiles.loc[state]

        name = "Mixed Demand"
        reason = "No dominant pattern"

        if (
            "zero_demand_ratio_90" in row.index
            and row["zero_demand_ratio_90"] > 0.30
        ):
            name = "Intermittent Demand"

            reason = (
                f"zero_demand_ratio_90="
                f"{row['zero_demand_ratio_90']:.2f}"
            )

        elif (
            "shock_score" in row.index
            and row["shock_score"] > 1.50
        ):
            name = "Demand Shock"

            reason = (
                f"shock_score="
                f"{row['shock_score']:.2f}"
            )

        elif (
            "trend_90" in row.index
            and row["trend_90"] < -1
        ):
            name = "Declining Demand"

            reason = (
                f"trend_90="
                f"{row['trend_90']:.2f}"
            )

        elif (
            "trend_90" in row.index
            and row["trend_90"] > 1
        ):
            name = "Growth Demand"

            reason = (
                f"trend_90="
                f"{row['trend_90']:.2f}"
            )

      
        elif (
            "rolling_memory_strength" in row.index
            and row["rolling_memory_strength"] > 0.50
        ):
            name = "Stable Demand"

            reason = (
                f"rolling_memory_strength="
                f"{row['rolling_memory_strength']:.2f}"
            )

     

       

        state_names[
            state_id
        ] = name
        state_name_reason[
            state_id
        ] = reason

    # ==========================================
    # UMAP VISUALIZATION
    # ==========================================

    umap_model = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )

    embedding = (
        umap_model
        .fit_transform(Z)
    )

    df["umap1"] = embedding[:, 0]
    df["umap2"] = embedding[:, 1]

    # ==========================================
    # REGIME SUMMARY
    # ==========================================

    regime_summary = (

        df

        .groupby("state")

        .agg(

            demand_mean=("demand", "mean"),
            demand_std=("demand", "std"),
            demand_min=("demand", "min"),
            demand_max=("demand", "max")

        )

        .round(3)

        .reset_index()

    )

    # ==========================================
    # RETURN
    # ==========================================

    return {

        "regimes":
        df.to_dict("records"),

        "best_k":
        result["best_k"],

        "silhouette_score":
        result["silhouette_score"],

        "davies_bouldin_score":
        result["davies_bouldin_score"],

        "variance":
        result["variance"],

        "top_features":
        result["top_features"],

        "state_counts":
        result["state_counts"],

        "state_profiles":
        result["state_profiles"],

        "cluster_centers":
        result["cluster_centers"],

        "regime_summary":
        regime_summary.to_dict(
            orient="records"
        ),

        "state_characteristics":
        state_characteristics,

        "state_names":
        state_names,

        "state_name_reason":
        state_name_reason
    }


if __name__ == "__main__":

    result = run_demand_regimes(
        "Overall"
    )
    print("\nState Characteristics:")

    for k, v in result["state_characteristics"].items():

        print(f"\nState {k}")

        print(
            v["top_features"]
        )

    print(
        "\nBest K:",
        result["best_k"]
    )

    print(
        "\nSilhouette:",
        result["silhouette_score"]
    )

    print(
        "\nDavies-Bouldin:",
        result["davies_bouldin_score"]
    )

    print(
        "\nState Counts:"
    )

    print(
        result["state_counts"]
    )
    print("\nState Names:")

    for k, v in result["state_names"].items():
        print(f"State {k}: {v}")

    print("\nState Name Reasons:")

    for k, v in result["state_name_reason"].items():
        print(f"State {k}: {v}")