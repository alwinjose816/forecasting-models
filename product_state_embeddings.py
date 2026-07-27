from load_data import load_dealer_orders

from demand_transitions import (
    run_demand_transitions
)

from demand_memory import (
    run_demand_memory
)
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from umap import UMAP
from sklearn.metrics.pairwise import (
    cosine_similarity
)


MAX_STATES = 5


def run_product_embeddings():

    # ==========================================
    # LOAD PRODUCTS
    # ==========================================

    df = load_dealer_orders()

    products = sorted(
        df["product_code"]
        .dropna()
        .unique()
    )

    rows = []
    skipped_products = []


    # ==========================================
    # BUILD PRODUCT EMBEDDINGS
    # ==========================================

    for product in products:

        try:

            transition_result = run_demand_transitions(
                product
            )

            memory_result = (
                run_demand_memory(
                    product
                )
            )
            if "error" in transition_result:

                skipped_products.append({
                    "product": product,
                    "error": transition_result["error"]
                })
                continue

            if "error" in memory_result:

                skipped_products.append({
                    "product": product,
                    "error": memory_result["error"]
                })
                continue
            required_keys = [

                "state_frequency",
                "transition_matrix",
                "expected_duration"

            ]

            if not all(
                k in transition_result
                for k in required_keys
            ):

                skipped_products.append({
                    "product": product,
                    "error": "Missing transition keys"
                })

                continue

           

            row = {
                "product": product
            }

            # ==================================
            # STATE FREQUENCIES
            # ==================================

            state_freq = transition_result.get(
                "state_frequency",
                {}
            )

            for state in range(MAX_STATES):

                row[
                    f"freq_state_{state}"
                ] = float(
                    state_freq.get(
                        state,
                        0
                    )
                )

            # ==================================
            # TRANSITION MATRIX
            # ==================================

            tm = pd.DataFrame(
                transition_result[
                    "transition_matrix"
                ]
            )

            for i in range(MAX_STATES):

                for j in range(MAX_STATES):

                    value = 0

                    if (
                        i in tm.index
                        and
                        j in tm.columns
                    ):
                        value = float(
                            tm.loc[i, j]
                        )

                    row[
                        f"P{i}{j}"
                    ] = value

            # ==================================
            # EXPECTED DURATION
            # ==================================

            durations = (
                transition_result.get(
                    "expected_duration",
                    {}
                )
            )

            for state in range(MAX_STATES):

                row[
                    f"duration_{state}"
                ] = float(
                    durations.get(
                        state,
                        0
                    )
                )

            # ==================================
            # STABILITY SCORE
            # ==================================

            duration_values = [

                row[f"duration_{i}"]

                for i in range(MAX_STATES)

                if row[f"duration_{i}"] > 0
            ]

            row["stability_score"] = (

                np.mean(duration_values)

                if len(duration_values) > 0

                else 0
            )

            # ==================================
            # MEMORY PROFILE
            # ==================================

            row["memory_strength"] = float(
                memory_result.get(
                    "ms",
                    0
                )
            )

            row["memory_length"] = float(
                memory_result.get(
                    "ml",
                    0
                )
            )

            row["memory_half_life"] = float(
                memory_result.get(
                    "ml50",
                    0
                )
            )

            row["seasonality_strength"] = float(
                memory_result.get(
                    "ss",
                    0
                )
            )

            row["entropy"] = float(
                memory_result.get(
                    "entropy",
                    0
                )
            )
            # ==================================
            # INTERMITTENCY + TREND PROFILE
            # ==================================

            row["zero_demand_ratio"] = float(
                memory_result.get(
                    "zero_demand_ratio_90",
                    0
                )
            )

            row["trend_score"] = float(
                memory_result.get(
                    "trend_90",
                    0
                )
            )
            print(
                product,
                "STAB=",
                row["stability_score"],
                "MEM=",
                row["memory_strength"],
                "SEAS=",
                row["seasonality_strength"]
            )
            trend = row["trend_score"]

            memory = row["memory_strength"]

            intermittent = row[
                "zero_demand_ratio"
            ]

            if intermittent > 0.70:

                lifecycle = "Intermittent"

            elif trend > 0.05:

                lifecycle = "Growth"

            elif trend < -0.05:

                lifecycle = "Decline"

            elif memory > 0.30:

                lifecycle = "Maturity"

            else:

                lifecycle = "Introduction"

            row["product_lifecycle"] = (
                lifecycle
            )
            # ==================================
            # CURRENT STATE
            # ==================================

            if row["memory_strength"] > 0.40:

                current_state = "Expansion"

            elif row["seasonality_strength"] > 0.60:

                current_state = "Seasonal"

            elif row["entropy"] > 0.60:

                current_state = "Shock"

            elif row["trend_score"] < -0.10:

                current_state = "Contraction"

            else:

                current_state = "Stable"

            row["current_state"] = current_state


            # ==================================
            # NEXT STATE
            # ==================================

            if current_state == "Expansion":

                next_state = "Stable Growth"

            elif current_state == "Seasonal":

                next_state = "Seasonal Rise"

            elif current_state == "Shock":

                next_state = "Recovery"

            elif current_state == "Contraction":

                next_state = "Decline"

            else:

                next_state = "Stable"

            row["next_state"] = next_state
            print(
                product,
                "ZERO=",
                row["zero_demand_ratio"],
                "TREND=",
                row["trend_score"],
                "MEM=",
                row["memory_strength"],
                "LIFE=",
                row["product_lifecycle"]
            )
                                

           

            rows.append(row)
      

        except Exception as e:

            skipped_products.append({
                "product": product,
                "error": str(e)
            })

            continue

    # ==========================================
    # DATAFRAME
    # ==========================================

    embeddings_df = pd.DataFrame(
        rows
    )

    embeddings_df = (
        embeddings_df
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    if len(embeddings_df) < 3:

        return {
            "error": "Not enough products",
            "skipped_products": skipped_products
        }
    # ==========================================
    # FEATURES
    # ==========================================

    numeric_df = (
        embeddings_df
        .select_dtypes(
            include=[np.number]
        )
    )

    embedding_dimensions = (
        numeric_df.shape[1]
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        numeric_df
    )
   

    pca = PCA(
        n_components=0.90,
        random_state=42
    )

    X_reduced = pca.fit_transform(
        X_scaled
    )

    explained_variance = float(
        pca.explained_variance_ratio_.sum()
    )

    pca_components = int(
        X_reduced.shape[1]
    )


   
    # ==========================================
    # BEST K
    # ==========================================

    best_k = 2
    best_score = float("-inf")

    max_k = min(
        10,
        len(embeddings_df) - 1
    )

    for k in range(
        2,
        max_k + 1
    ):

        try:

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=20
            )

            labels = (
                model.fit_predict(
                    X_reduced
                )
            )

            score = (
                silhouette_score(
                    X_reduced,
                    labels
                )
            )

            if score > best_score:

                best_score = score
                best_k = k

        except Exception:

            pass

    # ==========================================
    # FINAL MODEL
    # ==========================================

    final_model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20
    )

    labels = final_model.fit_predict(
        X_reduced
    )

    embeddings_df[
        "product_cluster"
    ] = labels

    # ==========================================
    # UMAP
    # ==========================================

    umap_model = UMAP(
        n_neighbors=10,
        min_dist=0.1,
        random_state=42
    )

    embedding = (
        umap_model.fit_transform(
            X_reduced
        )
    )

    embeddings_df[
        "umap1"
    ] = embedding[:, 0]

    embeddings_df[
        "umap2"
    ] = embedding[:, 1]
    similarity_matrix = (
        cosine_similarity(
            X_reduced
        )
    )
    product_list = (
        embeddings_df["product"]
        .tolist()
    )

    product_similarity = {

        product: dict(
            sorted(
                {
                    product_list[j]:
                    float(
                        similarity_matrix[i][j]
                    )

                    for j in range(
                        len(product_list)
                    )

                    if i != j
                }.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

        for i, product in enumerate(
            product_list
        )
    }

      
    

    # ==========================================
    # CLUSTER PROFILES
    # ==========================================

    profile_df = embeddings_df.drop(
        columns=[
            "umap1",
            "umap2"
        ]
    )

    cluster_profiles = (
        profile_df
        .groupby(
            "product_cluster"
        )
        .mean(
            numeric_only=True
        )
    )
    seasonality_threshold = (
        cluster_profiles["seasonality_strength"]
        .quantile(0.75)
    )

    entropy_threshold = (
        cluster_profiles["entropy"]
        .quantile(0.75)
    )

    memory_threshold = (
        cluster_profiles["memory_strength"]
        .quantile(0.75)
    )

    stability_threshold = (
        cluster_profiles["stability_score"]
        .quantile(0.75)
    )
    intermittent_threshold = (
        cluster_profiles["zero_demand_ratio"]
        .quantile(0.75)
    )

    decline_threshold = (
        cluster_profiles["trend_score"]
        .quantile(0.25)
    )
  

    cluster_names = {}

    for cluster in cluster_profiles.index:

        row = cluster_profiles.loc[
            cluster
        ]

        mean_stability = row.get(
            "stability_score",
            0
        )

        mean_memory = row.get(
            "memory_strength",
            0
        )

        mean_entropy = row.get(
            "entropy",
            0
        )
        mean_seasonality = row.get(
            "seasonality_strength",
            0
        )

        mean_intermittent = row.get(
            "zero_demand_ratio",
            0
        )

        mean_trend = row.get(
            "trend_score",
            0
        )

        tags = []

        if mean_seasonality > seasonality_threshold:
            tags.append("Seasonal")

        if mean_entropy > entropy_threshold:
            tags.append("Volatile")

        if mean_memory > memory_threshold:
            tags.append("Fast-Moving")
        if mean_intermittent > intermittent_threshold:
            tags.append("Intermittent")

        if mean_trend < decline_threshold:
            tags.append("Declining")

        if mean_stability > stability_threshold:
            tags.append("Stable")

        if len(tags) == 0:

            name = "Mixed Product"

        else:

            name = " + ".join(tags)

        cluster_names[
            int(cluster)
        ] = name
    embeddings_df["cluster_name"] = (
        embeddings_df["product_cluster"]
        .map(cluster_names)
    )

    # ==========================================
    # CLUSTER COUNTS
    # ==========================================

    cluster_counts = (
        embeddings_df[
            "product_cluster"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    cluster_members = (
        embeddings_df
        .groupby(
            "product_cluster"
        )["product"]
        .apply(list)
        .to_dict()
    )
    cluster_profile_records = []

    for cluster in cluster_profiles.index:

        row = cluster_profiles.loc[cluster]

        cluster_profile_records.append({
            "cluster": int(cluster),

            "memory_strength":
            float(row["memory_strength"]),

            "stability_score":
            float(row["stability_score"]),

            "seasonality_strength":
            float(row["seasonality_strength"]),

            "entropy":
            float(row["entropy"]),

            "zero_demand_ratio":
            float(row["zero_demand_ratio"]),

            "trend_score":
            float(row["trend_score"])
        })

    # ==========================================
    # RETURN
    # ==========================================

    return {

        "products":
        embeddings_df.to_dict(
            "records"
        ),

        "best_k":
        int(best_k),

        "silhouette_score":
        float(best_score),

        "cluster_counts":
        cluster_counts,

        "cluster_profiles":
        cluster_profiles.to_dict(),

        "cluster_members":
        cluster_members,
        "cluster_names":
        cluster_names,

        "product_similarity":
        product_similarity,
        "embedding_dimensions":
        embedding_dimensions,

        "pca_components":
        pca_components,

        "explained_variance":
        explained_variance,
        "skipped_products":
         skipped_products,
         "cluster_profile_records":
        cluster_profile_records,
    }


if __name__ == "__main__":

    result = (
        run_product_embeddings()
    )

    print(
        "\nBest K:",
        result["best_k"]
    )

    print(
        "\nSilhouette:",
        result[
            "silhouette_score"
        ]
    )

    print(
        "\nCluster Counts:"
    )

    print(
        result[
            "cluster_counts"
        ]
    )

    print(
        "\nCluster Names:"
    )

    print(
        result[
            "cluster_names"
        ]
    )
    print(
        "\nPCA Components:",
        result["pca_components"]
    )

    print(
        "\nExplained Variance:",
        f"{result['explained_variance']:.2%}"
    )