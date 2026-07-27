# ==========================================================
# PHASE 3
# DEMAND PHASE DISCOVERY
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import networkx as nx

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering
)

from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from load_supabase import supabase
# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42
TRAIN_RATIO = 0.80

MIN_PHASES = 2

MAX_PHASES = 12

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
# ==========================================================
# BEHAVIOUR EMBEDDING COLUMNS
# ==========================================================

EMBEDDING_COLUMNS = [

    f"embedding_{i}"

    for i in range(1, 13)

]
# ==========================================================
# LOAD BEHAVIOUR EMBEDDINGS + ORIGINAL DEMAND
# ==========================================================

def load_behaviour_embeddings(product="Overall"):

    print("\n" + "=" * 70)
    print("LOADING BEHAVIOUR EMBEDDINGS")
    print("=" * 70)

    # --------------------------------------------
    # Behaviour Embeddings
    # --------------------------------------------

    emb = (
        supabase
        .table("behaviour_embeddings")
        .select("*")
        .eq("product", product)
        .order("sales_date")
        .execute()
    )

    emb_df = pd.DataFrame(emb.data)

    # --------------------------------------------
    # Demand Features
    # --------------------------------------------

    demand = (
        supabase
        .table("demand_phase_features")
        .select("*")
        .eq("product", product)
        .order("sales_date")
        .execute()
    )
    demand_df = pd.DataFrame(demand.data)

    if emb_df.empty:
        raise ValueError("No Behaviour Embeddings found.")

    if demand_df.empty:
        raise ValueError("No Demand Features found.")

    emb_df["sales_date"] = pd.to_datetime(
        emb_df["sales_date"]
    )

    demand_df["sales_date"] = pd.to_datetime(
        demand_df["sales_date"]
    )

    # --------------------------------------------
    # Merge
    # --------------------------------------------

    df = emb_df.merge(

        demand_df,

        on=[
            "sales_date",
            "product"
        ],

        how="left"

    )
    print("\nMerged Columns")
    print(df.columns.tolist())

    print("\nDemand Missing")
    print(df["demand"].isna().sum())

    print("\nMerged Preview")
    print(df[["sales_date", "product", "demand"]].head())

    print()

    print("Rows :", len(df))

    print("Columns :", len(df.columns))

    print()

    print(df.head())

    return df
# ==========================================================
# PREPARE BEHAVIOUR SPACE
# ==========================================================

def prepare_behaviour_space(df):

    print("\n" + "=" * 70)
    print("PREPARING BEHAVIOUR SPACE")
    print("=" * 70)

    # ------------------------------------------------------
    # Select Embeddings
    # ------------------------------------------------------

    X = df[EMBEDDING_COLUMNS].copy()

    # ------------------------------------------------------
    # Chronological Train/Test Split
    # ------------------------------------------------------

    split = int(len(X) * TRAIN_RATIO)

    X_train = X.iloc[:split].copy()
    X_test = X.iloc[split:].copy()

    # ------------------------------------------------------
    # Missing Values
    # (Use TRAIN statistics only)
    # ------------------------------------------------------

    train_median = X_train.median()

    X_train = X_train.fillna(train_median)
    X_test = X_test.fillna(train_median)

    # ------------------------------------------------------
    # Standardization
    # (Fit ONLY on training)
    # ------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    X_scaled = pd.DataFrame(
        np.vstack([X_train_scaled, X_test_scaled]),
        columns=EMBEDDING_COLUMNS,
        index=df.index
    )

 
    # ------------------------------------------------------
    # Quality Report
    # ------------------------------------------------------

    quality = {

        "rows": len(df),

        "embedding_dimensions": len(EMBEDDING_COLUMNS),

        "missing_values":
            int(X.isna().sum().sum())

    }

    print()

    print("Quality")

    print(quality)

    print()

    print("Embedding Dimensions")

    print(len(EMBEDDING_COLUMNS))

    print()

    print("Preview")

    print(X_scaled.head())

    return {

        "X": X_scaled,

        "scaler": scaler,

        "quality": quality

    }
# ==========================================================
# FIND OPTIMAL NUMBER OF DEMAND PHASES
# ==========================================================

# ==========================================================
# COMPARE CLUSTERING ALGORITHMS
# ==========================================================

def compare_clustering_algorithms(X):

    print("\n" + "=" * 70)
    print("COMPARING CLUSTERING ALGORITHMS")
    print("=" * 70)

    results = []

    algorithms = {

        "KMeans": lambda k: KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=50
        ),

        "GMM": lambda k: GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=10,
            random_state=RANDOM_STATE
        )

      
    }

    for name, builder in algorithms.items():

        print(f"\n{name}")

        for k in range(MIN_PHASES, MAX_PHASES + 1):

            model = builder(k)

            # -----------------------------
            # Labels
            # -----------------------------

            labels = model.fit_predict(X)

            # -----------------------------
            # Metrics
            # -----------------------------

            silhouette = silhouette_score(X, labels)

            db = davies_bouldin_score(X, labels)

            ch = calinski_harabasz_score(X, labels)

            results.append({

                "algorithm": name,

                "k": k,

                "silhouette": silhouette,

                "davies_bouldin": db,

                "calinski_harabasz": ch

            })

    results = pd.DataFrame(results)
  

    # -----------------------------------
    # Ranking
    # -----------------------------------

    results["rank_silhouette"] = results["silhouette"].rank(
    ascending=False
    )

    results["rank_db"] = results["davies_bouldin"].rank(
        ascending=True
    )

    results["rank_ch"] = results["calinski_harabasz"].rank(
        ascending=False
    )

    results["total_rank"] = (

        results["rank_silhouette"]

        +

        results["rank_db"]

        +

        results["rank_ch"]

    )
    results = results.sort_values(

        ["total_rank", "silhouette"],

        ascending=[True, False]

    ).reset_index(drop=True)

    best = results.iloc[0]

    print()

    print(
        results.sort_values("total_rank")
    )

    print()

    print("=" * 70)

    print("BEST CLUSTERING MODEL")

    print("=" * 70)

    print(f"Algorithm : {best['algorithm']}")
    print(f"Clusters  : {int(best['k'])}")
    print(f"Silhouette: {best['silhouette']:.4f}")
    print(f"Davies-Bouldin: {best['davies_bouldin']:.4f}")
    print(f"Calinski-Harabasz: {best['calinski_harabasz']:.2f}")

    return best, results
# ==========================================================
# TRAIN FINAL DEMAND PHASE MODEL
# ==========================================================

def train_final_phase_model(df, X_train, X_test, best_model):

    print("\n" + "=" * 70)
    print("TRAINING FINAL DEMAND PHASE MODEL")
    print("=" * 70)

    algorithm = best_model["algorithm"]

    k = int(best_model["k"])

    print(f"\nAlgorithm : {algorithm}")
    print(f"Phases    : {k}")

    # ------------------------------------------------------
    # Train Winning Model
    # ------------------------------------------------------

    if algorithm == "KMeans":

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=50
        )

        model.fit(X_train)

        train_labels = model.predict(X_train)

        test_labels = model.predict(X_test)

        labels = np.concatenate([
            train_labels,
            test_labels
        ])

        centroids = pd.DataFrame(
            model.cluster_centers_,
            columns=EMBEDDING_COLUMNS
        )

    elif algorithm == "GMM":

        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=10,
            random_state=RANDOM_STATE
        )

        model.fit(X_train)

        train_labels = model.predict(X_train)

        test_labels = model.predict(X_test)

        labels = np.concatenate([
            train_labels,
            test_labels
        ])

        centroids = pd.DataFrame(
            model.means_,
            columns=EMBEDDING_COLUMNS
        )


    # ------------------------------------------------------
    # INITIAL CLUSTER LABELS
    # ------------------------------------------------------

    phase_df = df.copy()

    phase_df["cluster"] = labels
    # ------------------------------------------------------
    # CLUSTER BEHAVIOUR PROFILE
    # ------------------------------------------------------

 
    

    cluster_profile = (

        phase_df

        .groupby("cluster")

        .agg(

            mean_demand=("demand","mean"),

            mean_memory=("rolling_memory_strength","mean"),

            mean_entropy=("demand_entropy_30","mean"),

            mean_cv=("local_cv_30","mean"),

            mean_transition=("transition_strength","mean"),

            observations=("cluster","size")

        )

    )
    from sklearn.preprocessing import MinMaxScaler

    score_cols = [

        "mean_demand",
        "mean_memory",
        "mean_entropy",
        "mean_cv",
        "mean_transition"

    ]

    scaler = MinMaxScaler()

    cluster_profile[score_cols] = scaler.fit_transform(

        cluster_profile[score_cols]

    )
    # ------------------------------------------------------
    # SORT CLUSTERS INTO SEMANTIC ORDER
    # ------------------------------------------------------

    # ------------------------------------------------------
    # COMPUTE BEHAVIOUR SCORE
    # ------------------------------------------------------

    cluster_profile["phase_quality_score"] = (

        0.30 * cluster_profile["mean_memory"]

        - 0.25 * cluster_profile["mean_cv"]

        - 0.20 * cluster_profile["mean_entropy"]

        - 0.15 * cluster_profile["mean_transition"]

        + 0.10 * cluster_profile["mean_demand"]

    )

    # ------------------------------------------------------
    # SORT SEMANTIC PHASES
    # ------------------------------------------------------

    cluster_profile = (

        cluster_profile

        .sort_values(

            "phase_quality_score",

            ascending=False

        )

    )

    print("\nCluster Behaviour Ranking")

    print(cluster_profile)
    phase_map = {}

    for i, cluster in enumerate(cluster_profile.index):

        phase_map[int(cluster)] = i + 1

    print("\nSemantic Phase Mapping")

    print(phase_map)
    # ------------------------------------------------------
    # REORDER CENTROIDS TO MATCH SEMANTIC PHASES
    # ------------------------------------------------------

    centroids["cluster"] = centroids.index

    centroids["phase"] = centroids["cluster"].map(

        phase_map

    )

    centroids = (

        centroids

        .sort_values("phase")

        .drop(

            columns=[

                "cluster",

                "phase"

            ]

        )

        .reset_index(drop=True)

    )
    # ------------------------------------------------------
    # ASSIGN SEMANTIC PHASES
    # ------------------------------------------------------

    phase_df["demand_phase"] = (

        phase_df["cluster"]

        .map(phase_map)

    )
    # ------------------------------------------------------
    # Phase Sizes
    # ------------------------------------------------------

    phase_summary = (

        phase_df

        .groupby("demand_phase")

        .agg(

            observations=("demand_phase", "size")

        )

        .reset_index()

        .sort_values("demand_phase")

    )

    print("\nPhase Summary")

    print(phase_summary)

    print("\nCentroids")

    print(centroids.round(3))

    return {

        "model": model,

        "phase_df": phase_df,

        "centroids": centroids,

        "summary": phase_summary

    }
# ==========================================================
# PHASE DYNAMICS
# ==========================================================

def analyze_phase_dynamics(phase_df):

    print("\n" + "=" * 70)
    print("ANALYZING PHASE DYNAMICS")
    print("=" * 70)

    df = phase_df.copy()

    df = df.sort_values(
        "sales_date"
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # Previous / Next Phase
    # ------------------------------------------------------

    df["previous_phase"] = (
        df["demand_phase"]
        .shift(1)
    )

    df["next_phase"] = (
        df["demand_phase"]
        .shift(-1)
    )

    # ------------------------------------------------------
    # Phase Change
    # ------------------------------------------------------

    df["phase_change"] = (

        df["demand_phase"]

        !=

        df["previous_phase"]

    ).astype(int)

    # ------------------------------------------------------
    # Phase Runs
    # ------------------------------------------------------

    df["run_id"] = (

        df["phase_change"]

        .cumsum()

    )

    persistence = (

        df

        .groupby(

            ["demand_phase", "run_id"]

        )

        .size()

        .reset_index(name="duration")

    )

    persistence_summary = (

        persistence

        .groupby("demand_phase")

        .agg(

            average_duration=("duration", "mean"),

            median_duration=("duration", "median"),

            maximum_duration=("duration", "max"),

            episodes=("duration", "count")

        )

        .reset_index()

    )

    print("\nPhase Persistence")

    print(
        persistence_summary
    )

    # ------------------------------------------------------
    # Transition Matrix
    # ------------------------------------------------------

    transitions = pd.crosstab(

        df["previous_phase"],

        df["demand_phase"],

        normalize="index"

    )

    print("\nTransition Matrix")

    print(

        transitions.round(3)

    )

    return {

        "phase_df": df,

        "persistence": persistence_summary,

        "transition_matrix": transitions

    }
# ==========================================================
# PHASE INTELLIGENCE
# ==========================================================

def compute_phase_intelligence(
    phase_results,
    dynamics
):

    print("\n" + "=" * 70)
    print("COMPUTING PHASE INTELLIGENCE")
    print("=" * 70)

    phase_df = dynamics["phase_df"].copy()

    transition_matrix = dynamics["transition_matrix"]

    centroids = phase_results["centroids"]

    X = phase_df[EMBEDDING_COLUMNS].values

    # ------------------------------------------------------
    # Phase Stability Index
    # ------------------------------------------------------

    psi = {}

    for phase in transition_matrix.index:

        psi[int(phase)] = transition_matrix.loc[
            phase,
            phase
        ]

    # ------------------------------------------------------
    # Rare Phase Score
    # ------------------------------------------------------

    counts = phase_df[
        "demand_phase"
    ].value_counts()

    total = len(phase_df)

    rare_score = {}

    for phase, n in counts.items():

        rare_score[int(phase)] = 1 - (n / total)

    # ------------------------------------------------------
    # Phase Entropy
    # ------------------------------------------------------

    entropy = {}

    for phase in transition_matrix.index:

        p = transition_matrix.loc[phase].values

        p = p[p > 0]

        entropy[int(phase)] = -np.sum(
            p * np.log2(p)
        )

    # ------------------------------------------------------
    # Phase Confidence
    # ------------------------------------------------------

    confidence = []

    centroid_array = centroids.values

    for i, row in phase_df.iterrows():

        phase = int(row["demand_phase"]) - 1

        x = X[i]

        c = centroid_array[phase]

        distance = np.linalg.norm(

            x - c

        )

        confidence.append(

            1 / (1 + distance)

        )

    phase_df["phase_confidence"] = confidence

    phase_df["phase_stability"] = phase_df[
        "demand_phase"
    ].map(psi)

    phase_df["rare_phase_score"] = phase_df[
        "demand_phase"
    ].map(rare_score)

    phase_df["phase_entropy"] = phase_df[
        "demand_phase"
    ].map(entropy)

    summary = pd.DataFrame({

        "phase": sorted(psi.keys()),

        "stability": [
            psi[p]
            for p in sorted(psi.keys())
        ],

        "entropy": [
            entropy[p]
            for p in sorted(entropy.keys())
        ],

        "rare_score": [
            rare_score[p]
            for p in sorted(rare_score.keys())
        ]

    })

    print("\nPhase Intelligence")

    print(summary.round(3))

    print("\nConfidence Summary")

    print(
        phase_df[
            "phase_confidence"
        ].describe()
    )

    return {

        "phase_df": phase_df,

        "summary": summary

    }
# ==========================================================
# PHASE ROLE CLASSIFICATION
# ==========================================================

def classify_phase_roles(intelligence, network):

    print("\n" + "=" * 70)
    print("PHASE ROLE CLASSIFICATION")
    print("=" * 70)

    summary = intelligence["summary"].copy()

    nodes = network["nodes"]

    summary = summary.merge(
        nodes,
        left_on="phase",
        right_on="phase"
    )

    roles = []

    for _, row in summary.iterrows():

        role = []

        if row["rare_score"] > 0.95:
            role.append("Rare")

        if row["stability"] > 0.85:
            role.append("Persistent")

        elif row["entropy"] > 1.5:
            role.append("Transition")

        else:
            role.append("Operational")

        roles.append(" + ".join(role))

    summary["role"] = roles

  
    print(summary[
        [
            "phase",
            "role",
            "stability",
            "entropy",
            "rare_score"
        ]
    ])

    return summary
# ==========================================================
# DEMAND PHASE TRANSITION NETWORK
# ==========================================================

def build_phase_transition_network(dynamics):

    print("\n" + "=" * 70)
    print("BUILDING DEMAND PHASE TRANSITION NETWORK")
    print("=" * 70)

    transition_matrix = dynamics["transition_matrix"]

    edges = []

    # -----------------------------------------
    # Extract Network Edges
    # -----------------------------------------

    for source in transition_matrix.index:

        for target in transition_matrix.columns:

            probability = transition_matrix.loc[source, target]

            if probability > 0:

                edges.append({

                    "source": int(source),

                    "target": int(target),

                    "probability": probability

                })

    edges = pd.DataFrame(edges)

    # -----------------------------------------
    # Network Statistics
    # -----------------------------------------

    node_summary = []

    phases = sorted(
        transition_matrix.index.astype(int)
    )

    for phase in phases:

        incoming = edges[
            edges.target == phase
        ]

        outgoing = edges[
            edges.source == phase
        ]

        self_transition = transition_matrix.loc[
            phase,
            phase
        ]

        node_summary.append({

            "phase": phase,

            "in_degree": len(incoming),

            "out_degree": len(outgoing),

            "incoming_probability":
                incoming.probability.sum(),

            "outgoing_probability":
                outgoing.probability.sum(),

            "self_transition":
                self_transition,


        })

    node_summary = pd.DataFrame(node_summary)

    print("\nNetwork Nodes")

    print(node_summary.round(3))

    print("\nNetwork Edges")

    print(edges.round(3))

    return {

        "edges": edges,

        "nodes": node_summary

    }
# ==========================================================
# GRAPH CENTRALITY ANALYSIS
# ==========================================================

def compute_graph_centrality(network):

    print("\n" + "=" * 70)
    print("GRAPH CENTRALITY ANALYSIS")
    print("=" * 70)

    edges = network["edges"]

    G = nx.DiGraph()

    # --------------------------------------------
    # Build Directed Graph
    # --------------------------------------------

    for _, row in edges.iterrows():

        G.add_edge(

            int(row["source"]),

            int(row["target"]),

            weight=float(row["probability"])

        )
    G.remove_edges_from(nx.selfloop_edges(G))

    # --------------------------------------------
    # Centrality Measures
    # --------------------------------------------

    degree = nx.degree_centrality(G)

    in_degree = nx.in_degree_centrality(G)

    out_degree = nx.out_degree_centrality(G)

    betweenness = nx.betweenness_centrality(

        G,

        weight="weight"

    )

    closeness = nx.closeness_centrality(G)

    pagerank = nx.pagerank(

        G,

        weight="weight"

    )

    centrality = pd.DataFrame({

        "phase": list(degree.keys()),

        "degree_centrality":
            list(degree.values()),

        "in_degree_centrality":
            list(in_degree.values()),

        "out_degree_centrality":
            list(out_degree.values()),

        "betweenness":
            list(betweenness.values()),

        "closeness":
            list(closeness.values()),

        "pagerank":
            list(pagerank.values())

    })
    centrality = centrality.sort_values(
        "pagerank",
        ascending=False
    ).reset_index(drop=True)

    print()

    print(centrality.round(3))

    return centrality

# ==========================================================
# SAVE PHASE DISCOVERY RESULTS
# ==========================================================

from datetime import datetime, date

def save_phase_discovery_results(df):

    print("\n" + "=" * 70)
    print("SAVING PHASE DISCOVERY RESULTS")
    print("=" * 70)

    df = df.copy()
    print("\nColumns received by save function")
    print(df.columns.tolist())

    # ------------------------------------------------------
    # Columns to Save
    # ------------------------------------------------------

    required_columns = [

        "sales_date",

        "product",

        "demand",

        "demand_phase",

        "phase_confidence",

        "phase_stability",

        "rare_phase_score",

        "phase_entropy",

        "role",

        "degree_centrality",

        "betweenness",

        "closeness",

        "pagerank"

    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[required_columns]



    # ------------------------------------------------------
    # Datetime Conversion
    # ------------------------------------------------------

    for col in df.columns:

        df[col] = df[col].apply(

            lambda x: x.isoformat()

            if isinstance(x, (pd.Timestamp, datetime, date))

            else x

        )
    duplicates = df.duplicated(
        subset=["product", "sales_date"]
    ).sum()

    print("Duplicate rows:", duplicates)

    if duplicates > 0:
        raise ValueError(
            "Duplicate product/sales_date rows found."
        )

    records = df.to_dict(orient="records")

    batch_size = 500

    for i in range(0, len(records), batch_size):

        batch = records[i:i + batch_size]

        supabase.table(
            "phase_discovery_results"
        ).upsert(

            batch,

            on_conflict="product,sales_date"

        ).execute()

    print(f"\nSaved {len(records)} Phase Discovery Results.")
if __name__ == "__main__":

    df = load_behaviour_embeddings("Overall")

    data = prepare_behaviour_space(df)

    print()

    print("Behaviour Space Shape")

    print(data["X"].shape)

    split = int(len(data["X"]) * TRAIN_RATIO)

    X_train = data["X"].iloc[:split]

    X_test = data["X"].iloc[split:]

    best_model, evaluation = compare_clustering_algorithms(
        X_train
    )

    phase_results = train_final_phase_model(

        df,

        X_train,

        X_test,

        best_model

    )

    dynamics = analyze_phase_dynamics(

        phase_results["phase_df"]

    )

    intelligence = compute_phase_intelligence(

        phase_results,

        dynamics

    )

    network = build_phase_transition_network(

        dynamics

    )

    roles = classify_phase_roles(

        intelligence,

        network

    )
    centrality = compute_graph_centrality(

        network

    )
   
    phase_summary = (
        roles
        .merge(
            centrality,
            on="phase",
            how="left"
        )
        .sort_values("phase")
        .reset_index(drop=True)
    )



    print("\nFinal Phase Summary")
    print(phase_summary.round(3))
    print(phase_summary)
    # ------------------------------------------------------
    # Merge Graph Metrics into Daily Phase Data
    # ------------------------------------------------------

    phase_df = intelligence["phase_df"].merge(

        phase_summary[
            [
                "phase",
                "role",
                "degree_centrality",
                "betweenness",
                "closeness",
                "pagerank"
            ]
        ],

        left_on="demand_phase",

        right_on="phase",

        how="left"

    )

    phase_df = phase_df.drop(columns=["phase"])
   

    print(
        phase_df[
            [
                "sales_date",
                "demand",
                "demand_phase",
                "role",
                "pagerank"
            ]
        ].head()
    )

    save_phase_discovery_results(
        phase_df
    )
    print("\nMissing values before saving")

    print(
        phase_df[
            [
                "demand",
                "role",
                "degree_centrality",
                "betweenness",
                "closeness",
                "pagerank"
            ]
        ].isna().sum()
    )


  