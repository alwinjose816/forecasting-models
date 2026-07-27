# ==========================================================
# PHASE GRAPH ANALYTICS
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import networkx as nx

from load_supabase import supabase
TRAIN_RATIO = 0.80
def load_dataset(product="Overall"):

    print("\n" + "="*70)
    print("LOADING PHASE DATASET")
    print("="*70)

    response = (

        supabase

        .table("final_training_data")

        .select("*")

        .eq("product", product)

        .order("sales_date")

        .execute()

    )

    df = pd.DataFrame(response.data)

    df["sales_date"] = pd.to_datetime(df["sales_date"])

    print()

    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    return df
def create_phase_pairs(df):

    print("\n" + "="*70)
    print("CREATING PHASE PAIRS")
    print("="*70)

    df = df.copy()

    df["next_phase"] = df["demand_phase"].shift(-1)

    # ONLY DROP THE LAST ROW
    df = df.dropna(subset=["next_phase"])

    df["next_phase"] = df["next_phase"].astype(int)

    print(df.head())

    print("Rows :", len(df))

    return df
    
def build_transition_matrix(df):

    matrix = (

        df

        .groupby(

            [

                "demand_phase",

                "next_phase"

            ]

        )

        .size()

        .reset_index(

            name="transition_count"

        )

    )

    matrix["transition_probability"] = (

        matrix["transition_count"]

        /

        matrix.groupby(

            "demand_phase"

        )["transition_count"]

        .transform("sum")

    )
    print(matrix.head())
    print(matrix.shape)

    return matrix
   
# ==========================================================
# BUILD PHASE GRAPH
# ==========================================================

def build_phase_graph(matrix):

    print("\n" + "="*70)
    print("BUILDING PHASE GRAPH")
    print("="*70)

    G = nx.DiGraph()

    for _, row in matrix.iterrows():

        G.add_edge(

            int(row["demand_phase"]),

            int(row["next_phase"]),

            weight=float(row["transition_probability"]),

            count=int(row["transition_count"])

        )

    print()

    print("Nodes :", G.number_of_nodes())

    print("Edges :", G.number_of_edges())

    print()

    print("Edges")

    for u, v, d in G.edges(data=True):

        print(

            f"{u} -> {v}"

            f"  Prob={d['weight']:.3f}"

            f"  Count={d['count']}"

        )

    return G
# ==========================================================
# GRAPH ANALYTICS
# ==========================================================

def compute_graph_metrics(G):

    print("\n" + "="*70)
    print("GRAPH ANALYTICS")
    print("="*70)

    pagerank = nx.pagerank(

        G,

        weight="weight"

    )

    betweenness = nx.betweenness_centrality(

        G,

        weight="weight"

    )

    closeness = nx.closeness_centrality(

        G

    )

    indegree = dict(

        G.in_degree()

    )

    outdegree = dict(

        G.out_degree()

    )

    weighted_degree = dict(

        G.degree(weight="weight")

    )

    metrics = []

    for node in sorted(G.nodes()):

        metrics.append({

            "phase": node,

            "in_degree": indegree[node],

            "out_degree": outdegree[node],

            "weighted_degree": weighted_degree[node],

            "pagerank": pagerank[node],

            "betweenness": betweenness[node],

            "closeness": closeness[node]

        })

    metrics = pd.DataFrame(metrics)

    print()

    print(metrics)

    return metrics
# ==========================================================
# PHASE STABILITY
# ==========================================================

def compute_phase_stability(matrix, metrics):

    print("\n" + "="*70)
    print("PHASE STABILITY")
    print("="*70)

    stay = (

        matrix

        [

            matrix["demand_phase"]

            ==

            matrix["next_phase"]

        ]

        [

            [

                "demand_phase",

                "transition_probability"

            ]

        ]

        .rename(

            columns={

                "transition_probability":"stay_probability"

            }

        )

    )

    metrics = metrics.merge(

        stay,

        left_on="phase",

        right_on="demand_phase",

        how="left"

    )

    metrics["stay_probability"] = (

        metrics["stay_probability"]

        .fillna(0)

    )

    metrics["escape_probability"] = (

        1

        -

        metrics["stay_probability"]

    )

    metrics.drop(

        columns=["demand_phase"],

        inplace=True

    )

    print()

    print(metrics)

    return metrics
# ==========================================================
# SAVE GRAPH METRICS
# ==========================================================

def save_graph_metrics(metrics):

    print("\n" + "="*70)
    print("SAVING GRAPH METRICS")
    print("="*70)

    data = metrics.copy()

    print()

    print(data)

    # Uncomment after creating table

    # supabase.table(
    #     "phase_graph_metrics"
    # ).upsert(
    #     data.to_dict("records")
    # ).execute()

    print()

    print("Ready to save.")
if __name__ == "__main__":

    df = load_dataset()

    df = create_phase_pairs(df)

    matrix = build_transition_matrix(df)
    print(matrix.shape)
    print(matrix.head())
    print(type(matrix))

    G = build_phase_graph(matrix)

    metrics = compute_graph_metrics(G)

    metrics = compute_phase_stability(

        matrix,

        metrics

    )

    save_graph_metrics(metrics)
