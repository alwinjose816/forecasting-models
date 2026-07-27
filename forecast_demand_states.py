from forecast_demand_memory import (
    run_forecast_demand_memory
)

import pandas as pd
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run_forecast_demand_states(product="Overall"):

    # ==========================================
    # LOAD PHASE 1 FEATURES
    # ==========================================

    result = run_forecast_demand_memory(
        product
    )

    if "error" in result:
        return result

    features_df = pd.DataFrame(
        result["features"]
    )

    # ==========================================
    # PCA INPUT
    # ==========================================

    exclude_cols = [
        "date",
        "demand"
    ]

    feature_cols = [
        c
        for c in features_df.columns
        if c not in exclude_cols
    ]
    if len(feature_cols) == 0:
        return {
            "error": "No features available for PCA"
        }
    if len(features_df) < 5:
        return {
            "error": "Need at least 5 observations for PCA"
        }
    import numpy as np
    X = (
        features_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    
   

    # ==========================================
    # STANDARDIZE
    # ==========================================

    scaler = StandardScaler()

    split_index = result[
        "split_index"
    ]

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )
        
    n_components = min(
        5,
        len(feature_cols),
        len(features_df)
    )
    
    loading_cols = [
        f"z{i+1}"
        for i in range(n_components)
    ]

   

    

    # ==========================================
    # PCA
    # ==========================================

    pca = PCA(
        n_components=n_components,
        random_state=42
    )
    Z_train = pca.fit_transform(
        X_train_scaled
    )

    Z_test = pca.transform(
        X_test_scaled
    )

    Z = np.vstack([
        Z_train,
        Z_test
    ])

    for i in range(n_components):
        features_df[f"z{i+1}"] = Z[:, i]

    

    os.makedirs(
        "models",
        exist_ok=True
    )
    

    joblib.dump(
        pca,
        "models/demand_state_pca.pkl"
    )

    joblib.dump(
        scaler,
        "models/demand_state_scaler.pkl"
    )

 

    # ==========================================
    # VARIANCE EXPLAINED
    # ==========================================

    variance = {
        f"z{i+1}": float(v)
        for i, v in enumerate(
            pca.explained_variance_ratio_
        )
    }

    variance["total"] = float(
        pca.explained_variance_ratio_.sum()
    )

    variance["cumulative"] = list(
        pca.explained_variance_ratio_.cumsum()
    )
    variance["n_components"] = n_components
   
   
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=loading_cols
    )
    top_features = {}

    for comp in loadings.columns:

        top_features[comp] = (

            loadings[comp]

            .sort_values(
                key=lambda x: abs(x),
                ascending=False
            )

            .head(10)

            .round(4)

            .to_dict()
        )
        

    from sklearn.metrics import silhouette_score

    best_k = 2
    best_score = -1

    max_k = min(
        10,
        len(X_train) - 1
    )

    for k in range(2, max_k + 1):

        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = km.fit_predict(
            Z_train
        )

        score = silhouette_score(
            Z_train,
            labels
        )

        if score > best_score:

            best_score = score
            best_k = k

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    train_labels = kmeans.fit_predict(
        Z_train
    )

    test_labels = kmeans.predict(
        Z_test
    )

    labels = np.concatenate([
        train_labels,
        test_labels
    ])

    features_df["state"] = labels
    profile_cols = [

        "demand",

        "memory_state",

        "memory_residual",

        "rolling_memory_strength",

        "memory_half_life",

        "entropy_90",

        "trend_90",

        "shock_score",

        "zero_demand_ratio_90"

    ]

    train_df = features_df.iloc[:split_index].copy()

    train_df["state"] = train_labels

    state_profiles = (
        train_df
        .groupby("state")[profile_cols]
        .mean()
        .round(3)
    )
    state_counts = (
        train_df["state"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    joblib.dump(
        kmeans,
        "models/demand_state_kmeans.pkl"
    )
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=loading_cols
    )

    cluster_centers["state"] = (
        cluster_centers.index
    )
    from sklearn.metrics import davies_bouldin_score

    db_score = davies_bouldin_score(
        Z_train,
        train_labels
    )

    return {
        "split_index":
        int(split_index),

        "train_states":
        features_df.iloc[
            :split_index
        ].to_dict("records"),

        "test_states":
        features_df.iloc[
            split_index:
        ].to_dict("records"),

        "states":
        features_df.to_dict("records"),

        "variance":
        variance,

        "loadings":
        loadings.to_dict(),

        "top_features":
        top_features,

        "state_counts":
        state_counts,

        "best_k":
        int(best_k),

        "silhouette_score":
        float(best_score),

        "state_profiles":
        state_profiles.to_dict(
            orient="index"
        ),

        "cluster_centers":
        cluster_centers
        .round(3)
        .to_dict(
            orient="records"
        ),

        "davies_bouldin_score":
        float(db_score)
        

    }