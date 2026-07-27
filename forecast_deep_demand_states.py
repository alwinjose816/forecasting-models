from forecast_demand_memory import run_forecast_demand_memory

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

import joblib
import os


LATENT_DIMS = [4, 8, 16]


def run_forecast_deep_demand_states(
    product="Overall",
    latent_dim=8
):

    # ==========================================
    # LOAD PHASE 1 FEATURES
    # ==========================================

    result = run_forecast_demand_memory(product)

    if "error" in result:
        return result

    features_df = pd.DataFrame(
        result["features"]
    )

    exclude_cols = [
        "date",
        "demand"
    ]

    feature_cols = [

        c

        for c in features_df.columns

        if c not in exclude_cols
    ]
    X = (
        features_df[feature_cols]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )



   # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    split_index = int(
        len(X) * 0.8
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    # ==========================================
    # SCALE
    # ==========================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    input_dim = X_train_scaled.shape[1]

    # ==========================================
    # AUTOENCODER
    # ==========================================

    inputs = Input(
        shape=(input_dim,)
    )

    x = Dense(
        64,
        activation="relu"
    )(inputs)

    x = Dense(
        32,
        activation="relu"
    )(x)

    latent = Dense(
        latent_dim,
        activation="tanh",
        name="latent"
    )(x)

    x = Dense(
        32,
        activation="relu"
    )(latent)

    x = Dense(
        64,
        activation="relu"
    )(x)

    outputs = Dense(
        input_dim,
        activation="linear"
    )(x)

    autoencoder = Model(
        inputs,
        outputs
    )

    encoder = Model(
        inputs,
        latent
    )

    autoencoder.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="mse"
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
    history = autoencoder.fit(
        X_train_scaled,
        X_train_scaled,
        validation_split=0.2,
        shuffle=False,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    training_epochs = len(
        history.history["loss"]
    )

    

    Z_train = encoder.predict(
        X_train_scaled,
        verbose=0
    )

    Z_test = encoder.predict(
        X_test_scaled,
        verbose=0
    )

    latent_scaler = StandardScaler()

    Z_train = latent_scaler.fit_transform(
        Z_train
    )

    Z_test = latent_scaler.transform(
        Z_test
    )

    Z = np.vstack([
        Z_train,
        Z_test
    ])
    if len(Z) < 10:
        return {
            "error":
            "Not enough observations for clustering"
        }
    # ==========================================
    # RECONSTRUCTION ERROR
    # ==========================================

    reconstructed = autoencoder.predict(
        X_test_scaled,
        verbose=0
    )

    reconstruction_error = np.mean(
        (X_test_scaled - reconstructed) ** 2
    )

    for i in range(
        latent_dim
    ):

        features_df[
            f"z{i+1}"
        ] = Z[:, i]

    # ==========================================
    # SAVE MODELS
    # ==========================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    encoder.save(
        f"models/deep_encoder_{latent_dim}.keras"
    )

    autoencoder.save(
        f"models/deep_autoencoder_{latent_dim}.keras"
    )

    joblib.dump(
        latent_scaler,
        f"models/deep_latent_scaler_{latent_dim}.pkl"
    )
    joblib.dump(
        scaler,
        f"models/deep_scaler_{latent_dim}.pkl"
    )
    # ==========================================
    # CLUSTERING
    # ==========================================

    best_k = 2
    best_score = -1

    max_k = min(
        10,
        len(Z_train) - 1
    )

    for k in range(
        2,
        max_k + 1
    ):
       

        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
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
    if best_score < 0:

        return {
            "error":
            "Could not find valid clustering solution"
        }

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20
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
    joblib.dump(
        kmeans,
        f"models/deep_kmeans_{latent_dim}.pkl"
    )

    db_score = (
        davies_bouldin_score(
            Z_train,
            train_labels
        )
    )
    cluster_quality = (
        best_score /
        (1 + db_score)
    )

    train_profiles = (
        features_df.iloc[:split_index]
        .copy()
    )

    train_profiles["state"] = train_labels

    state_counts = (

        train_profiles["state"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    state_profiles = (
        train_profiles
        .groupby("state")
        .mean(numeric_only=True)
    )

    state_profiles = (
        train_profiles
        .groupby("state")
        .mean(numeric_only=True)
    )
    # ==========================================
    # STATE NAMES
    # ==========================================

    state_names = {}

    for state in state_profiles.index:

        row = state_profiles.loc[state]

        name = "Mixed Demand"

        if (
            "zero_demand_ratio_90" in row.index
            and
            row["zero_demand_ratio_90"] > 0.30
        ):
            name = "Intermittent Demand"

        elif (
            "shock_score" in row.index
            and
            row["shock_score"] > 1.50
        ):
            name = "Demand Shock"

        elif (
            "trend_90" in row.index
            and
            row["trend_90"] > 0.10
        ):
            name = "Growth Demand"

        elif (
            "rolling_memory_strength" in row.index
            and
            row["rolling_memory_strength"] > 0.50
        ):
            name = "Stable Demand"

        state_names[
            int(state)
        ] = name
    best_val_loss = min(
        history.history["val_loss"]
    )

    return {

        "states":
        features_df.to_dict(
            "records"
        ),
        "split_index":
        split_index,

        "latent_dim":
        latent_dim,

        "best_k":
        best_k,

        "silhouette_score":
        float(best_score),

        "davies_bouldin_score":
        float(db_score),
        "cluster_quality":
        float(cluster_quality),

        "state_counts":
        state_counts,

        "state_profiles":
        state_profiles.round(3).to_dict(
            orient="index"
        ),
        "reconstruction_error":
        float(reconstruction_error),
        "state_names":
        state_names,
        "training_epochs":
        int(training_epochs),
        "best_val_loss":
        float(best_val_loss)
    }
def run_latent_dimension_study():

    rows = []

    for dim in LATENT_DIMS:

        result = run_forecast_deep_demand_states(
            product="Overall",
            latent_dim=dim
        )

        if "error" in result:
            continue


        rows.append({

            "latent_dim":
            dim,

            "reconstruction_error":
            result["reconstruction_error"],

            "best_val_loss":
            result["best_val_loss"],

            "silhouette":
            result["silhouette_score"],

            "davies_bouldin":
            result["davies_bouldin_score"],
            "cluster_quality":
            result["cluster_quality"],

            "best_k":
            result["best_k"],
            "training_epochs":
            result["training_epochs"]

        })

    study = pd.DataFrame(rows)

    study["rank"] = (
        study["cluster_quality"]
        .rank(
            ascending=False,
            method="dense"
        )
        .astype(int)
    )

    return study.sort_values(
        "cluster_quality",
        ascending=False
    )

if __name__ == "__main__":

    study = run_latent_dimension_study()

    print(study)

    result = run_forecast_deep_demand_states(
        product="Overall",
        latent_dim=8
    )

    print(
        "\nLatent Dimension:",
        result["latent_dim"]
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
        "\nTraining Epochs:",
        result["training_epochs"]
    )
    print(
        "\nReconstruction Error:",
        result["reconstruction_error"]
    )

    print(
        "\nBest Validation Loss:",
        result["best_val_loss"]
    )
    print(
        "\nState Counts:"
    )

    print(
        result["state_counts"]
    )
    print(
        "\nState Names:"
    )

    print(
        result["state_names"]
    )