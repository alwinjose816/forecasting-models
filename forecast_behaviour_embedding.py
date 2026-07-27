# ==========================================================
# PHASE 2
# DEMAND BEHAVIOUR EMBEDDING
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    GaussianNoise,
    Dropout
)

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

from load_supabase import (
    supabase,
    save_behaviour_embeddings
)


# ==========================================================
# CONFIGURATION
# ==========================================================

LATENT_DIM = 12

BATCH_SIZE = 32

EPOCHS = 150

RANDOM_STATE = 42
# ==========================================================
# REPRODUCIBILITY
# ==========================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ==========================================================
# BEHAVIOUR FEATURES
# ==========================================================

BEHAVIOUR_FEATURES = [

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

    "phase_potential",
    "phase_velocity",
    "phase_curvature"

]


# ==========================================================
# LOAD PHASE FEATURES
# ==========================================================

def load_phase_features(product="Overall"):

    query = (
        supabase
        .table("demand_phase_features")
        .select("*")
        .eq("product", product)
        .order("sales_date")
    )

    response = query.execute()

    df = pd.DataFrame(response.data)

    if df.empty:
        raise ValueError(
            f"No phase features found for {product}"
        )

    df["sales_date"] = pd.to_datetime(
        df["sales_date"]
    )

    return df

# ==========================================================
# PREPARE BEHAVIOUR MATRIX
# ==========================================================

def prepare_behaviour_matrix(df):

    print("\n" + "=" * 70)
    print("PREPARING BEHAVIOUR MATRIX")
    print("=" * 70)

    # ------------------------------------------------------
    # Select Behaviour Features
    # ------------------------------------------------------

    X = df[BEHAVIOUR_FEATURES].copy()

    # ------------------------------------------------------
    # Chronological Train/Test Split
    # ------------------------------------------------------

    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    meta_train = df.iloc[:split_index][
        ["sales_date", "product"]
    ].copy()

    meta_test = df.iloc[split_index:][
        ["sales_date", "product"]
    ].copy()

    # ------------------------------------------------------
    # Missing Values
    # (Use TRAIN statistics only)
    # ------------------------------------------------------

    train_median = X_train.median()

    X_train = X_train.fillna(train_median)
    X_test = X_test.fillna(train_median)

    # ------------------------------------------------------
    # Standardization
    # (Fit ONLY on training data)
    # ------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=BEHAVIOUR_FEATURES,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=BEHAVIOUR_FEATURES,
        index=X_test.index
    )

    # ------------------------------------------------------
    # Quality Report
    # ------------------------------------------------------

    quality = {

        "rows": len(df),

        "train_rows": len(X_train),

        "test_rows": len(X_test),

        "behaviour_dimensions": len(BEHAVIOUR_FEATURES),

        "missing_values_train":
            int(X_train.isna().sum().sum()),

        "missing_values_test":
            int(X_test.isna().sum().sum())

    }

    print("\nQuality")
    print(quality)

    print("\nBehaviour Dimensions")
    print(len(BEHAVIOUR_FEATURES))

    print("\nBehaviour Features")

    for feature in BEHAVIOUR_FEATURES:
        print(feature)

    return {

        "X_train": X_train_scaled,

        "X_test": X_test_scaled,

        "meta_train": meta_train,

        "meta_test": meta_test,

        "scaler": scaler,

        "train_median": train_median,

        "quality": quality

    }
# ==========================================================
# BUILD DENOISING BEHAVIOUR AUTOENCODER
# ==========================================================

def build_behaviour_autoencoder():

    print("\n" + "=" * 70)
    print("BUILDING BEHAVIOUR AUTOENCODER")
    print("=" * 70)

    # ------------------------------------------------------
    # Encoder
    # ------------------------------------------------------

    inputs = Input(
        shape=(len(BEHAVIOUR_FEATURES),),
        name="Behaviour_Input"
    )

    x = GaussianNoise(
        0.05,
        name="Gaussian_Noise"
    )(inputs)

    x = Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(1e-4),
        name="Encoder_64"
    )(x)

    x = Dropout(
        0.10,
        name="Dropout_64"
    )(x)

    x = Dense(
        32,
        activation="relu",
        kernel_regularizer=l2(1e-4),
        name="Encoder_32"
    )(x)

    x = Dropout(
        0.10,
        name="Dropout_32"
    )(x)

    x = Dense(
        16,
        activation="relu",
        kernel_regularizer=l2(1e-4),
        name="Encoder_16"
    )(x)

    embedding = Dense(
        LATENT_DIM,
        activation="tanh",
        name="Behaviour_Embedding"
    )(x)

    # ------------------------------------------------------
    # Decoder
    # ------------------------------------------------------

    x = Dense(
        16,
        activation="relu",
        kernel_regularizer=l2(1e-4),
        name="Decoder_16"
    )(embedding)

    x = Dense(
        32,
        activation="relu",
        kernel_regularizer=l2(1e-4),
        name="Decoder_32"
    )(x)

    x = Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(1e-4),
        name="Decoder_64"
    )(x)

    outputs = Dense(
        len(BEHAVIOUR_FEATURES),
        activation="linear",
        name="Reconstruction"
    )(x)

    # ------------------------------------------------------
    # Models
    # ------------------------------------------------------

    autoencoder = Model(
        inputs,
        outputs,
        name="Behaviour_Autoencoder"
    )

    encoder = Model(
        inputs,
        embedding,
        name="Behaviour_Encoder"
    )

    autoencoder.compile(

        optimizer=Adam(
            learning_rate=0.001
        ),

        loss="mse"

    )

    print()

    autoencoder.summary()

    return autoencoder, encoder
# ==========================================================
# TRAIN BEHAVIOUR AUTOENCODER
# ==========================================================

def train_behaviour_autoencoder(
    autoencoder,
    X_train,
    X_test
):

    print("\n" + "=" * 70)
    print("TRAINING BEHAVIOUR AUTOENCODER")
    print("=" * 70)

    # ------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=20,

        restore_best_weights=True,

        verbose=1

    )

    reduce_lr = ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=8,

        min_lr=1e-6,

        verbose=1

    )

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    history = autoencoder.fit(

        X_train,

        X_train,

        validation_data=(
            X_test,
            X_test
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        shuffle=False,

        callbacks=[
            early_stop,
            reduce_lr
        ],

        verbose=1

    )

    # ------------------------------------------------------
    # Final Loss
    # ------------------------------------------------------

    train_loss = history.history["loss"][-1]

    val_loss = history.history["val_loss"][-1]

    best_val_loss = min(
        history.history["val_loss"]
    )

    print("\nTraining Complete")

    print(f"Final Train Loss : {train_loss:.6f}")

    print(f"Final Validation Loss : {val_loss:.6f}")

    print(f"Best Validation Loss : {best_val_loss:.6f}")

    return history
# ==========================================================
# EXTRACT BEHAVIOUR EMBEDDINGS
# ==========================================================

def extract_behaviour_embeddings(

    autoencoder,
    encoder,
    scaler,
    train_median,
    df

):

    print("\n" + "=" * 70)
    print("EXTRACTING BEHAVIOUR EMBEDDINGS")
    print("=" * 70)

    # ------------------------------------------------------
    # Behaviour Matrix
    # ------------------------------------------------------

    X = df[BEHAVIOUR_FEATURES].copy()

    X = X.fillna(
        train_median
    )

    X_scaled = scaler.transform(X)

    # ------------------------------------------------------
    # Generate Embeddings
    # ------------------------------------------------------

    embeddings = encoder.predict(

        X_scaled,

        verbose=0

    )

    embedding_df = pd.DataFrame(

        embeddings,

        columns=[

            f"embedding_{i+1}"

            for i in range(LATENT_DIM)

        ]

    )

    # ------------------------------------------------------
    # Reconstruction Error
    # ------------------------------------------------------

    reconstructed = autoencoder.predict(

        X_scaled,

        verbose=0

    )

    reconstruction_error = np.mean(

        np.square(

            X_scaled - reconstructed

        ),

        axis=1

    )

    embedding_df["reconstruction_error"] = reconstruction_error

 

    # easier approach below

    # Use full autoencoder externally

    embedding_df["sales_date"] = df["sales_date"].values

    embedding_df["product"] = df["product"].values

    print()

    print("Embedding Shape")

    print(embedding_df.shape)

    print()

    print(embedding_df.head())

    return embedding_df
if __name__ == "__main__":

    print("=" * 70)
    print("LOADING DEMAND PHASE FEATURES")
    print("=" * 70)

    df = load_phase_features("Overall")

    data = prepare_behaviour_matrix(df)

    autoencoder, encoder = build_behaviour_autoencoder()

    history = train_behaviour_autoencoder(

        autoencoder,

        data["X_train"],

        data["X_test"]

    )

    embedding_df = extract_behaviour_embeddings(

        autoencoder,

        encoder,

        data["scaler"],

        data["train_median"],

        df

    )
    print("\nEmbedding Columns")
    print(embedding_df.columns.tolist())

    print("\nMissing Values")
    print(embedding_df.isna().sum())

    print("\nEmbedding Summary")
    print(
        embedding_df.describe(include="all")
    )
    print("\nSaving Behaviour Embeddings...")

    save_behaviour_embeddings(
        embedding_df
    )