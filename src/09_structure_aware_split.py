from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "polymer_tg_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "structure_split"
)

TARGET_COLUMN = "labels.Exp_Tg(K)"

STRUCTURE_COLUMN = "PSMILES"

CLASS_COLUMN = "meta.polymer_class"

BIGSMILES_COLUMN = "BIGSMILES"

TEST_SIZE = 0.20

FINGERPRINT_RADIUS = 2

FINGERPRINT_BITS = 2048

SIMILARITY_THRESHOLD = 0.60

RANDOM_STATE = 42


def load_data():
    print("\nSTEP 1 - LOAD DATA")

    df = pd.read_csv(
        DATA_FILE
    )

    print(f"\nSamples: {len(df)}")

    return df


def create_fingerprints(df):
    print("\nSTEP 2 - CREATE MORGAN FINGERPRINTS")

    fingerprints = []

    valid_indices = []

    invalid_indices = []

    for index, smiles in enumerate(
        df[STRUCTURE_COLUMN]
    ):
        mol = Chem.MolFromSmiles(
            str(smiles)
        )

        if mol is None:
            invalid_indices.append(index)
            continue

        fingerprint = (
            AllChem.GetMorganGenerator(
                radius=FINGERPRINT_RADIUS,
                fpSize=FINGERPRINT_BITS,
            ).GetFingerprint(mol)
        )

        fingerprints.append(
            fingerprint
        )

        valid_indices.append(
            index
        )

    print(
        f"\nValid structures: "
        f"{len(valid_indices)}"
    )

    print(
        f"Invalid structures: "
        f"{len(invalid_indices)}"
    )

    return (
        fingerprints,
        valid_indices,
        invalid_indices,
    )


def calculate_distances(
    fingerprints,
):
    print("\nSTEP 3 - CALCULATE STRUCTURE DISTANCES")

    distances = []

    for i in range(1, len(fingerprints)):
        similarities = (
            DataStructs.BulkTanimotoSimilarity(
                fingerprints[i],
                fingerprints[:i],
            )
        )

        distances.extend(
            [
                1 - similarity
                for similarity in similarities
            ]
        )

    print(
        f"\nPairwise distances calculated: "
        f"{len(distances)}"
    )

    return distances


def cluster_structures(
    fingerprints,
    distances,
):
    print("\nSTEP 4 - BUTINA CLUSTERING")

    distance_threshold = (
        1 - SIMILARITY_THRESHOLD
    )

    clusters = Butina.ClusterData(
        distances,
        len(fingerprints),
        distance_threshold,
        isDistData=True,
    )

    clusters = [
        list(cluster)
        for cluster in clusters
    ]

    cluster_sizes = [
        len(cluster)
        for cluster in clusters
    ]

    print(
        f"\nNumber of clusters: "
        f"{len(clusters)}"
    )

    print(
        f"Largest cluster: "
        f"{max(cluster_sizes)}"
    )

    print(
        f"Median cluster size: "
        f"{np.median(cluster_sizes):.1f}"
    )

    return clusters


def assign_train_test(
    clusters,
    total_samples,
):
    print("\nSTEP 5 - ASSIGN CLUSTERS TO TRAIN AND TEST")

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    cluster_order = np.arange(
        len(clusters)
    )

    rng.shuffle(
        cluster_order
    )

    target_test_samples = int(
        total_samples
        * TEST_SIZE
    )

    train_cluster_indices = []

    test_cluster_indices = []

    test_count = 0

    for cluster_index in cluster_order:

        cluster = clusters[
            cluster_index
        ]

        if test_count < target_test_samples:
            test_cluster_indices.append(
                cluster_index
            )

            test_count += len(
                cluster
            )

        else:
            train_cluster_indices.append(
                cluster_index
            )

    train_positions = []

    test_positions = []

    for cluster_index in train_cluster_indices:
        train_positions.extend(
            clusters[cluster_index]
        )

    for cluster_index in test_cluster_indices:
        test_positions.extend(
            clusters[cluster_index]
        )

    print(
        f"\nTraining structures: "
        f"{len(train_positions)}"
    )

    print(
        f"Test structures: "
        f"{len(test_positions)}"
    )

    return (
        train_positions,
        test_positions,
    )


def create_split(
    df,
    valid_indices,
    train_positions,
    test_positions,
):
    print("\nSTEP 6 - CREATE MODEL DATA")

    valid_df = df.iloc[
        valid_indices
    ].reset_index(
        drop=True
    )

    train_df = valid_df.iloc[
        train_positions
    ].copy()

    test_df = valid_df.iloc[
        test_positions
    ].copy()

    excluded_columns = [
        TARGET_COLUMN,
        STRUCTURE_COLUMN,
        CLASS_COLUMN,
        BIGSMILES_COLUMN,
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    X_train = train_df[
        feature_columns
    ].copy()

    X_test = test_df[
        feature_columns
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].copy()

    metadata_train = train_df[
        [
            STRUCTURE_COLUMN,
            CLASS_COLUMN,
            BIGSMILES_COLUMN,
        ]
    ].copy()

    metadata_test = test_df[
        [
            STRUCTURE_COLUMN,
            CLASS_COLUMN,
            BIGSMILES_COLUMN,
        ]
    ].copy()

    print(
        f"\nX_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"X_test shape: "
        f"{X_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test,
    )


def save_data(
    X_train,
    X_test,
    y_train,
    y_test,
    metadata_train,
    metadata_test,
):
    print("\nSTEP 7 - SAVE STRUCTURE-AWARE SPLIT")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train.to_csv(
        OUTPUT_DIR / "X_train.csv",
        index=False,
    )

    X_test.to_csv(
        OUTPUT_DIR / "X_test.csv",
        index=False,
    )

    y_train.to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False,
    )

    y_test.to_csv(
        OUTPUT_DIR / "y_test.csv",
        index=False,
    )

    metadata_train.to_csv(
        OUTPUT_DIR / "metadata_train.csv",
        index=False,
    )

    metadata_test.to_csv(
        OUTPUT_DIR / "metadata_test.csv",
        index=False,
    )

    print("\nStructure-aware data saved to:")
    print(OUTPUT_DIR)


def main():
    print("\nPOLYMER Tg STRUCTURE-AWARE SPLIT")

    df = load_data()

    (
        fingerprints,
        valid_indices,
        invalid_indices,
    ) = create_fingerprints(
        df
    )

    distances = calculate_distances(
        fingerprints
    )

    clusters = cluster_structures(
        fingerprints,
        distances,
    )

    (
        train_positions,
        test_positions,
    ) = assign_train_test(
        clusters,
        len(fingerprints),
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test,
    ) = create_split(
        df,
        valid_indices,
        train_positions,
        test_positions,
    )

    save_data(
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test,
    )

    print("\nSTRUCTURE-AWARE SPLIT COMPLETED")


if __name__ == "__main__":
    main()