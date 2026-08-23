from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "structure_split"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

TABLES_DIR = (
    RESULTS_DIR
    / "tables"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

TRAIN_METADATA_FILE = (
    DATA_DIR
    / "metadata_train.csv"
)

TEST_METADATA_FILE = (
    DATA_DIR
    / "metadata_test.csv"
)

STRUCTURE_COLUMN = "PSMILES"

FINGERPRINT_RADIUS = 2

FINGERPRINT_BITS = 2048


def create_output_directories():
    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_metadata():
    print("\nSTEP 1 - LOAD STRUCTURE METADATA")

    train_metadata = pd.read_csv(
        TRAIN_METADATA_FILE
    )

    test_metadata = pd.read_csv(
        TEST_METADATA_FILE
    )

    train_metadata = train_metadata.reset_index(
        drop=True
    )

    test_metadata = test_metadata.reset_index(
        drop=True
    )

    train_metadata.insert(
        0,
        "Sample_ID",
        np.arange(
            len(train_metadata)
        ),
    )

    test_metadata.insert(
        0,
        "Sample_ID",
        np.arange(
            len(test_metadata)
        ),
    )

    print(
        f"\nTraining structures: "
        f"{len(train_metadata)}"
    )

    print(
        f"Test structures: "
        f"{len(test_metadata)}"
    )

    print(
        f"\nUnique test Sample_ID: "
        f"{test_metadata['Sample_ID'].nunique()}"
    )

    return (
        train_metadata,
        test_metadata,
    )


def create_fingerprints(
    smiles_series,
):
    generator = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=FINGERPRINT_RADIUS,
            fpSize=FINGERPRINT_BITS,
        )
    )

    fingerprints = []

    for smiles in smiles_series:

        molecule = Chem.MolFromSmiles(
            str(smiles)
        )

        if molecule is None:
            raise ValueError(
                f"Invalid PSMILES found:\n"
                f"{smiles}"
            )

        fingerprint = (
            generator.GetFingerprint(
                molecule
            )
        )

        fingerprints.append(
            fingerprint
        )

    return fingerprints


def calculate_max_train_similarity(
    train_fingerprints,
    test_fingerprints,
):
    print(
        "\nSTEP 2 - CALCULATE TEST TO TRAIN "
        "SIMILARITY"
    )

    max_similarities = []

    nearest_train_indices = []

    for index, test_fp in enumerate(
        test_fingerprints,
        start=1,
    ):

        similarities = np.array(
            DataStructs.BulkTanimotoSimilarity(
                test_fp,
                train_fingerprints,
            )
        )

        nearest_index = int(
            np.argmax(
                similarities
            )
        )

        maximum_similarity = float(
            similarities[
                nearest_index
            ]
        )

        max_similarities.append(
            maximum_similarity
        )

        nearest_train_indices.append(
            nearest_index
        )

        if index % 250 == 0:
            print(
                f"Processed test structures: "
                f"{index}"
            )

    return (
        np.array(
            max_similarities
        ),
        np.array(
            nearest_train_indices
        ),
    )


def summarize_similarity(
    similarities,
):
    print("\nSTEP 3 - SIMILARITY SUMMARY")

    summary = {
        "Mean": np.mean(
            similarities
        ),
        "Median": np.median(
            similarities
        ),
        "Minimum": np.min(
            similarities
        ),
        "Maximum": np.max(
            similarities
        ),
        "P90": np.percentile(
            similarities,
            90,
        ),
        "P95": np.percentile(
            similarities,
            95,
        ),
        "Fraction_ge_0.60": np.mean(
            similarities >= 0.60
        ),
        "Fraction_ge_0.70": np.mean(
            similarities >= 0.70
        ),
        "Fraction_ge_0.80": np.mean(
            similarities >= 0.80
        ),
        "Fraction_ge_0.90": np.mean(
            similarities >= 0.90
        ),
    }

    for metric, value in summary.items():

        if "Fraction" in metric:
            print(
                f"{metric}: "
                f"{value:.2%}"
            )

        else:
            print(
                f"{metric}: "
                f"{value:.4f}"
            )

    return pd.DataFrame(
        [
            summary
        ]
    )


def create_similarity_table(
    train_metadata,
    test_metadata,
    similarities,
    nearest_train_indices,
):
    print(
        "\nSTEP 4 - CREATE ALIGNED "
        "SIMILARITY TABLE"
    )

    result = test_metadata.copy()

    result[
        "Max_Train_Tanimoto"
    ] = similarities

    result[
        "Nearest_Train_Index"
    ] = nearest_train_indices

    nearest_train_smiles = []

    for train_index in nearest_train_indices:

        nearest_train_smiles.append(
            train_metadata.iloc[
                train_index
            ][
                STRUCTURE_COLUMN
            ]
        )

    result[
        "Nearest_Train_PSMILES"
    ] = nearest_train_smiles

    print(
        f"\nSimilarity rows: "
        f"{len(result)}"
    )

    print(
        f"Unique Sample_ID: "
        f"{result['Sample_ID'].nunique()}"
    )

    print(
        f"Missing similarities: "
        f"{result['Max_Train_Tanimoto'].isna().sum()}"
    )

    return result


def save_results(
    summary_df,
    similarity_table,
):
    print("\nSTEP 5 - SAVE AUDIT RESULTS")

    summary_file = (
        TABLES_DIR
        / "structure_split_similarity_summary.csv"
    )

    aligned_file = (
        TABLES_DIR
        / "structure_split_test_similarity_aligned.csv"
    )

    ranked_file = (
        TABLES_DIR
        / "structure_split_test_similarity_ranked.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    similarity_table.to_csv(
        aligned_file,
        index=False,
    )

    ranked_table = (
        similarity_table
        .sort_values(
            "Max_Train_Tanimoto",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    ranked_table.to_csv(
        ranked_file,
        index=False,
    )

    print("\nSummary saved to:")
    print(summary_file)

    print("\nAligned similarity table saved to:")
    print(aligned_file)

    print("\nRanked similarity table saved to:")
    print(ranked_file)


def plot_similarity_distribution(
    similarities,
):
    print(
        "\nSTEP 6 - CREATE SIMILARITY "
        "DISTRIBUTION"
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        similarities,
        bins=30,
    )

    plt.xlabel(
        "Maximum Tanimoto Similarity "
        "to Training Set"
    )

    plt.ylabel(
        "Number of Test Polymers"
    )

    plt.title(
        "Test to Train Structural Similarity"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "structure_split_similarity_distribution.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nFigure saved to:")
    print(output_file)


def main():
    print(
        "\nPOLYMER Tg STRUCTURE SPLIT AUDIT V2"
    )

    create_output_directories()

    (
        train_metadata,
        test_metadata,
    ) = load_metadata()

    print(
        "\nCreating training fingerprints..."
    )

    train_fingerprints = (
        create_fingerprints(
            train_metadata[
                STRUCTURE_COLUMN
            ]
        )
    )

    print(
        "Creating test fingerprints..."
    )

    test_fingerprints = (
        create_fingerprints(
            test_metadata[
                STRUCTURE_COLUMN
            ]
        )
    )

    (
        similarities,
        nearest_train_indices,
    ) = calculate_max_train_similarity(
        train_fingerprints,
        test_fingerprints,
    )

    summary_df = summarize_similarity(
        similarities
    )

    similarity_table = (
        create_similarity_table(
            train_metadata,
            test_metadata,
            similarities,
            nearest_train_indices,
        )
    )

    save_results(
        summary_df,
        similarity_table,
    )

    plot_similarity_distribution(
        similarities
    )

    print(
        "\nSTRUCTURE SPLIT AUDIT V2 COMPLETED"
    )


if __name__ == "__main__":
    main()