from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "model"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

TABLES_DIR = (
    RESULTS_DIR
    / "tables"
)

MODEL_FILE = (
    MODELS_DIR
    / "xgboost_optimized.joblib"
)

TRAIN_METADATA_FILE = (
    MODEL_DATA_DIR
    / "metadata_train.csv"
)

TEST_METADATA_FILE = (
    MODEL_DATA_DIR
    / "metadata_test.csv"
)

X_TRAIN_FILE = (
    MODEL_DATA_DIR
    / "X_train.csv"
)

X_TEST_FILE = (
    MODEL_DATA_DIR
    / "X_test.csv"
)

Y_TEST_FILE = (
    MODEL_DATA_DIR
    / "y_test.csv"
)

FINGERPRINT_RADIUS = 2

FINGERPRINT_BITS = 2048

SAMPLE_INDEX = 0

TOP_SHAP_FEATURES = 10


def create_output_directory():
    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_data():
    print("\nSTEP 1 - LOAD MODEL DATA")

    X_train = pd.read_csv(
        X_TRAIN_FILE
    )

    X_test = pd.read_csv(
        X_TEST_FILE
    )

    y_test = pd.read_csv(
        Y_TEST_FILE
    ).squeeze("columns")

    train_metadata = pd.read_csv(
        TRAIN_METADATA_FILE
    )

    test_metadata = pd.read_csv(
        TEST_METADATA_FILE
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Test samples: "
        f"{len(X_test)}"
    )

    print(
        f"Features: "
        f"{X_train.shape[1]}"
    )

    return (
        X_train,
        X_test,
        y_test,
        train_metadata,
        test_metadata,
    )


def load_model():
    print("\nSTEP 2 - LOAD XGBOOST MODEL")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "\nOptimized XGBoost loaded."
    )

    return model


def select_new_polymer(
    X_test,
    y_test,
    test_metadata,
):
    print("\nSTEP 3 - SELECT NEW POLYMER")

    if SAMPLE_INDEX >= len(X_test):
        raise IndexError(
            "SAMPLE_INDEX is larger than "
            "the test dataset."
        )

    X_new = X_test.iloc[
        [
            SAMPLE_INDEX
        ]
    ].copy()

    experimental_tg = float(
        y_test.iloc[
            SAMPLE_INDEX
        ]
    )

    metadata = test_metadata.iloc[
        SAMPLE_INDEX
    ].copy()

    print(
        f"\nSample index: "
        f"{SAMPLE_INDEX}"
    )

    if "PSMILES" in metadata.index:
        print(
            f"\nPSMILES:\n"
            f"{metadata['PSMILES']}"
        )

    if (
        "meta.polymer_class"
        in metadata.index
    ):
        print(
            f"\nPolymer class: "
            f"{metadata['meta.polymer_class']}"
        )

    print(
        f"\nExperimental Tg: "
        f"{experimental_tg:.2f} K"
    )

    return (
        X_new,
        experimental_tg,
        metadata,
    )


def predict_tg(
    model,
    X_new,
):
    print("\nSTEP 4 - PREDICT Tg")

    predicted_tg = float(
        model.predict(
            X_new
        )[0]
    )

    print(
        f"\nPredicted Tg: "
        f"{predicted_tg:.2f} K"
    )

    return predicted_tg


def create_fingerprint_generator():
    generator = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=FINGERPRINT_RADIUS,
            fpSize=FINGERPRINT_BITS,
        )
    )

    return generator


def calculate_similarity(
    new_smiles,
    train_metadata,
):
    print(
        "\nSTEP 5 - CALCULATE STRUCTURAL SIMILARITY"
    )

    generator = (
        create_fingerprint_generator()
    )

    new_molecule = Chem.MolFromSmiles(
        str(new_smiles)
    )

    if new_molecule is None:
        raise ValueError(
            f"Invalid PSMILES:\n"
            f"{new_smiles}"
        )

    new_fingerprint = (
        generator.GetFingerprint(
            new_molecule
        )
    )

    train_fingerprints = []

    valid_train_indices = []

    for index, smiles in enumerate(
        train_metadata["PSMILES"]
    ):

        molecule = Chem.MolFromSmiles(
            str(smiles)
        )

        if molecule is None:
            continue

        fingerprint = (
            generator.GetFingerprint(
                molecule
            )
        )

        train_fingerprints.append(
            fingerprint
        )

        valid_train_indices.append(
            index
        )

    similarities = np.array(
        DataStructs.BulkTanimotoSimilarity(
            new_fingerprint,
            train_fingerprints,
        )
    )

    best_position = int(
        np.argmax(
            similarities
        )
    )

    maximum_similarity = float(
        similarities[
            best_position
        ]
    )

    nearest_train_index = (
        valid_train_indices[
            best_position
        ]
    )

    nearest_train_smiles = (
        train_metadata.iloc[
            nearest_train_index
        ][
            "PSMILES"
        ]
    )

    print(
        f"\nMaximum train similarity: "
        f"{maximum_similarity:.4f}"
    )

    print(
        f"\nNearest training polymer:\n"
        f"{nearest_train_smiles}"
    )

    return (
        maximum_similarity,
        nearest_train_smiles,
    )


def assign_reliability(
    similarity,
):
    print(
        "\nSTEP 6 - ASSIGN RELIABILITY LEVEL"
    )

    if similarity >= 0.80:
        reliability = "High"

    elif similarity >= 0.60:
        reliability = "Medium"

    elif similarity >= 0.50:
        reliability = "Low"

    else:
        reliability = "OOD Warning"

    print(
        f"\nReliability level: "
        f"{reliability}"
    )

    return reliability


def calculate_local_shap(
    model,
    X_new,
):
    print(
        "\nSTEP 7 - LOCAL SHAP EXPLANATION"
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        X_new
    )

    local_values = (
        shap_values.values[
            0
        ]
    )

    shap_df = pd.DataFrame(
        {
            "Feature":
                X_new.columns,

            "Feature_Value":
                X_new.iloc[
                    0
                ].values,

            "SHAP_Value_K":
                local_values,

            "Absolute_SHAP":
                np.abs(
                    local_values
                ),
        }
    )

    shap_df = shap_df.sort_values(
        "Absolute_SHAP",
        ascending=False,
    ).reset_index(
        drop=True
    )

    top_shap = shap_df.head(
        TOP_SHAP_FEATURES
    ).copy()

    top_shap.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(top_shap) + 1,
        ),
    )

    print(
        "\nTop local SHAP features:"
    )

    print(
        top_shap[
            [
                "Rank",
                "Feature",
                "Feature_Value",
                "SHAP_Value_K",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    return top_shap


def calculate_prediction_error(
    experimental_tg,
    predicted_tg,
):
    print(
        "\nSTEP 8 - PREDICTION ERROR"
    )

    residual = (
        predicted_tg
        - experimental_tg
    )

    absolute_error = abs(
        residual
    )

    print(
        f"\nPrediction error: "
        f"{residual:.2f} K"
    )

    print(
        f"Absolute error: "
        f"{absolute_error:.2f} K"
    )

    return (
        residual,
        absolute_error,
    )


def save_prediction_report(
    metadata,
    experimental_tg,
    predicted_tg,
    residual,
    absolute_error,
    similarity,
    reliability,
    nearest_train_smiles,
    top_shap,
):
    print(
        "\nSTEP 9 - SAVE PREDICTION REPORT"
    )

    report = {
        "Sample_Index":
            SAMPLE_INDEX,

        "Experimental_Tg_K":
            experimental_tg,

        "Predicted_Tg_K":
            predicted_tg,

        "Residual_K":
            residual,

        "Absolute_Error_K":
            absolute_error,

        "Max_Train_Tanimoto":
            similarity,

        "Reliability":
            reliability,

        "Nearest_Train_PSMILES":
            nearest_train_smiles,
    }

    if "PSMILES" in metadata.index:
        report[
            "PSMILES"
        ] = metadata[
            "PSMILES"
        ]

    if (
        "meta.polymer_class"
        in metadata.index
    ):
        report[
            "Polymer_Class"
        ] = metadata[
            "meta.polymer_class"
        ]

    report_df = pd.DataFrame(
        [
            report
        ]
    )

    report_file = (
        TABLES_DIR
        / "new_polymer_prediction_report.csv"
    )

    shap_file = (
        TABLES_DIR
        / "new_polymer_local_shap.csv"
    )

    report_df.to_csv(
        report_file,
        index=False,
    )

    top_shap.to_csv(
        shap_file,
        index=False,
    )

    print(
        "\nPrediction report saved to:"
    )

    print(
        report_file
    )

    print(
        "\nLocal SHAP table saved to:"
    )

    print(
        shap_file
    )


def main():
    print(
        "\nPOLYMER Tg NEW POLYMER PREDICTION"
    )

    create_output_directory()

    (
        X_train,
        X_test,
        y_test,
        train_metadata,
        test_metadata,
    ) = load_data()

    model = load_model()

    (
        X_new,
        experimental_tg,
        metadata,
    ) = select_new_polymer(
        X_test,
        y_test,
        test_metadata,
    )

    predicted_tg = predict_tg(
        model,
        X_new,
    )

    new_smiles = metadata[
        "PSMILES"
    ]

    (
        similarity,
        nearest_train_smiles,
    ) = calculate_similarity(
        new_smiles,
        train_metadata,
    )

    reliability = assign_reliability(
        similarity
    )

    top_shap = calculate_local_shap(
        model,
        X_new,
    )

    (
        residual,
        absolute_error,
    ) = calculate_prediction_error(
        experimental_tg,
        predicted_tg,
    )

    save_prediction_report(
        metadata,
        experimental_tg,
        predicted_tg,
        residual,
        absolute_error,
        similarity,
        reliability,
        nearest_train_smiles,
        top_shap,
    )

    print(
        "\nPREDICTION SUMMARY"
    )

    print(
        f"\nExperimental Tg: "
        f"{experimental_tg:.2f} K"
    )

    print(
        f"Predicted Tg: "
        f"{predicted_tg:.2f} K"
    )

    print(
        f"Absolute error: "
        f"{absolute_error:.2f} K"
    )

    print(
        f"Maximum train similarity: "
        f"{similarity:.4f}"
    )

    print(
        f"Reliability: "
        f"{reliability}"
    )

    print(
        "\nNEW POLYMER PREDICTION COMPLETED"
    )


if __name__ == "__main__":
    main()