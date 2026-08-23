from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STRUCTURE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "structure_split"
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

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

PREDICTION_FILE = (
    TABLES_DIR
    / "similarity_ood_predictions.csv"
)

OPTIMIZED_MODEL_FILE = (
    MODELS_DIR
    / "xgboost_optimized.joblib"
)

TOP_SHAP_FEATURES = 8


def create_output_directories():
    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_data():
    print("\nSTEP 1 - LOAD DATA")

    X_train = pd.read_csv(
        STRUCTURE_DATA_DIR
        / "X_train.csv"
    )

    X_test = pd.read_csv(
        STRUCTURE_DATA_DIR
        / "X_test.csv"
    )

    y_train = pd.read_csv(
        STRUCTURE_DATA_DIR
        / "y_train.csv"
    ).squeeze("columns")

    prediction_df = pd.read_csv(
        PREDICTION_FILE
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
        f"Prediction rows: "
        f"{len(prediction_df)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        prediction_df,
    )


def validate_data(
    X_test,
    prediction_df,
):
    print("\nSTEP 2 - VALIDATE DATA")

    if len(X_test) != len(prediction_df):
        raise ValueError(
            "X_test and prediction table "
            "lengths do not match."
        )

    expected_ids = np.arange(
        len(X_test)
    )

    actual_ids = prediction_df[
        "Sample_ID"
    ].to_numpy()

    if not np.array_equal(
        expected_ids,
        actual_ids,
    ):
        raise ValueError(
            "Sample_ID order does not match "
            "X_test."
        )

    print(
        "\nSample alignment check passed."
    )


def load_model_parameters():
    print(
        "\nSTEP 3 - LOAD MODEL PARAMETERS"
    )

    if not OPTIMIZED_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n"
            f"{OPTIMIZED_MODEL_FILE}"
        )

    optimized_model = joblib.load(
        OPTIMIZED_MODEL_FILE
    )

    parameters = (
        optimized_model.get_params()
    )

    print(
        "\nOptimized parameters loaded."
    )

    return parameters


def train_cluster_model(
    X_train,
    y_train,
    parameters,
):
    print(
        "\nSTEP 4 - TRAIN CLUSTER MODEL"
    )

    model = XGBRegressor(
        **parameters
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "\nCluster-based model trained."
    )

    return model


def select_case(
    prediction_df,
    minimum_similarity,
    maximum_similarity,
    target_similarity,
):
    subset = prediction_df[
        (
            prediction_df[
                "Max_Train_Tanimoto"
            ] >= minimum_similarity
        )
        &
        (
            prediction_df[
                "Max_Train_Tanimoto"
            ] < maximum_similarity
        )
    ].copy()

    if len(subset) == 0:
        raise ValueError(
            "No samples found in requested "
            "similarity range."
        )

    subset[
        "Distance_To_Target"
    ] = np.abs(
        subset[
            "Max_Train_Tanimoto"
        ]
        - target_similarity
    )

    subset = subset.sort_values(
        [
            "Distance_To_Target",
            "Absolute_Error_K",
        ],
        ascending=[
            True,
            True,
        ],
    )

    return subset.iloc[
        0
    ].copy()


def select_representative_cases(
    prediction_df,
):
    print(
        "\nSTEP 5 - SELECT REPRESENTATIVE CASES"
    )

    high_case = select_case(
        prediction_df,
        minimum_similarity=0.80,
        maximum_similarity=np.inf,
        target_similarity=0.85,
    )

    medium_case = select_case(
        prediction_df,
        minimum_similarity=0.60,
        maximum_similarity=0.80,
        target_similarity=0.70,
    )

    ood_case = select_case(
        prediction_df,
        minimum_similarity=-np.inf,
        maximum_similarity=0.50,
        target_similarity=0.40,
    )

    cases = [
        (
            "High",
            high_case,
        ),
        (
            "Medium",
            medium_case,
        ),
        (
            "OOD Warning",
            ood_case,
        ),
    ]

    print(
        "\nSelected cases:"
    )

    for reliability, case in cases:

        print(
            f"\n{reliability}"
        )

        print(
            f"Sample ID: "
            f"{int(case['Sample_ID'])}"
        )

        print(
            f"Similarity: "
            f"{case['Max_Train_Tanimoto']:.4f}"
        )

        print(
            f"Experimental Tg: "
            f"{case['Experimental_Tg_K']:.2f} K"
        )

        print(
            f"Predicted Tg: "
            f"{case['Predicted_Tg_K']:.2f} K"
        )

        print(
            f"Absolute error: "
            f"{case['Absolute_Error_K']:.2f} K"
        )

    return cases


def calculate_local_shap(
    model,
    X_test,
    cases,
):
    print(
        "\nSTEP 6 - CALCULATE LOCAL SHAP"
    )

    explainer = shap.TreeExplainer(
        model
    )

    case_results = []

    shap_tables = []

    for reliability, case in cases:

        sample_id = int(
            case[
                "Sample_ID"
            ]
        )

        X_new = X_test.iloc[
            [
                sample_id
            ]
        ].copy()

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

        top_shap.insert(
            0,
            "Reliability",
            reliability,
        )

        top_shap.insert(
            1,
            "Sample_ID",
            sample_id,
        )

        shap_tables.append(
            top_shap
        )

        result = {
            "Reliability":
                reliability,

            "Sample_ID":
                sample_id,

            "Experimental_Tg_K":
                case[
                    "Experimental_Tg_K"
                ],

            "Predicted_Tg_K":
                case[
                    "Predicted_Tg_K"
                ],

            "Absolute_Error_K":
                case[
                    "Absolute_Error_K"
                ],

            "Max_Train_Tanimoto":
                case[
                    "Max_Train_Tanimoto"
                ],
        }

        if "PSMILES" in case.index:
            result[
                "PSMILES"
            ] = case[
                "PSMILES"
            ]

        if (
            "Nearest_Train_PSMILES"
            in case.index
        ):
            result[
                "Nearest_Train_PSMILES"
            ] = case[
                "Nearest_Train_PSMILES"
            ]

        case_results.append(
            result
        )

        print(
            f"\n{reliability} case "
            f"top SHAP features:"
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

    case_df = pd.DataFrame(
        case_results
    )

    all_shap_df = pd.concat(
        shap_tables,
        ignore_index=True,
    )

    return (
        case_df,
        all_shap_df,
    )


def save_results(
    case_df,
    all_shap_df,
):
    print(
        "\nSTEP 7 - SAVE CASE STUDIES"
    )

    case_file = (
        TABLES_DIR
        / "prediction_case_studies.csv"
    )

    shap_file = (
        TABLES_DIR
        / "prediction_case_studies_shap.csv"
    )

    case_df.to_csv(
        case_file,
        index=False,
    )

    all_shap_df.to_csv(
        shap_file,
        index=False,
    )

    print(
        "\nCase study table saved to:"
    )

    print(
        case_file
    )

    print(
        "\nCase SHAP table saved to:"
    )

    print(
        shap_file
    )


def create_case_comparison_figure(
    case_df,
):
    print(
        "\nSTEP 8 - CREATE CASE COMPARISON FIGURE"
    )

    positions = np.arange(
        len(case_df)
    )

    width = 0.35

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        positions - width / 2,
        case_df[
            "Experimental_Tg_K"
        ],
        width=width,
        label="Experimental",
    )

    plt.bar(
        positions + width / 2,
        case_df[
            "Predicted_Tg_K"
        ],
        width=width,
        label="Predicted",
    )

    plt.xticks(
        positions,
        case_df[
            "Reliability"
        ],
    )

    plt.ylabel(
        "Tg (K)"
    )

    plt.xlabel(
        "Prediction Reliability"
    )

    plt.title(
        "Representative Polymer Prediction Cases"
    )

    plt.legend()

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "prediction_case_studies.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nFigure saved to:"
    )

    print(
        output_file
    )


def main():
    print(
        "\nPOLYMER Tg PREDICTION CASE STUDIES"
    )

    create_output_directories()

    (
        X_train,
        X_test,
        y_train,
        prediction_df,
    ) = load_data()

    validate_data(
        X_test,
        prediction_df,
    )

    parameters = (
        load_model_parameters()
    )

    model = train_cluster_model(
        X_train,
        y_train,
        parameters,
    )

    cases = (
        select_representative_cases(
            prediction_df
        )
    )

    (
        case_df,
        all_shap_df,
    ) = calculate_local_shap(
        model,
        X_test,
        cases,
    )

    save_results(
        case_df,
        all_shap_df,
    )

    create_case_comparison_figure(
        case_df
    )

    print(
        "\nCASE STUDY SUMMARY"
    )

    print(
        case_df[
            [
                "Reliability",
                "Sample_ID",
                "Experimental_Tg_K",
                "Predicted_Tg_K",
                "Absolute_Error_K",
                "Max_Train_Tanimoto",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nPREDICTION CASE STUDIES COMPLETED"
    )


if __name__ == "__main__":
    main()