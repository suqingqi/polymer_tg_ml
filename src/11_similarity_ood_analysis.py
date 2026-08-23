from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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

SIMILARITY_FILE = (
    TABLES_DIR
    / "structure_split_test_similarity_aligned.csv"
)

OPTIMIZED_MODEL_FILE = (
    MODELS_DIR
    / "xgboost_optimized.joblib"
)


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
    print(
        "\nSTEP 1 - LOAD STRUCTURE SPLIT DATA"
    )

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

    y_test = pd.read_csv(
        STRUCTURE_DATA_DIR
        / "y_test.csv"
    ).squeeze("columns")

    similarity_df = pd.read_csv(
        SIMILARITY_FILE
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
        f"Similarity rows: "
        f"{len(similarity_df)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        similarity_df,
    )


def validate_alignment(
    X_test,
    y_test,
    similarity_df,
):
    print(
        "\nSTEP 2 - VALIDATE DATA ALIGNMENT"
    )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test lengths do not match."
        )

    if len(X_test) != len(similarity_df):
        raise ValueError(
            "X_test and similarity table lengths "
            "do not match."
        )

    expected_ids = np.arange(
        len(X_test)
    )

    actual_ids = similarity_df[
        "Sample_ID"
    ].to_numpy()

    if not np.array_equal(
        expected_ids,
        actual_ids,
    ):
        raise ValueError(
            "Sample_ID order does not match "
            "X_test row order."
        )

    if similarity_df[
        "Sample_ID"
    ].duplicated().any():
        raise ValueError(
            "Duplicated Sample_ID found."
        )

    if similarity_df[
        "Max_Train_Tanimoto"
    ].isna().any():
        raise ValueError(
            "Missing similarity values found."
        )

    print(
        "\nSample alignment check passed."
    )

    print(
        f"Validated samples: "
        f"{len(X_test)}"
    )


def load_parameters():
    print(
        "\nSTEP 3 - LOAD OPTIMIZED PARAMETERS"
    )

    if not OPTIMIZED_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Optimized model not found:\n"
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
        "\nSTEP 4 - TRAIN CLUSTER-BASED MODEL"
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


def create_prediction_table(
    model,
    X_test,
    y_test,
    similarity_df,
):
    print(
        "\nSTEP 5 - CREATE PREDICTION TABLE"
    )

    predictions = model.predict(
        X_test
    )

    result_df = pd.DataFrame(
        {
            "Sample_ID":
                similarity_df[
                    "Sample_ID"
                ].values,

            "Experimental_Tg_K":
                y_test.to_numpy(),

            "Predicted_Tg_K":
                predictions,

            "Max_Train_Tanimoto":
                similarity_df[
                    "Max_Train_Tanimoto"
                ].values,
        }
    )

    result_df[
        "Residual_K"
    ] = (
        result_df[
            "Predicted_Tg_K"
        ]
        - result_df[
            "Experimental_Tg_K"
        ]
    )

    result_df[
        "Absolute_Error_K"
    ] = np.abs(
        result_df[
            "Residual_K"
        ]
    )

    if "PSMILES" in similarity_df.columns:
        result_df[
            "PSMILES"
        ] = similarity_df[
            "PSMILES"
        ].values

    if (
        "Nearest_Train_PSMILES"
        in similarity_df.columns
    ):
        result_df[
            "Nearest_Train_PSMILES"
        ] = similarity_df[
            "Nearest_Train_PSMILES"
        ].values

    print(
        f"\nPrediction rows: "
        f"{len(result_df)}"
    )

    return result_df


def assign_similarity_groups(
    result_df,
):
    print(
        "\nSTEP 6 - ASSIGN SIMILARITY GROUPS"
    )

    bins = [
        -np.inf,
        0.50,
        0.60,
        0.70,
        0.80,
        np.inf,
    ]

    labels = [
        "< 0.50",
        "0.50 to 0.60",
        "0.60 to 0.70",
        "0.70 to 0.80",
        ">= 0.80",
    ]

    result_df[
        "Similarity_Group"
    ] = pd.cut(
        result_df[
            "Max_Train_Tanimoto"
        ],
        bins=bins,
        labels=labels,
        right=False,
    )

    print(
        "\nSimilarity group counts:"
    )

    print(
        result_df[
            "Similarity_Group"
        ]
        .value_counts(
            sort=False
        )
    )

    return result_df


def calculate_group_metrics(
    result_df,
):
    print(
        "\nSTEP 7 - CALCULATE GROUP METRICS"
    )

    groups = [
        "< 0.50",
        "0.50 to 0.60",
        "0.60 to 0.70",
        "0.70 to 0.80",
        ">= 0.80",
    ]

    rows = []

    for group in groups:

        subset = result_df[
            result_df[
                "Similarity_Group"
            ] == group
        ]

        if len(subset) == 0:
            continue

        y_true = subset[
            "Experimental_Tg_K"
        ]

        y_pred = subset[
            "Predicted_Tg_K"
        ]

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        if len(subset) >= 2:
            r2 = r2_score(
                y_true,
                y_pred,
            )
        else:
            r2 = np.nan

        rows.append(
            {
                "Similarity_Group":
                    group,

                "Samples":
                    len(subset),

                "Mean_Similarity":
                    subset[
                        "Max_Train_Tanimoto"
                    ].mean(),

                "R2":
                    r2,

                "MAE_K":
                    mae,

                "RMSE_K":
                    rmse,
            }
        )

    metrics_df = pd.DataFrame(
        rows
    )

    print(
        "\nSimilarity-based performance:"
    )

    print(
        metrics_df.round(
            4
        ).to_string(
            index=False
        )
    )

    return metrics_df


def calculate_error_correlation(
    result_df,
):
    print(
        "\nSTEP 8 - SIMILARITY ERROR CORRELATION"
    )

    similarity = result_df[
        "Max_Train_Tanimoto"
    ].to_numpy()

    absolute_error = result_df[
        "Absolute_Error_K"
    ].to_numpy()

    pearson_result = pearsonr(
        similarity,
        absolute_error,
    )

    spearman_result = spearmanr(
        similarity,
        absolute_error,
    )

    print(
        f"\nPearson correlation: "
        f"{pearson_result.statistic:.4f}"
    )

    print(
        f"Pearson p-value: "
        f"{pearson_result.pvalue:.6g}"
    )

    print(
        f"\nSpearman correlation: "
        f"{spearman_result.statistic:.4f}"
    )

    print(
        f"Spearman p-value: "
        f"{spearman_result.pvalue:.6g}"
    )

    correlation_df = pd.DataFrame(
        [
            {
                "Method": "Pearson",
                "Correlation":
                    pearson_result.statistic,
                "P_value":
                    pearson_result.pvalue,
            },
            {
                "Method": "Spearman",
                "Correlation":
                    spearman_result.statistic,
                "P_value":
                    spearman_result.pvalue,
            },
        ]
    )

    return correlation_df


def calculate_ood_metrics(
    result_df,
):
    print(
        "\nSTEP 9 - LOW SIMILARITY OOD SUMMARY"
    )

    thresholds = [
        0.60,
        0.50,
        0.40,
    ]

    rows = []

    for threshold in thresholds:

        subset = result_df[
            result_df[
                "Max_Train_Tanimoto"
            ] < threshold
        ]

        if len(subset) < 2:
            continue

        y_true = subset[
            "Experimental_Tg_K"
        ]

        y_pred = subset[
            "Predicted_Tg_K"
        ]

        r2 = r2_score(
            y_true,
            y_pred,
        )

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        rows.append(
            {
                "OOD_Definition":
                    f"Tanimoto < {threshold:.2f}",

                "Samples":
                    len(subset),

                "R2":
                    r2,

                "MAE_K":
                    mae,

                "RMSE_K":
                    rmse,
            }
        )

    ood_df = pd.DataFrame(
        rows
    )

    print(
        "\nOOD performance:"
    )

    if len(ood_df) > 0:
        print(
            ood_df.round(
                4
            ).to_string(
                index=False
            )
        )

    return ood_df


def save_results(
    result_df,
    metrics_df,
    correlation_df,
    ood_df,
):
    print(
        "\nSTEP 10 - SAVE RESULTS"
    )

    result_df.to_csv(
        TABLES_DIR
        / "similarity_ood_predictions.csv",
        index=False,
    )

    metrics_df.to_csv(
        TABLES_DIR
        / "similarity_ood_group_metrics.csv",
        index=False,
    )

    correlation_df.to_csv(
        TABLES_DIR
        / "similarity_error_correlation.csv",
        index=False,
    )

    ood_df.to_csv(
        TABLES_DIR
        / "low_similarity_ood_metrics.csv",
        index=False,
    )

    print(
        "\nOOD analysis tables saved."
    )


def plot_error_vs_similarity(
    result_df,
):
    print(
        "\nSTEP 11 - CREATE ERROR "
        "SIMILARITY FIGURE"
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        result_df[
            "Max_Train_Tanimoto"
        ],
        result_df[
            "Absolute_Error_K"
        ],
        alpha=0.4,
    )

    plt.xlabel(
        "Maximum Tanimoto Similarity "
        "to Training Set"
    )

    plt.ylabel(
        "Absolute Prediction Error (K)"
    )

    plt.title(
        "Structural Similarity vs "
        "Prediction Error"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "similarity_vs_prediction_error.png"
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


def plot_mae_by_similarity(
    metrics_df,
):
    print(
        "\nSTEP 12 - CREATE GROUP MAE FIGURE"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        metrics_df[
            "Similarity_Group"
        ],
        metrics_df[
            "MAE_K"
        ],
    )

    plt.xlabel(
        "Maximum Train Similarity"
    )

    plt.ylabel(
        "MAE (K)"
    )

    plt.title(
        "Prediction Error Across "
        "Structural Similarity"
    )

    plt.xticks(
        rotation=25
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "mae_by_similarity_group.png"
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
        "\nPOLYMER Tg SIMILARITY OOD ANALYSIS V2"
    )

    create_output_directories()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        similarity_df,
    ) = load_data()

    validate_alignment(
        X_test,
        y_test,
        similarity_df,
    )

    parameters = load_parameters()

    model = train_cluster_model(
        X_train,
        y_train,
        parameters,
    )

    result_df = create_prediction_table(
        model,
        X_test,
        y_test,
        similarity_df,
    )

    result_df = assign_similarity_groups(
        result_df
    )

    metrics_df = calculate_group_metrics(
        result_df
    )

    correlation_df = (
        calculate_error_correlation(
            result_df
        )
    )

    ood_df = calculate_ood_metrics(
        result_df
    )

    save_results(
        result_df,
        metrics_df,
        correlation_df,
        ood_df,
    )

    plot_error_vs_similarity(
        result_df
    )

    plot_mae_by_similarity(
        metrics_df
    )

    print(
        "\nSIMILARITY OOD ANALYSIS V2 COMPLETED"
    )


if __name__ == "__main__":
    main()