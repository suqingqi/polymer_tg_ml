from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

BASELINE_FILE = (
    TABLES_DIR
    / "baseline_model_results.csv"
)

CV_FILE = (
    TABLES_DIR
    / "xgboost_cv_summary.csv"
)

SPLIT_FILE = (
    TABLES_DIR
    / "random_vs_cluster_split.csv"
)

OOD_FILE = (
    TABLES_DIR
    / "low_similarity_ood_metrics.csv"
)

SHAP_FILE = (
    TABLES_DIR
    / "shap_feature_importance.csv"
)

CASE_FILE = (
    TABLES_DIR
    / "prediction_case_studies.csv"
)

OUTPUT_PERFORMANCE_FILE = (
    TABLES_DIR
    / "final_model_performance.csv"
)

OUTPUT_METRICS_FILE = (
    TABLES_DIR
    / "final_project_metrics.csv"
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


def check_file(
    file_path,
):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n"
            f"{file_path}"
        )


def load_results():
    print(
        "\nSTEP 1 - LOAD PROJECT RESULTS"
    )

    required_files = [
        BASELINE_FILE,
        CV_FILE,
        SPLIT_FILE,
        OOD_FILE,
        SHAP_FILE,
        CASE_FILE,
    ]

    for file_path in required_files:
        check_file(
            file_path
        )

    baseline_df = pd.read_csv(
        BASELINE_FILE
    )

    cv_df = pd.read_csv(
        CV_FILE
    )

    split_df = pd.read_csv(
        SPLIT_FILE
    )

    ood_df = pd.read_csv(
        OOD_FILE
    )

    shap_df = pd.read_csv(
        SHAP_FILE
    )

    case_df = pd.read_csv(
        CASE_FILE
    )

    print(
        "\nAll project result files loaded."
    )

    return (
        baseline_df,
        cv_df,
        split_df,
        ood_df,
        shap_df,
        case_df,
    )


def get_baseline_row(
    baseline_df,
    model_name,
):
    row = baseline_df[
        baseline_df[
            "Model"
        ] == model_name
    ]

    if len(row) == 0:
        raise ValueError(
            f"Model not found in baseline results: "
            f"{model_name}"
        )

    return row.iloc[
        0
    ]


def get_split_row(
    split_df,
    split_name,
):
    row = split_df[
        split_df[
            "Split"
        ] == split_name
    ]

    if len(row) == 0:
        raise ValueError(
            f"Split not found: "
            f"{split_name}"
        )

    return row.iloc[
        0
    ]


def create_performance_table(
    baseline_df,
    split_df,
    ood_df,
):
    print(
        "\nSTEP 2 - CREATE FINAL PERFORMANCE TABLE"
    )

    rows = []

    linear = get_baseline_row(
        baseline_df,
        "Linear Regression",
    )

    random_forest = get_baseline_row(
        baseline_df,
        "Random Forest",
    )

    baseline_xgb = get_baseline_row(
        baseline_df,
        "XGBoost",
    )

    random_split = get_split_row(
        split_df,
        "Random",
    )

    cluster_split = get_split_row(
        split_df,
        "Cluster-based",
    )

    rows.append(
        {
            "Evaluation":
                "Linear Regression",

            "Evaluation_Type":
                "Random split baseline",

            "R2":
                linear["Test_R2"],

            "MAE_K":
                linear["Test_MAE_K"],

            "RMSE_K":
                linear["Test_RMSE_K"],
        }
    )

    rows.append(
        {
            "Evaluation":
                "Random Forest",

            "Evaluation_Type":
                "Random split baseline",

            "R2":
                random_forest["Test_R2"],

            "MAE_K":
                random_forest["Test_MAE_K"],

            "RMSE_K":
                random_forest["Test_RMSE_K"],
        }
    )

    rows.append(
        {
            "Evaluation":
                "Baseline XGBoost",

            "Evaluation_Type":
                "Random split baseline",

            "R2":
                baseline_xgb["Test_R2"],

            "MAE_K":
                baseline_xgb["Test_MAE_K"],

            "RMSE_K":
                baseline_xgb["Test_RMSE_K"],
        }
    )

    rows.append(
        {
            "Evaluation":
                "Optimized XGBoost",

            "Evaluation_Type":
                "Random split",

            "R2":
                random_split["Test_R2"],

            "MAE_K":
                random_split["Test_MAE_K"],

            "RMSE_K":
                random_split["Test_RMSE_K"],
        }
    )

    rows.append(
        {
            "Evaluation":
                "Cluster-based XGBoost",

            "Evaluation_Type":
                "Chemical-space-aware split",

            "R2":
                cluster_split["Test_R2"],

            "MAE_K":
                cluster_split["Test_MAE_K"],

            "RMSE_K":
                cluster_split["Test_RMSE_K"],
        }
    )

    for _, row in ood_df.iterrows():

        rows.append(
            {
                "Evaluation":
                    row[
                        "OOD_Definition"
                    ],

                "Evaluation_Type":
                    "Low-similarity OOD",

                "R2":
                    row[
                        "R2"
                    ],

                "MAE_K":
                    row[
                        "MAE_K"
                    ],

                "RMSE_K":
                    row[
                        "RMSE_K"
                    ],
            }
        )

    performance_df = pd.DataFrame(
        rows
    )

    print(
        "\nFinal model performance:"
    )

    print(
        performance_df.round(
            4
        ).to_string(
            index=False
        )
    )

    return performance_df


def extract_cv_metrics(
    cv_df,
):
    columns = set(
        cv_df.columns
    )

    if {
        "Metric",
        "Mean",
        "Std",
    }.issubset(
        columns
    ):

        r2_row = cv_df[
            cv_df[
                "Metric"
            ].astype(
                str
            ).str.upper() == "R2"
        ]

        mae_row = cv_df[
            cv_df[
                "Metric"
            ].astype(
                str
            ).str.upper() == "MAE"
        ]

        if len(r2_row) == 0:
            raise ValueError(
                "R2 row not found in CV summary."
            )

        cv_r2_mean = float(
            r2_row.iloc[
                0
            ][
                "Mean"
            ]
        )

        cv_r2_std = float(
            r2_row.iloc[
                0
            ][
                "Std"
            ]
        )

        if len(mae_row) > 0:
            cv_mae_mean = float(
                mae_row.iloc[
                    0
                ][
                    "Mean"
                ]
            )

        else:
            cv_mae_mean = None

        return (
            cv_r2_mean,
            cv_r2_std,
            cv_mae_mean,
        )

    if {
        "R2_Mean",
        "R2_Std",
    }.issubset(
        columns
    ):

        cv_r2_mean = float(
            cv_df.iloc[
                0
            ][
                "R2_Mean"
            ]
        )

        cv_r2_std = float(
            cv_df.iloc[
                0
            ][
                "R2_Std"
            ]
        )

        if (
            "MAE_Mean_K"
            in cv_df.columns
        ):
            cv_mae_mean = float(
                cv_df.iloc[
                    0
                ][
                    "MAE_Mean_K"
                ]
            )

        else:
            cv_mae_mean = None

        return (
            cv_r2_mean,
            cv_r2_std,
            cv_mae_mean,
        )

    raise ValueError(
        "Unknown CV summary format.\n"
        f"Columns found:\n"
        f"{list(cv_df.columns)}"
    )


def create_project_metrics(
    cv_df,
    split_df,
    ood_df,
    shap_df,
):
    print(
        "\nSTEP 3 - CREATE PROJECT METRICS"
    )

    random_split = get_split_row(
        split_df,
        "Random",
    )

    cluster_split = get_split_row(
        split_df,
        "Cluster-based",
    )

    (
        cv_r2_mean,
        cv_r2_std,
        cv_mae_mean,
    ) = extract_cv_metrics(
        cv_df
    )

    top_shap = shap_df.iloc[
        0
    ]

    lowest_ood = ood_df.sort_values(
        "Samples"
    ).iloc[
        0
    ]

    metrics = [
        {
            "Metric":
                "Dataset samples",

            "Value":
                "7367",
        },
        {
            "Metric":
                "Model features",

            "Value":
                "99",
        },
        {
            "Metric":
                "Polymer classes",

            "Value":
                "22",
        },
        {
            "Metric":
                "Random test R2",

            "Value":
                f"{random_split['Test_R2']:.4f}",
        },
        {
            "Metric":
                "Random test MAE",

            "Value":
                f"{random_split['Test_MAE_K']:.2f} K",
        },
        {
            "Metric":
                "5-fold CV R2",

            "Value":
                (
                    f"{cv_r2_mean:.4f} "
                    f"+/- {cv_r2_std:.4f}"
                ),
        },
        {
            "Metric":
                "Cluster test R2",

            "Value":
                f"{cluster_split['Test_R2']:.4f}",
        },
        {
            "Metric":
                "Cluster test MAE",

            "Value":
                f"{cluster_split['Test_MAE_K']:.2f} K",
        },
        {
            "Metric":
                "Lowest similarity OOD",

            "Value":
                lowest_ood[
                    "OOD_Definition"
                ],
        },
        {
            "Metric":
                "Lowest similarity OOD R2",

            "Value":
                f"{lowest_ood['R2']:.4f}",
        },
        {
            "Metric":
                "Lowest similarity OOD MAE",

            "Value":
                f"{lowest_ood['MAE_K']:.2f} K",
        },
        {
            "Metric":
                "Top SHAP feature",

            "Value":
                top_shap[
                    "Feature"
                ],
        },
        {
            "Metric":
                "Top SHAP importance",

            "Value":
                f"{top_shap['Mean_Abs_SHAP']:.4f}",
        },
    ]

    if cv_mae_mean is not None:
        metrics.append(
            {
                "Metric":
                    "5-fold CV MAE",

                "Value":
                    f"{cv_mae_mean:.2f} K",
            }
        )

    metrics_df = pd.DataFrame(
        metrics
    )

    print(
        "\nProject summary metrics:"
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    return metrics_df


def create_case_summary(
    case_df,
):
    print(
        "\nSTEP 4 - CASE STUDY SUMMARY"
    )

    columns = [
        "Reliability",
        "Sample_ID",
        "Experimental_Tg_K",
        "Predicted_Tg_K",
        "Absolute_Error_K",
        "Max_Train_Tanimoto",
    ]

    available_columns = [
        column
        for column in columns
        if column in case_df.columns
    ]

    summary = case_df[
        available_columns
    ].copy()

    print(
        "\nRepresentative prediction cases:"
    )

    print(
        summary.round(
            4
        ).to_string(
            index=False
        )
    )

    return summary


def save_tables(
    performance_df,
    metrics_df,
    case_summary,
):
    print(
        "\nSTEP 5 - SAVE FINAL TABLES"
    )

    performance_df.to_csv(
        OUTPUT_PERFORMANCE_FILE,
        index=False,
    )

    metrics_df.to_csv(
        OUTPUT_METRICS_FILE,
        index=False,
    )

    case_summary.to_csv(
        TABLES_DIR
        / "final_case_studies.csv",
        index=False,
    )

    print(
        "\nPerformance table saved to:"
    )

    print(
        OUTPUT_PERFORMANCE_FILE
    )

    print(
        "\nProject metrics saved to:"
    )

    print(
        OUTPUT_METRICS_FILE
    )


def create_r2_figure(
    performance_df,
):
    print(
        "\nSTEP 6 - CREATE R2 SUMMARY FIGURE"
    )

    selected_names = [
        "Optimized XGBoost",
        "Cluster-based XGBoost",
        "Tanimoto < 0.60",
        "Tanimoto < 0.50",
        "Tanimoto < 0.40",
    ]

    plot_df = performance_df[
        performance_df[
            "Evaluation"
        ].isin(
            selected_names
        )
    ].copy()

    category_order = {
        name: index
        for index, name in enumerate(
            selected_names
        )
    }

    plot_df[
        "Plot_Order"
    ] = plot_df[
        "Evaluation"
    ].map(
        category_order
    )

    plot_df = plot_df.sort_values(
        "Plot_Order"
    )

    plt.figure(
        figsize=(9, 5)
    )

    bars = plt.bar(
        plot_df[
            "Evaluation"
        ],
        plot_df[
            "R2"
        ],
    )

    plt.ylabel(
        "R2"
    )

    plt.xlabel(
        "Evaluation Setting"
    )

    plt.title(
        "Model Performance Across "
        "Chemical-Space Difficulty"
    )

    plt.ylim(
        0,
        1.0,
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    for bar, value in zip(
        bars,
        plot_df[
            "R2"
        ],
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "final_r2_across_evaluation_settings.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nR2 figure saved to:"
    )

    print(
        output_file
    )


def create_mae_figure(
    performance_df,
):
    print(
        "\nSTEP 7 - CREATE MAE SUMMARY FIGURE"
    )

    selected_names = [
        "Optimized XGBoost",
        "Cluster-based XGBoost",
        "Tanimoto < 0.60",
        "Tanimoto < 0.50",
        "Tanimoto < 0.40",
    ]

    plot_df = performance_df[
        performance_df[
            "Evaluation"
        ].isin(
            selected_names
        )
    ].copy()

    category_order = {
        name: index
        for index, name in enumerate(
            selected_names
        )
    }

    plot_df[
        "Plot_Order"
    ] = plot_df[
        "Evaluation"
    ].map(
        category_order
    )

    plot_df = plot_df.sort_values(
        "Plot_Order"
    )

    plt.figure(
        figsize=(9, 5)
    )

    bars = plt.bar(
        plot_df[
            "Evaluation"
        ],
        plot_df[
            "MAE_K"
        ],
    )

    plt.ylabel(
        "MAE (K)"
    )

    plt.xlabel(
        "Evaluation Setting"
    )

    plt.title(
        "Prediction Error Across "
        "Chemical-Space Difficulty"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    maximum_mae = plot_df[
        "MAE_K"
    ].max()

    plt.ylim(
        0,
        maximum_mae * 1.18,
    )

    for bar, value in zip(
        bars,
        plot_df[
            "MAE_K"
        ],
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + maximum_mae * 0.02,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "final_mae_across_evaluation_settings.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nMAE figure saved to:"
    )

    print(
        output_file
    )


def main():
    print(
        "\nPOLYMER Tg PROJECT FINAL SUMMARY"
    )

    create_output_directories()

    (
        baseline_df,
        cv_df,
        split_df,
        ood_df,
        shap_df,
        case_df,
    ) = load_results()

    performance_df = (
        create_performance_table(
            baseline_df,
            split_df,
            ood_df,
        )
    )

    metrics_df = (
        create_project_metrics(
            cv_df,
            split_df,
            ood_df,
            shap_df,
        )
    )

    case_summary = (
        create_case_summary(
            case_df
        )
    )

    save_tables(
        performance_df,
        metrics_df,
        case_summary,
    )

    create_r2_figure(
        performance_df
    )

    create_mae_figure(
        performance_df
    )

    print(
        "\nFINAL PROJECT SUMMARY"
    )

    print(
        "\nDataset:"
    )

    print(
        "7367 polymers"
    )

    print(
        "99 molecular descriptors"
    )

    print(
        "22 polymer classes"
    )

    random_split = get_split_row(
        split_df,
        "Random",
    )

    cluster_split = get_split_row(
        split_df,
        "Cluster-based",
    )

    print(
        "\nOptimized XGBoost:"
    )

    print(
        f"Random test R2: "
        f"{random_split['Test_R2']:.4f}"
    )

    print(
        f"Random test MAE: "
        f"{random_split['Test_MAE_K']:.2f} K"
    )

    print(
        "\nChemical-space-aware evaluation:"
    )

    print(
        f"Cluster test R2: "
        f"{cluster_split['Test_R2']:.4f}"
    )

    print(
        f"Cluster test MAE: "
        f"{cluster_split['Test_MAE_K']:.2f} K"
    )

    lowest_ood = ood_df.sort_values(
        "Samples"
    ).iloc[
        0
    ]

    print(
        "\nLow-similarity OOD:"
    )

    print(
        f"{lowest_ood['OOD_Definition']}"
    )

    print(
        f"R2: "
        f"{lowest_ood['R2']:.4f}"
    )

    print(
        f"MAE: "
        f"{lowest_ood['MAE_K']:.2f} K"
    )

    print(
        "\nPROJECT FINAL SUMMARY COMPLETED"
    )


if __name__ == "__main__":
    main()